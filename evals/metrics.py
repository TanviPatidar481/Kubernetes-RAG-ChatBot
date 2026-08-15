"""
Phase 2 — RAGAS + Tool Correctness metrics.

Uses JUDGE_GROQ for evaluation so production GROQ_API_KEY is not exhausted.
Scores the full captured answer against up to five captured retrieval chunks.
"""

import asyncio
import os

import logfire
import pandas as pd
from openai import AsyncOpenAI
from ragas.embeddings import HuggingFaceEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerCorrectness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
JUDGE_MODEL = "llama-3.1-8b-instant"

# Groq on-demand rate-limit buffer.
GENERAL_BATCH_SIZE = 1
COOLDOWN_MINI = 70
COOLDOWN_STANDARD = 62

# Evaluates substantially the same evidence used by the live responder:
# five retrieved chunks, with a bounded prefix from each.
CONTEXT_TRUNCATE = 2000
CONTEXT_LIMIT = 3


def _build_judge():
    api_key = os.getenv("JUDGE_GROQ") or os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Set JUDGE_GROQ or GROQ_API_KEY before running metrics.")

    client = AsyncOpenAI(api_key=api_key, base_url=GROQ_BASE_URL)

    llm = llm_factory(
        JUDGE_MODEL,
        provider="openai",
        client=client,
        max_tokens=4000,
    )

    embeddings = HuggingFaceEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        use_api=False,
    )

    return llm, embeddings


async def _cooldown(seconds: int, label: str, status_cb=None):
    message = f"⏳ {seconds}s cooldown after {label} (Groq TPM buffer)..."

    if status_cb:
        status_cb(message)

    await asyncio.sleep(seconds)

    if status_cb:
        status_cb("✅ Ready — starting next experiment.")


def _prep_samples(golden_dataset: dict) -> list[dict]:
    """
    Creates immutable evaluation inputs from the Phase 1 snapshot.

    Phase 1 must have captured:
      - the full actual_response
      - the exact actual_contexts returned with that response
    """
    valid_samples = []

    for sample in golden_dataset["rag_samples"]:
        response = (sample.get("actual_response") or "").strip()
        if not response:
            continue

        raw_contexts = (
            sample.get("actual_contexts")
            or sample.get("relevant_contexts")
            or []
        )

        contexts = [
            str(context)[:CONTEXT_TRUNCATE]
            for context in raw_contexts[:CONTEXT_LIMIT]
            if context
        ]

        valid_samples.append(
            {
                **sample,
                "actual_response": response,
                "actual_contexts": contexts,
            }
        )

    return valid_samples


def _score_df(metric_key: str, samples: list[dict], scores) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "question": sample["question"][:65],
                metric_key: round(float(score.value), 3),
            }
            for sample, score in zip(samples, scores)
        ]
    )


async def _batched_score(
    metric,
    inputs: list[dict],
    status_cb=None,
    label: str = "",
) -> list:
    """
    Runs one sample at a time to avoid stacking concurrent Groq requests.
    """
    all_scores = []

    batches = [
        inputs[index:index + GENERAL_BATCH_SIZE]
        for index in range(0, len(inputs), GENERAL_BATCH_SIZE)
    ]

    for batch_index, batch in enumerate(batches):
        if batch_index > 0:
            await _cooldown(
                COOLDOWN_MINI,
                f"{label} sample {batch_index}",
                status_cb,
            )

        scores = await metric.abatch_score(batch)
        all_scores.extend(scores)

    return all_scores


async def run_all_metrics(golden_dataset: dict, status_cb=None) -> dict:
    """
    Runs Faithfulness, Answer Relevancy, Context Precision, Context Recall,
    Answer Correctness, and Tool Correctness.
    """
    judge_llm, ragas_embeddings = _build_judge()
    samples = _prep_samples(golden_dataset)

    if not samples:
        raise ValueError(
            "No samples with actual_response found. Run Phase 1 first."
        )

    results = {}

    with logfire.span(
        "Eval Phase 2 — All Metrics",
        total_samples=len(samples),
    ):
        # Exp 1 — Faithfulness
        if status_cb:
            status_cb(f"🧪 Exp 1/6 — Faithfulness ({len(samples)} samples)...")

        with logfire.span("Exp 1 — Faithfulness"):
            inputs = [
                {
                    "user_input": sample["question"],
                    "response": sample["actual_response"],
                    "retrieved_contexts": sample["actual_contexts"],
                }
                for sample in samples
            ]

            scores = await _batched_score(
                Faithfulness(llm=judge_llm),
                inputs,
                status_cb,
                "Faithfulness",
            )

            df = _score_df("faithfulness", samples, scores)
            results["faithfulness"] = df

            logfire.info(
                "Faithfulness done",
                avg=round(df["faithfulness"].mean(), 3),
            )

        await _cooldown(COOLDOWN_STANDARD, "Faithfulness", status_cb)

        # Exp 2 — Answer Relevancy
        if status_cb:
            status_cb(
                f"🧪 Exp 2/6 — Answer Relevancy ({len(samples)} samples)..."
            )

        with logfire.span("Exp 2 — Answer Relevancy"):
            inputs = [
                {
                    "user_input": sample["question"],
                    "response": sample["actual_response"],
                }
                for sample in samples
            ]

            scores = await _batched_score(
                AnswerRelevancy(
                    llm=judge_llm,
                    embeddings=ragas_embeddings,
                ),
                inputs,
                status_cb,
                "Answer Relevancy",
            )

            df = _score_df("answer_relevancy", samples, scores)
            results["answer_relevancy"] = df

            logfire.info(
                "Answer Relevancy done",
                avg=round(df["answer_relevancy"].mean(), 3),
            )

        await _cooldown(COOLDOWN_STANDARD, "Answer Relevancy", status_cb)

        # Exp 3 — Context Precision
        if status_cb:
            status_cb(
                f"🧪 Exp 3/6 — Context Precision ({len(samples)} samples)..."
            )

        with logfire.span("Exp 3 — Context Precision"):
            inputs = [
                {
                    "user_input": sample["question"],
                    "reference": sample["reference"],
                    "retrieved_contexts": sample["actual_contexts"],
                }
                for sample in samples
            ]

            scores = await _batched_score(
                ContextPrecision(llm=judge_llm),
                inputs,
                status_cb,
                "Context Precision",
            )

            df = _score_df("context_precision", samples, scores)
            results["context_precision"] = df

            logfire.info(
                "Context Precision done",
                avg=round(df["context_precision"].mean(), 3),
            )

        await _cooldown(COOLDOWN_STANDARD, "Context Precision", status_cb)

        # Exp 4 — Context Recall
        if status_cb:
            status_cb(
                f"🧪 Exp 4/6 — Context Recall ({len(samples)} samples)..."
            )

        with logfire.span("Exp 4 — Context Recall"):
            inputs = [
                {
                    "user_input": sample["question"],
                    "reference": sample["reference"],
                    "retrieved_contexts": sample["actual_contexts"],
                }
                for sample in samples
            ]

            scores = await _batched_score(
                ContextRecall(llm=judge_llm),
                inputs,
                status_cb,
                "Context Recall",
            )

            df = _score_df("context_recall", samples, scores)
            results["context_recall"] = df

            logfire.info(
                "Context Recall done",
                avg=round(df["context_recall"].mean(), 3),
            )

        await _cooldown(COOLDOWN_STANDARD, "Context Recall", status_cb)

        # Exp 5 — Answer Correctness
        if status_cb:
            status_cb(
                f"🧪 Exp 5/6 — Answer Correctness ({len(samples)} samples)..."
            )

        with logfire.span("Exp 5 — Answer Correctness"):
            inputs = [
                {
                    "user_input": sample["question"],
                    "response": sample["actual_response"],
                    "reference": sample["reference"],
                }
                for sample in samples
            ]

            scores = await _batched_score(
                AnswerCorrectness(
                    llm=judge_llm,
                    embeddings=ragas_embeddings,
                ),
                inputs,
                status_cb,
                "Answer Correctness",
            )

            df = _score_df("answer_correctness", samples, scores)
            results["answer_correctness"] = df

            logfire.info(
                "Answer Correctness done",
                avg=round(df["answer_correctness"].mean(), 3),
            )

        await _cooldown(COOLDOWN_STANDARD, "Answer Correctness", status_cb)

        # Exp 6 — Tool Correctness
        if status_cb:
            status_cb("⚡ Exp 6/6 — Tool Correctness (no LLM calls)...")

        with logfire.span("Exp 6 — Tool Correctness"):
            tool_rows = []

            for sample in samples:
                called = set(sample.get("actual_tools_called") or [])
                expected = set(sample.get("expected_tools") or [])

                union = called | expected
                score = len(called & expected) / len(union) if union else 0.0

                tool_rows.append(
                    {
                        "question": sample["question"][:65],
                        "tool_correctness": round(score, 3),
                    }
                )

            df = pd.DataFrame(tool_rows)
            results["tool_correctness"] = df

            logfire.info(
                "Tool Correctness done",
                avg=round(df["tool_correctness"].mean(), 3),
            )

        if status_cb:
            status_cb("✅ All 6 experiments complete!")

    return results
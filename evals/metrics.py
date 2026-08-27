"""
Phase 2 — RAGAS + Tool Correctness metrics.

Uses JUDGE_GEMINI (gemini-3.5-flash-lite via the Gemini OpenAI-compatible API)
for evaluation so the production LLM keys are not exhausted.
Scores the full captured answer against up to five captured retrieval chunks.
"""

import asyncio
import os

import logfire
import pandas as pd
from openai import AsyncOpenAI
from ragas.cache import DiskCacheBackend
from ragas.embeddings import HuggingFaceEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.result import MetricResult
from ragas.metrics.collections import (
    AnswerCorrectness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

# GROQ_BASE_URL = "https://api.groq.com/openai/v1"
# JUDGE_MODEL = "openai/gpt-oss-120b"

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
JUDGE_MODEL = "gemini-3.5-flash-lite"

# Groq on-demand rate-limit buffer.
GENERAL_BATCH_SIZE = 1
COOLDOWN_MINI = 70
COOLDOWN_STANDARD = 62

# Evaluates substantially the same evidence used by the live responder:
# five retrieved chunks, with a bounded prefix from each.
CONTEXT_TRUNCATE = 2000
CONTEXT_LIMIT = 5

# Recovery pacing for transient judge-provider failures (Gemini answers
# capacity spikes with 503 UNAVAILABLE / "high demand"). A failed sample is
# retried after RECOVERY_WAIT_SECONDS, at most RECOVERY_MAX_RETRIES extra
# times; after that it is recorded as NaN so the run never crashes on one
# unavailable judge request.
RECOVERY_WAIT_SECONDS = 90
RECOVERY_MAX_RETRIES = 2


def _build_judge():
    # api_key = os.getenv("JUDGE_GROQ") or os.getenv("GROQ_API_KEY")
    # if not api_key:
    #     raise ValueError("Set JUDGE_GROQ or GROQ_API_KEY before running metrics.")

    # client = AsyncOpenAI(api_key=api_key, base_url=GROQ_BASE_URL)
    api_key = os.getenv("JUDGE_GEMINI") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Set JUDGE_GEMINI or GEMINI_API_KEY before running metrics.")

    # Gemini returns transient 503 UNAVAILABLE during demand spikes. The
    # OpenAI SDK retries those automatically with exponential backoff, so
    # give it enough attempts/time to ride out short spikes.
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=GEMINI_BASE_URL,
        timeout=120,
        max_retries=8,
    )

    llm = llm_factory(
        JUDGE_MODEL,
        provider="openai",
        client=client,
        max_tokens=6000,
        # Persistent disk cache: re-running after an interruption does not
        # re-spend tokens on already-scored prompt/model pairs.
        cache=DiskCacheBackend(),
    )

    embeddings = HuggingFaceEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        use_api=False,
    )

    return llm, embeddings


async def _cooldown(seconds: int, label: str, status_cb=None):
    message = f"⏳ {seconds}s cooldown after {label} (Gemini rate-limit buffer)..."

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
        # response = response[:300]

        # raw_contexts = (
        #     sample.get("actual_contexts")
        #     or sample.get("relevant_contexts")
        #     or []
        # )

        raw_contexts = sample.get("actual_contexts", [])

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


def _samples_with_contexts(
    samples: list[dict], label: str = ""
) -> tuple[list[dict], int]:
    """
    Returns (filtered, n_skipped) where filtered only contains samples with at
    least one retrieved context.

    Metrics that judge the retrieved evidence (Faithfulness, Context Precision,
    Context Recall) cannot score samples whose retrieval returned nothing, and
    RAGAS raises ValueError for empty retrieved_contexts. Those samples are
    excluded and reported so retrieval failures stay visible instead of
    crashing the run.
    """
    filtered = [s for s in samples if s.get("actual_contexts")]
    skipped = len(samples) - len(filtered)
    if skipped:
        logfire.info(
            f"{label}: skipped {skipped} sample(s) with empty retrieved contexts",
            skipped_ids=[s.get("id") for s in samples if not s.get("actual_contexts")],
        )
    return filtered, skipped


def _is_transient_unavailable(exc: BaseException) -> bool:
    """
    True when the exception chain looks like a judge-provider capacity error
    (HTTP 503 / UNAVAILABLE / "high demand" / overloaded), i.e. worth waiting
    out and retrying instead of aborting the evaluation.
    """
    seen = set()
    current = exc

    while current is not None and id(current) not in seen:
        seen.add(id(current))

        if getattr(current, "status_code", None) == 503:
            return True

        text = f"{type(current).__name__}: {current}".lower()
        if any(
            marker in text
            for marker in (
                "503",
                "service unavailable",
                "unavailable",
                "high demand",
                "overloaded",
            )
        ):
            return True

        current = current.__cause__ or current.__context__

    return False


async def _score_with_recovery(
    metric,
    batch: list[dict],
    batch_index: int,
    label: str,
    status_cb=None,
) -> list:
    """
    Scores one single-sample batch with bounded retry for transient 503s.

    - Waits RECOVERY_WAIT_SECONDS between attempts.
    - Tries at most 1 + RECOVERY_MAX_RETRIES times (never indefinitely).
    - Non-transient errors are re-raised so real bugs still fail loudly.
    - If a transient failure survives every attempt, records NaN for that
      sample instead of crashing the whole run.
    """
    for attempt in range(1 + RECOVERY_MAX_RETRIES):
        try:
            return await metric.abatch_score(batch)
        except Exception as exc:
            if not _is_transient_unavailable(exc):
                raise

            attempts_left = RECOVERY_MAX_RETRIES - attempt

            if attempts_left == 0:
                logfire.warning(
                    f"{label}: sample {batch_index} still unavailable after "
                    f"{1 + RECOVERY_MAX_RETRIES} attempts — recorded as blank.",
                    error=str(exc)[:300],
                )
                if status_cb:
                    status_cb(
                        f"⚠️ {label}: sample stayed unavailable after retries "
                        f"— recorded as blank, continuing."
                    )
                return [MetricResult(value=float("nan"))]

            logfire.warning(
                f"{label}: sample {batch_index} hit a transient capacity "
                f"(503) error — retrying in {RECOVERY_WAIT_SECONDS}s "
                f"({attempts_left} attempt(s) left).",
                error=str(exc)[:300],
            )
            if status_cb:
                status_cb(
                    f"⚠️ Judge 503 (high demand) — waiting {RECOVERY_WAIT_SECONDS}s "
                    f"before retrying {label} sample "
                    f"({attempts_left} attempt(s) left)..."
                )
            await asyncio.sleep(RECOVERY_WAIT_SECONDS)

    return [MetricResult(value=float("nan"))]


async def _batched_score(
    metric,
    inputs: list[dict],
    status_cb=None,
    label: str = "",
) -> list:
    """
    Runs one sample at a time to avoid stacking concurrent judge requests.

    Each single-sample batch goes through _score_with_recovery, so transient
    Gemini 503s are retried with a pause and permanently unavailable samples
    are recorded as NaN rather than crashing the run.
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

        scores = await _score_with_recovery(
            metric,
            batch,
            batch_index,
            label,
            status_cb,
        )
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
        with logfire.span("Exp 1 — Faithfulness"):
            ctx_samples, n_skipped = _samples_with_contexts(
                samples, "Exp 1 — Faithfulness"
            )

            if status_cb:
                status_cb(
                    f"🧪 Exp 1/6 — Faithfulness "
                    f"({len(ctx_samples)} scored, {n_skipped} skipped — no retrieved context)..."
                )

            if ctx_samples:
                inputs = [
                    {
                        "user_input": sample["question"],
                        "response": sample["actual_response"],
                        "retrieved_contexts": sample["actual_contexts"],
                    }
                    for sample in ctx_samples
                ]

                scores = await _batched_score(
                    Faithfulness(llm=judge_llm),
                    inputs,
                    status_cb,
                    "Faithfulness",
                )

                df = _score_df("faithfulness", ctx_samples, scores)
                results["faithfulness"] = df

                logfire.info(
                    "Faithfulness done",
                    avg=round(df["faithfulness"].mean(), 3),
                    n_scored=len(ctx_samples),
                    n_skipped=n_skipped,
                )
            else:
                logfire.warning(
                    "Faithfulness skipped — all samples have empty retrieved contexts."
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
        with logfire.span("Exp 3 — Context Precision"):
            ctx_samples, n_skipped = _samples_with_contexts(
                samples, "Exp 3 — Context Precision"
            )

            if status_cb:
                status_cb(
                    f"🧪 Exp 3/6 — Context Precision "
                    f"({len(ctx_samples)} scored, {n_skipped} skipped — no retrieved context)..."
                )

            if ctx_samples:
                inputs = [
                    {
                        "user_input": sample["question"],
                        "reference": sample["reference"],
                        "retrieved_contexts": sample["actual_contexts"],
                    }
                    for sample in ctx_samples
                ]

                scores = await _batched_score(
                    ContextPrecision(llm=judge_llm),
                    inputs,
                    status_cb,
                    "Context Precision",
                )

                df = _score_df("context_precision", ctx_samples, scores)
                results["context_precision"] = df

                logfire.info(
                    "Context Precision done",
                    avg=round(df["context_precision"].mean(), 3),
                    n_scored=len(ctx_samples),
                    n_skipped=n_skipped,
                )
            else:
                logfire.warning(
                    "Context Precision skipped — all samples have empty retrieved contexts."
                )

        await _cooldown(COOLDOWN_STANDARD, "Context Precision", status_cb)

        # Exp 4 — Context Recall
        with logfire.span("Exp 4 — Context Recall"):
            ctx_samples, n_skipped = _samples_with_contexts(
                samples, "Exp 4 — Context Recall"
            )

            if status_cb:
                status_cb(
                    f"🧪 Exp 4/6 — Context Recall "
                    f"({len(ctx_samples)} scored, {n_skipped} skipped — no retrieved context)..."
                )

            if ctx_samples:
                inputs = [
                    {
                        "user_input": sample["question"],
                        "reference": sample["reference"],
                        "retrieved_contexts": sample["actual_contexts"],
                    }
                    for sample in ctx_samples
                ]

                scores = await _batched_score(
                    ContextRecall(llm=judge_llm),
                    inputs,
                    status_cb,
                    "Context Recall",
                )

                df = _score_df("context_recall", ctx_samples, scores)
                results["context_recall"] = df

                logfire.info(
                    "Context Recall done",
                    avg=round(df["context_recall"].mean(), 3),
                    n_scored=len(ctx_samples),
                    n_skipped=n_skipped,
                )
            else:
                logfire.warning(
                    "Context Recall skipped — all samples have empty retrieved contexts."
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
# ─────────────────────────────────────────────────────────────────────────────
# CRITICAL: logfire must be configured before all other imports
# ─────────────────────────────────────────────────────────────────────────────
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import logfire
logfire.configure(token=os.getenv("LOGFIRE_TOKEN"), service_name="evals")

# ─────────────────────────────────────────────────────────────────────────────
import asyncio
import json
import re
import nest_asyncio
import pandas as pd
import streamlit as st

nest_asyncio.apply()

from evals.pipeline import run_pipeline, load_golden_dataset
from evals.guardrails_eval import run_guardrails_eval, compute_guardrails_metrics
from evals.metrics import run_all_metrics

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Agentic RAG — Evaluation Console",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
SCORE_COLORS = {
    "green":  "#d4edda",
    "yellow": "#fff3cd",
    "red":    "#f8d7da",
}

# -----------------------------------------------------------------------------
# Lucide-style outline icon set (inline SVG, dependency-free).
# Each value is one 24x24 stroked icon. Drawn at a consistent stroke width
# and coloured via currentColor so the family reads as uniform outlines —
# never as emoji glyphs.
# -----------------------------------------------------------------------------
_ICON_PATHS: dict[str, str] = {
    # Authentic Lucide outline paths (ISC-licensed stroke geometry).
    "clipboard-list": (
        "<rect width='8' height='4' x='8' y='2' rx='1' ry='1'/>"
        "<path d='M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2'/>"
        "<path d='M12 11h4'/>"
        "<path d='M12 16h4'/>"
        "<path d='M8 11h.01'/>"
        "<path d='M8 16h.01'/>"
    ),
    "workflow": (
        "<rect width='8' height='8' x='3' y='3' rx='2'/>"
        "<path d='M7 11v4a2 2 0 0 0 2 2h4'/>"
        "<rect width='8' height='8' x='13' y='13' rx='2'/>"
    ),
    "chart-no-axes-combined": (
        "<path d='M12 16v5'/>"
        "<path d='M16 14v7'/>"
        "<path d='M20 10v11'/>"
        "<path d='m22 3-8.646 8.646a.5.5 0 0 1-.708 0L9.354 8.354a.5.5 0 0 0-.708 0L2 15'/>"
        "<path d='M4 18v3'/>"
        "<path d='M8 14v7'/>"
    ),
    "shield-check": (
        "<path d='M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z'/>"
        "<path d='m9 12 2 2 4-4'/>"
    ),
    "play": "<polygon points='6 3 20 12 6 21 6 3'/>",
    "refresh-cw": (
        "<path d='M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8'/>"
        "<path d='M21 3v5h-5'/>"
        "<path d='M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16'/>"
        "<path d='M8 16H3v5'/>"
    ),
    "circle-check": (
        "<path d='M21.801 10A10 10 0 1 1 17 3.335'/>"
        "<path d='m9 11 3 3L22 4'/>"
    ),
    "circle-alert": (
        "<circle cx='12' cy='12' r='10'/>"
        "<line x1='12' x2='12' y1='8' y2='12'/>"
        "<line x1='12' x2='12.01' y1='16' y2='16'/>"
    ),
    "circle-x": (
        "<circle cx='12' cy='12' r='10'/>"
        "<path d='m15 9-6 6'/>"
        "<path d='m9 9 6 6'/>"
    ),
    "info": (
        "<circle cx='12' cy='12' r='10'/>"
        "<path d='M12 16v-4'/>"
        "<path d='M12 8h.01'/>"
    ),
    "save": (
        "<path d='M15.2 3a2 2 0 0 1 1.4.6l3.8 3.8a2 2 0 0 1 .6 1.4V19a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2z'/>"
        "<path d='M17 21v-7a1 1 0 0 0-1-1H8a1 1 0 0 0-1 1v7'/>"
        "<path d='M7 3v4a1 1 0 0 0 1 1h7'/>"
    ),
    "search": (
        "<circle cx='11' cy='11' r='8'/>"
        "<path d='m21 21-4.3-4.3'/>"
    ),
    "file-text": (
        "<path d='M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z'/>"
        "<path d='M14 2v4a2 2 0 0 0 2 2h4'/>"
        "<path d='M10 9H8'/>"
        "<path d='M16 13H8'/>"
        "<path d='M16 17H8'/>"
    ),
    "settings": (
        "<path d='M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z'/>"
        "<circle cx='12' cy='12' r='3'/>"
    ),
}


_TONE_LABEL = {"good": "Good", "fair": "Fair", "poor": "Poor"}
_TONE_ICON = {"good": "circle-check", "fair": "circle-alert", "poor": "circle-x"}


def render_icon(name: str, size: int = 15) -> str:
    """Inline SVG icon (Lucide-style outline). Coloured via currentColor."""
    body = _ICON_PATHS.get(name, "")
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" '
        f'stroke-linecap="round" stroke-linejoin="round">{body}</svg>'
    )


def _tone(score: float) -> str:
    if score >= 0.75:
        return "good"
    if score >= 0.5:
        return "fair"
    return "poor"


def _grade(score: float) -> str:
    return _TONE_LABEL[_tone(score)]


def render_status_badge(label: str, tone: str = "good", icon_name: str | None = None) -> str:
    """Compact pill badge: small outline icon + status text."""
    icon = icon_name or _TONE_ICON.get(tone, "circle-check")
    css_tone = tone if tone in {"good", "fair", "poor", "neutral"} else "neutral"
    return (
        f'<span class="kv-badge kv-{css_tone}">{render_icon(icon)}'
        f"<span>{label}</span></span>"
    )


def render_metric_status(score: float) -> str:
    return render_status_badge(_TONE_LABEL[_tone(score)], _tone(score))


def _section_heading(icon_name: str, label: str) -> None:
    st.markdown(
        f'<div class="kv-heading">{render_icon(icon_name, 16)} '
        f"<span>{label}</span></div>",
        unsafe_allow_html=True,
    )


def _score_color(val):
    if not isinstance(val, (int, float)):
        return ""
    if val >= 0.75:
        return f"background-color: {SCORE_COLORS['green']}"
    elif val >= 0.5:
        return f"background-color: {SCORE_COLORS['yellow']}"
    return f"background-color: {SCORE_COLORS['red']}"


_STATUS_BG = {
    "BLOCKED": "#e6f4ec", "PASSED": "#e6f4ec",
    "FALSE POSITIVE": "#fdf3d7", "MISSED": "#fdecea",
    "Block": "#fdecea", "Pass": "#e6f4ec",
    "Collected": "#e6f4ec", "Failed": "#fdecea",
}


def _status_color(value):
    """Subtle, muted cell tint for table status columns."""
    bg = _STATUS_BG.get(str(value))
    return f"background-color: {bg}" if bg else ""


# Strip status-symbol emoji from messages that arrive from other eval modules
# (progress/cooldown text in run_all_metrics) so none reach the console.
_STATUS_SYMBOL_RE = re.compile(
    "[\U0001F000-\U0001FFFF\u2600-\u29FF\u2B00-\u2BFF\u2700-\u27BF\ufe0f]+"
)


def _strip_emojis(text: str) -> str:
    return _STATUS_SYMBOL_RE.sub("", text or "").strip()


# Professional console styling — compact badge pills + subtle heading rows.
_EVAL_CSS = """
<style>
 .kv-badge { display:inline-flex; align-items:center; gap:5px;
    padding:2px 9px; border-radius:999px; font-size:0.73rem; font-weight:650;
    letter-spacing:0.04em; line-height:1.2; border:1px solid transparent; }
 .kv-badge.kv-good { color:#166534; background:#e6f4ec; border-color:#b7e0c4; }
 .kv-badge.kv-fair { color:#854a09; background:#fdf3d7; border-color:#eccf9d; }
 .kv-badge.kv-poor { color:#981b1b; background:#fdecea; border-color:#f2c2ba; }
 .kv-badge.kv-neutral { color:#3f4c5c; background:#eef1f5; border-color:#d8dee6; }
 .kv-heading { display:flex; align-items:center; gap:7px; font-weight:680;
    font-size:1.02rem; letter-spacing:-0.01em; margin:0.2rem 0 0.1rem; }
 .kv-metric-head { display:flex; align-items:center; gap:8px; margin:0 0 8px; }
 .kv-metric-title { font-weight:640; }
 .kv-metric-score { font-weight:700; }

   /* Primary buttons */
[data-testid="stTabPanel"] [data-testid="stBaseButton-primary"],
[data-testid="stTabPanel"] [data-testid="stBaseButton-primary"]:focus-visible {
    background-color: #2563EB !important;
    border-color: #2563EB !important;
    color: #FFFFFF !important;
}

/* Disabled Run Live Pipeline */
[data-testid="stTabPanel"] [data-testid="stBaseButton-primary"]:disabled {
    background-color: #2563EB !important;
    border-color: #2563EB !important;
    color: #FFFFFF !important;
    opacity: 1 !important;
}

/* Hover */
[data-testid="stTabPanel"] [data-testid="stBaseButton-primary"]:hover:not(:disabled) {
    background-color: #1D4ED8 !important;
    border-color: #1D4ED8 !important;
    color: #FFFFFF !important;
}
</style>
"""


def _render_metric_table(df: pd.DataFrame, metric_col: str, title: str):
    avg = df[metric_col].mean()
    st.markdown(
        '<div class="kv-metric-head">'
        + render_icon("chart-no-axes-combined", 16)
        + f'<span class="kv-metric-title">{title}</span>'
        + f'<span class="kv-metric-score">{avg:.2f}</span>'
        + render_metric_status(avg)
        + "</div>",
        unsafe_allow_html=True,
    )
    styled = df.style.map(_score_color, subset=[metric_col]).format({metric_col: "{:.3f}"})
    st.dataframe(styled, use_container_width=True, hide_index=True)


def _run_async(coro):
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coro)


# ─────────────────────────────────────────────────────────────────────────────
# Session state init
# ─────────────────────────────────────────────────────────────────────────────
if "golden" not in st.session_state:
    st.session_state.golden = load_golden_dataset()
if "pipeline_done" not in st.session_state:
    st.session_state.pipeline_done = False
if "enriched_dataset" not in st.session_state:
    try:
        with open("evals/enriched_dataset.json", "r", encoding="utf-8") as f:
            st.session_state.enriched_dataset = json.load(f)
            st.session_state.pipeline_done = True
    except FileNotFoundError:
        st.session_state.enriched_dataset = None
if "guardrails_results" not in st.session_state:
    st.session_state.guardrails_results = None
if "metric_results" not in st.session_state:
    st.session_state.metric_results = None
if "pipeline_rows" not in st.session_state:
    st.session_state.pipeline_rows = []

golden = st.session_state.golden

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.title("Agentic RAG — Evaluation Console")
st.caption(
    "01 Review dataset → 02 Run live pipeline → 03 Score responses"
)
st.markdown(_EVAL_CSS, unsafe_allow_html=True)
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(
    ["01 · Ground Truth", "02 · Live Pipeline", "03 · Evaluation"]
)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — Ground Truth
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    _section_heading("clipboard-list", "Evaluation Dataset")
    st.markdown(
        "These are the **golden Q&A pairs** built by parsing your real enterprise documents. "
        "Each entry has a question, a reference answer (ground truth), and the expected tool the RAG agent should call."
    )

    rag_rows = []
    for s in golden["rag_samples"]:
        rag_rows.append({
            "ID": s["id"],
            "Domain": s["domain"].replace("_", " ").title(),
            "Question": s["question"],
            "Reference Answer": s["reference"][:120] + "..." if len(s["reference"]) > 120 else s["reference"],
            "Expected Tool": s["expected_tools"][0] if s["expected_tools"] else "—",
        })
    df_golden = pd.DataFrame(rag_rows)
    st.dataframe(df_golden, use_container_width=True, hide_index=True)
    st.caption(f"{len(rag_rows)} golden RAG samples from 5 enterprise docs")

    st.divider()

    _section_heading("shield-check", "Guardrails Test Cases")
    st.markdown(
        "These inputs test whether the safety rails correctly **block adversarial inputs** "
        "and **let through legitimate questions**."
    )

    g_rows = []
    for g in golden["guardrails_samples"]:
        expected_label = "Block" if g["expected_blocked"] else "Pass"
        g_rows.append({
            "ID": g["id"],
            "Input": g["input"],
            "Expected": expected_label,
            "Type": g["type"],
            "Description": g["description"],
        })
    st.dataframe(
        pd.DataFrame(g_rows).style.map(_status_color, subset=["Expected"]),
        use_container_width=True,
        hide_index=True,
    )
    st.caption("6 guardrails test cases: 3 adversarial (should block) + 3 legit (should pass)")

    with st.expander("View raw golden_dataset.json"):
        st.json(golden)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — Live Pipeline
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    _section_heading("workflow", "Live Evaluation Pipeline")
    st.markdown(
        "Sends each golden question to your **running FastAPI app** (`localhost:8000/query`). "
        "Captures the actual response, retrieved contexts, and tool called. "
        "Responses are truncated to 300 chars to save tokens for the RAGAS judging step."
    )
    st.info(
        "Make sure your FastAPI backend is running first: `uvicorn app.main:app --reload --port 8000`"
    )

    col_p1, col_p2, col_p3 = st.columns([1, 1, 2])
    run_pipeline_btn = col_p1.button(
        "Run Live Pipeline",
        type="primary",
        width="stretch",
        disabled=st.session_state.pipeline_done,
    )
    reset_btn = col_p2.button(
        "Reset & Re-run",
        width="stretch",
        disabled=not st.session_state.pipeline_done,
    )

    if reset_btn:
        st.session_state.pipeline_done = False
        st.session_state.enriched_dataset = None
        st.session_state.guardrails_results = None
        st.session_state.metric_results = None
        st.session_state.pipeline_rows = []
        st.rerun()

    if run_pipeline_btn:
        st.session_state.pipeline_rows = []
        progress_bar = st.progress(0, text="Starting pipeline...")
        live_table_slot = st.empty()
        status_slot = st.empty()

        def pipeline_cb(i, total, question, stage, response=""):
            pct = int((i / total) * 100)
            if stage == "calling":
                progress_bar.progress(pct, text=f"[{i+1}/{total}] Calling /query: {question[:60]}...")
            else:
                short_q = question[:55] + "..." if len(question) > 55 else question
                short_r = response[:80] + "..." if len(response) > 80 else response
                st.session_state.pipeline_rows.append({
                    "#": i + 1,
                    "Question": short_q,
                    "Live Response (truncated)": short_r if short_r else "No response",
                    "Status": "Collected" if short_r else "Failed",
                })
                live_table_slot.dataframe(
                    pd.DataFrame(st.session_state.pipeline_rows),
                    use_container_width=True,
                    hide_index=True,
                )
                progress_bar.progress(
                    int(((i + 1) / total) * 100),
                    text=f"[{i+1}/{total}] Complete",
                )

        with logfire.span("Streamlit — Run Pipeline Button"):
            enriched = run_pipeline(golden, progress_callback=pipeline_cb)
            st.session_state.enriched_dataset = enriched
            # Save the enriched dataset so Step 3 can be run later without rerunning Step 2
            with open("evals/enriched_dataset.json", "w", encoding="utf-8") as f:
                json.dump(enriched, f, indent=2, ensure_ascii=False)
        progress_bar.progress(100, text="All responses collected.")
        status_slot.success(f"Saved {len(enriched['rag_samples'])} responses to the session.")

        # ── Guardrails tests ──────────────────────────────────────────────────
        st.divider()
        _section_heading("shield-check", "Guardrails Tests")
        g_progress = st.progress(0, text="Running guardrails tests...")
        g_status_slot = st.empty()

        def g_cb(i, total, input_text):
            g_progress.progress(
                int((i / total) * 100),
                text=f"[{i+1}/{total}] Testing: {input_text[:60]}...",
            )

        with logfire.span("Streamlit — Guardrails Tests"):
            g_results = run_guardrails_eval(enriched["guardrails_samples"], progress_callback=g_cb)
            g_metrics = compute_guardrails_metrics(g_results)
            st.session_state.guardrails_results = g_results
            st.session_state.pipeline_done = True

        g_progress.progress(100, text="Guardrails tests complete.")

        g_rows_live = []
        for r in g_results:
            result_label = {
                "TP": "BLOCKED", "TN": "PASSED",
                "FP": "FALSE POSITIVE", "FN": "MISSED",
            }.get(r["result"], r["result"].upper())
            g_rows_live.append({
                "ID": r["id"],
                "Input": r["input"][:70],
                "Expected": "Block" if r["expected_blocked"] else "Pass",
                "Actual": "Blocked" if r["actual_blocked"] else "Passed",
                "Result": result_label,
            })
        st.dataframe(
            pd.DataFrame(g_rows_live).style.map(_status_color, subset=["Result"]),
            use_container_width=True,
            hide_index=True,
        )

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Correct", f"{g_metrics['correct']}/{g_metrics['total']}")
        mc2.metric("Precision", f"{g_metrics['precision']:.2f}")
        mc3.metric("Recall", f"{g_metrics['recall']:.2f}")
        mc4.metric("Accuracy", f"{g_metrics['accuracy']:.2f}")

    elif st.session_state.pipeline_done:
        st.success("Pipeline results are available below.")

        resp_rows = []
        for s in st.session_state.enriched_dataset["rag_samples"]:
            resp_rows.append({
                "#": s["id"],
                "Domain": s["domain"].replace("_", " ").title(),
                "Question": s["question"][:60],
                "Live Response": s["actual_response"][:100] + "..." if len(s.get("actual_response","")) > 100 else s.get("actual_response",""),
                "Tool Called": s["actual_tools_called"][0] if s.get("actual_tools_called") else "—",
                "Contexts Retrieved": len(s.get("actual_contexts", [])),
            })
        st.dataframe(pd.DataFrame(resp_rows), use_container_width=True, hide_index=True)

        if st.session_state.guardrails_results:
            st.divider()
            _section_heading("shield-check", "Guardrails Results — Previous Run")
            g_rows_prev = []
            for r in st.session_state.guardrails_results:
                result_label = {
                    "TP": "BLOCKED", "TN": "PASSED",
                    "FP": "FALSE POSITIVE", "FN": "MISSED",
                }.get(r["result"], r["result"].upper())
                g_rows_prev.append({
                    "ID": r["id"],
                    "Input": r["input"][:70],
                    "Result": result_label,
                })
            st.dataframe(
                pd.DataFrame(g_rows_prev).style.map(_status_color, subset=["Result"]),
                use_container_width=True,
                hide_index=True,
            )
            gm = compute_guardrails_metrics(st.session_state.guardrails_results)
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Correct", f"{gm['correct']}/{gm['total']}")
            mc2.metric("Precision", f"{gm['precision']:.2f}")
            mc3.metric("Recall", f"{gm['recall']:.2f}")
            mc4.metric("Accuracy", f"{gm['accuracy']:.2f}")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — Eval Metrics
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    _section_heading("chart-no-axes-combined", "Evaluation Metrics")

    if not st.session_state.pipeline_done:
        st.warning("Complete Step 2 (Live Pipeline) first to collect responses.")
    else:
        st.markdown(
            "Runs all **6 metric experiments** on the stored responses. "
            "LLM-based metrics use the `JUDGE_GEMINI` key (`gemini-3.5-flash-lite`) — "
            "samples are scored one at a time with **70 s between samples** and "
            "**62 s between experiments**. Transient judge 503s are retried "
            "automatically (up to 2 recovery attempts per sample)."
        )
        st.info(
            "Token key used: `JUDGE_GEMINI` (separate from production keys). "
            "Judge responses are cached on disk, so re-running after an interruption "
            "does not re-spend tokens on already-scored samples."
        )

        run_metrics_btn = st.button(
            "Run Evaluation",
            type="primary",
            disabled=not st.session_state.pipeline_done,
        )

        if run_metrics_btn:
            status_slot = st.empty()
            results_slots = {}

            metric_display_names = {
                "faithfulness":      "Exp 1 — Faithfulness",
                "answer_relevancy":  "Exp 2 — Answer Relevancy",
                "context_precision": "Exp 3 — Context Precision",
                "context_recall":    "Exp 4 — Context Recall",
                "answer_correctness":"Exp 5 — Answer Correctness",
                "tool_correctness":  "Exp 6 — Tool Correctness",
            }
            for key, title in metric_display_names.items():
                results_slots[key] = st.empty()

            def status_cb(msg: str):
                status_slot.info(_strip_emojis(msg))

            with logfire.span("Streamlit — Run Metrics Button"):
                metric_results = _run_async(
                    run_all_metrics(st.session_state.enriched_dataset, status_cb=status_cb)
                )
                st.session_state.metric_results = metric_results

            status_slot.success("All 6 experiments complete.")

            for key, title in metric_display_names.items():
                if key in metric_results:
                    with results_slots[key].container():
                        _render_metric_table(metric_results[key], key, title)

        elif st.session_state.metric_results:
            st.success("Metrics already computed — showing results below.")
            metric_display_names = {
                "faithfulness":      "Exp 1 — Faithfulness",
                "answer_relevancy":  "Exp 2 — Answer Relevancy",
                "context_precision": "Exp 3 — Context Precision",
                "context_recall":    "Exp 4 — Context Recall",
                "answer_correctness":"Exp 5 — Answer Correctness",
                "tool_correctness":  "Exp 6 — Tool Correctness",
            }
            for key, title in metric_display_names.items():
                if key in st.session_state.metric_results:
                    _render_metric_table(st.session_state.metric_results[key], key, title)

        # ── Final Summary ─────────────────────────────────────────────────────
        if st.session_state.metric_results:
            st.divider()
            _section_heading("chart-no-axes-combined", "Evaluation Summary")

            mr = st.session_state.metric_results
            summary = [
                ("Faithfulness",       mr.get("faithfulness",      pd.DataFrame()).get("faithfulness",      pd.Series()).mean()),
                ("Answer Relevancy",   mr.get("answer_relevancy",  pd.DataFrame()).get("answer_relevancy",  pd.Series()).mean()),
                ("Context Precision",  mr.get("context_precision", pd.DataFrame()).get("context_precision", pd.Series()).mean()),
                ("Context Recall",     mr.get("context_recall",    pd.DataFrame()).get("context_recall",    pd.Series()).mean()),
                ("Answer Correctness", mr.get("answer_correctness",pd.DataFrame()).get("answer_correctness",pd.Series()).mean()),
                ("Tool Correctness",   mr.get("tool_correctness",  pd.DataFrame()).get("tool_correctness",  pd.Series()).mean()),
            ]

            cols = st.columns(len(summary))
            for col, (name, score) in zip(cols, summary):
                if pd.notna(score):
                    col.metric(
                        label=name,
                        value=f"{score:.2f}",
                        delta=_grade(score),
                    )

            if st.session_state.guardrails_results:
                gm = compute_guardrails_metrics(st.session_state.guardrails_results)
                st.metric(
                    label="Guardrails Accuracy",
                    value=f"{gm['correct']}/{gm['total']}",
                    delta=f"Precision {gm['precision']:.2f} | Recall {gm['recall']:.2f}",
                )

            summary_df = pd.DataFrame([
                {"Metric": name, "Score": f"{score:.3f}" if pd.notna(score) else "—", "Grade": _grade(score) if pd.notna(score) else "—"}
                for name, score in summary
            ])
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
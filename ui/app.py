import os
import base64
import time
import uuid
from urllib.parse import quote

import streamlit as st
import requests
import logfire
from dotenv import load_dotenv


# Load environment variables explicitly from the root directory
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
load_dotenv(dotenv_path=env_path)


# --- LOGFIRE ---
try:
    _token = os.getenv("LOGFIRE_TOKEN")
    if not _token:
        print("ERROR: LOGFIRE_TOKEN is empty or None!")
    logfire.configure(token=_token)
    LOGFIRE_STATUS = "Connected & tracing"
except Exception as _e:
    print(f"Logfire Init Error in UI: {_e}")
    LOGFIRE_STATUS = f"Standby (Error: {_e})"


# --- PAGE CONFIG ---
st.set_page_config(page_title="Agent Core", layout="wide")


def _svg_data_uri(svg: str) -> str:
    """Encode an SVG string as a base64 data URI so it renders as an image
    offline, without depending on an external icon font."""
    encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


# --- CHAT AVATARS (white glyphs on dark chips) ---
_AI_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
    'stroke="#F5F5F5" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M6 4h12v13H6z"/>'
    '<path d="M9 6.6h6M9 9.6h6M9 12.6h6"/>'
    "</svg>"
)
_USER_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
    'stroke="#F5F5F5" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round">'
    '<circle cx="12" cy="7" r="4"/>'
    '<path d="M6.5 12.5Q12 17 17.5 12.5"/>'
    "</svg>"
)
AI_AVATAR = _svg_data_uri(_AI_SVG)
USER_AVATAR = _svg_data_uri(_USER_SVG)


# --- SIDEBAR STATUS ICONS (inline SVG, rendered as real vectors) ---
ICON_LOG = (
    '<svg viewBox="0 0 24 24" fill="#22C55E">'
    '<rect x="4" y="14" width="4" height="6" rx="2"/>'
    '<rect x="10" y="10" width="4" height="10" rx="2"/>'
    '<rect x="16" y="6" width="4" height="14" rx="2"/>'
    "</svg>"
)
ICON_MEMORY = (
    '<svg viewBox="0 0 24 24" fill="none" stroke="#3B82F6" stroke-width="2" stroke-linejoin="round">'
    '<path d="M5 6h14v11H5z"/>'
    '<ellipse cx="12" cy="6" rx="7" ry="2.5"/>'
    '<ellipse cx="12" cy="17" rx="7" ry="2.5"/>'
    "</svg>"
)
# Trash icon encoded for use as a CSS background-image (destructive accent).
_TRASH_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
    'stroke="#EF4444" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M5 7h14v9H5z"/>'
    '<path d="M7 8h10M7 11h10M7 14h10"/>'
    '<path d="M8 4h8v2H8z"/>'
    "</svg>"
)
_TRASH_CSS_URI = "data:image/svg+xml;charset=utf-8," + quote(_TRASH_SVG, safe="")


# --- SESSION MANAGEMENT (unchanged) ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    logfire.info(f"New User Session Created: {st.session_state.session_id}")

if "messages" not in st.session_state:
    st.session_state.messages = []


def clear_history() -> None:
    """Wipe the current conversation and start a fresh session memory."""
    st.session_state.messages = []
    st.session_state.session_id = str(uuid.uuid4())
    logfire.warn(f"Memory wipe triggered for session: {st.session_state.session_id}")


INJECT_CSS = """
<style>
:root {
  --bg: #050505;
  --sidebar: #030405;
  --card: #0D0D0F;
  --field: #151517;
  --border: #1C1C1F;
  --input-border: #26262A;
  --ptext: #F5F5F5;
  --stext: #8A8A8F;
  --green: #22C55E;
  --blue: #3B82F6;
  --red: #EF4444;
}

html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"],
[data-testid="stMainBlockContainer"] {
  background-color: var(--bg);
  color: var(--ptext);
}
[data-testid="stHeader"] { background: transparent; }

html, body, [data-testid="stSidebar"], [class*="st"], button, a, h1, h2, h3 {
  font-family: "Inter", "Segoe UI", system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif;
}

/* ---- Sidebar (slightly darker than the main workspace) ---- */
[data-testid="stSidebar"] {
  background-color: var(--sidebar);
  border-right: 1px solid #121316;
  min-width: 320px;
  max-width: 320px;
  overflow-x: hidden;
}
[data-testid="stSidebarUserContent"] { padding: 1.4rem 1.1rem; overflow-x: hidden; }
[data-testid="stSidebarContent"] { overflow-x: hidden; }
[data-testid="stSidebar"] hr { border-color: #16161A; }

.brand {
  display: flex; align-items: center; gap: 10px;
  font-size: 1rem; font-weight: 650; letter-spacing: -0.01em;
  color: var(--ptext); margin: 0;
}
.brand-badge {
  display: inline-flex; align-items: center; justify-content: center;
  width: 30px; height: 30px; border-radius: 8px;
  background: transparent; border: 1px solid #2A2A2E;
}
.section-label {
  color: var(--stext); font-size: 0.68rem; font-weight: 600;
  letter-spacing: 0.1em; text-transform: uppercase;
  margin: 24px 0 10px 0;
}

/* ---- System status cards ---- */
.sys-card {
  display: flex; align-items: center; gap: 12px;
  width: 100%;
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 11px;
  padding: 12px 16px;
  margin-bottom: 10px;
}
.sys-icon {
  display: inline-flex; align-items: center; justify-content: center;
  width: 33px; height: 33px; min-width: 33px;
  border-radius: 8px;
  background: #111114;
  border: 1px solid rgba(255, 255, 255, 0.06);
}
.sys-title { color: var(--ptext); font-size: 0.9rem; font-weight: 600; line-height: 1.3; }
.sys-sub { color: var(--stext); font-size: 0.77rem; line-height: 1.35; margin-top: 1px; }
.sys-dot { width: 8px; height: 8px; border-radius: 50%; margin-left: auto; min-width: 8px; }

/* ---- Header title ---- */
.main-title {
  color: var(--ptext);
  font-size: 1.5rem; font-weight: 680; letter-spacing: -0.02em;
  margin: 0;
}

/* ---- Chat messages ---- */
[data-testid="stChatMessage"] {
  border: 1px solid var(--border);
  border-radius: 13px;
  background: rgba(255, 255, 255, 0.02);
  padding: 16px 18px;
}
[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"] {
  background: #141417;
  border: 1px solid #2A2A2E;
  border-radius: 8px;
}
[data-testid="stChatMessageContent"] p,
[data-testid="stChatMessageContent"] li {
  color: var(--ptext);
  font-size: 15px;
  line-height: 1.6;
}
[data-testid="stChatMessageContent"] h1,
[data-testid="stChatMessageContent"] h2,
[data-testid="stChatMessageContent"] h3 { color: var(--ptext); }
[data-testid="stChatMessageContent"] code {
  background: #141417; color: #EDEDF0;
  border: 1px solid #2A2A2E; border-radius: 5px;
  padding: 1px 5px; font-size: 0.86em;
}

/* ---- Assistant Answer header (no emoji, no icon font) ---- */
.ans-header { display: flex; align-items: center; gap: 9px; margin: 4px 0 14px 0; }
.ans-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--green); flex-shrink: 0; }
.ans-title { color: var(--ptext); font-size: 0.9rem; font-weight: 600; letter-spacing: -0.01em; }
.spinner {
  display: inline-block; width: 14px; height: 14px; border-radius: 50%;
  border: 2px solid rgba(245, 245, 245, 0.2);
  border-top-color: #F5F5F5;
  -webkit-animation: spinfwd 0.8s linear infinite;
  animation: spinfwd 0.8s linear infinite;
  flex-shrink: 0;
}
@keyframes spinfwd { to { transform: rotate(360deg); } }
.rag-meta {
  display: inline-block;
  color: var(--stext); font-size: 0.74rem; font-weight: 550;
  border: 1px solid var(--border); background: #111114;
  border-radius: 999px; padding: 2px 10px; margin-left: 4px;
}

/* ---- Documentation input ---- */
[data-testid="stChatInput"] {
  background-color: var(--field);
  border: 1px solid var(--input-border);
  border-radius: 13px;
  box-shadow: none;
}
[data-testid="stChatInput"] textarea { color: var(--ptext); font-size: 0.93rem; }
[data-testid="stChatInput"] textarea::placeholder { color: var(--stext); }
[data-testid="stChatInput"] [data-testid="stChatInputSubmitButton"] {
  background: #2A2A2E; border-radius: 9px;
}
[data-testid="stChatInput"] [data-testid="stChatInputSubmitButton"] svg { fill: #F5F5F5; }

/* ---- Focus (subtle, no bright blue) ---- */
[data-testid="stChatInput"]:focus-within { border-color: #303036; box-shadow: none; }
[data-testid="stChatInput"] [data-testid="stChatInputSubmitButton"]:focus,
[data-testid="stChatInput"] textarea:focus { outline: none; box-shadow: none; }
[data-testid="stBaseButton-primary"]:focus,
[data-testid="stBaseButton-secondary"]:focus { outline: none; box-shadow: none; }

/* ---- Clear history & memory (dark neutral button, red accent) ---- */
[data-testid="stBaseButton-primary"] {
  display: flex;
  align-items: center;
  justify-content: flex-start;
  width: 100%;
  text-align: left;
  white-space: pre-line;
  height: auto;
  padding: 12px 16px;
  background: var(--card);
  border: 1px solid rgba(239, 68, 68, 0.32);
  border-radius: 11px;
  color: var(--ptext);
  font-weight: 600;
  box-shadow: none;
}
[data-testid="stBaseButton-primary"]::before {
  content: "";
  display: inline-block;
  width: 17px; height: 17px;
  background: url('__TRASH__') center/contain no-repeat;
  margin-right: 10px;
}
[data-testid="stBaseButton-primary"]:hover {
  background: #101013;
  border-color: rgba(239, 68, 68, 0.55);
}

/* ---- Secondary buttons (neutral) ---- */
[data-testid="stBaseButton-secondary"] {
  background: var(--card);
  border: 1px solid var(--border);
  color: var(--ptext);
  border-radius: 9px;
  font-weight: 550;
}

/* ---- Material icon-name leak fix ----
   The Material Symbols font is not bundled with this install, so Streamlit's
   icon component (<span data-testid="stIconMaterial">) renders the raw
   snake_case icon name as visible text (e.g. "keyboard_arrow_right"),
   which collides with expander labels. Draw a CSS chevron for expander
   summaries instead, and drop the raw-text icons inside alerts. */
[data-testid="stExpander"] summary [data-testid="stIconMaterial"] {
  font-size: 0 !important;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 16px;
  height: 16px;
  flex-shrink: 0;
}
[data-testid="stExpander"] summary [data-testid="stIconMaterial"]::before {
  content: "";
  width: 7px;
  height: 7px;
  border-right: 1.6px solid var(--stext);
  border-bottom: 1.6px solid var(--stext);
  transform: rotate(-45deg); /* collapsed: points right */
  transition: transform 0.15s ease;
}
details[open] > summary [data-testid="stIconMaterial"]::before {
  transform: rotate(45deg); /* expanded: points down */
}
[data-testid="stAlertContainer"] [data-testid="stIconMaterial"] {
  display: none;
}

/* ---- Expanders / alerts ---- */
[data-testid="stExpander"] {
  border: 1px solid var(--border);
  border-radius: 10px;
  background: var(--card);
}
[data-testid="stExpander"] summary span {
  color: var(--stext); font-size: 0.8rem; font-weight: 550;
}
[data-testid="stAlertContainer"] {
  border-radius: 9px;
  border: 1px solid var(--border);
}
[data-testid="stInfoElement"] p,
[data-testid="stErrorElement"] p { color: var(--stext); }

/* ---- Hide unwanted native header controls ----
   NOTE: do NOT hide [data-testid="stToolbar"] itself — in this Streamlit
   version the native Running…|STOP status pill (stStatusWidget) is a child
   of the toolbar. Its unwanted siblings are hidden individually below. */
[data-testid="stMainMenuPopover"],
[data-testid="stAppDeployButton"],
[data-testid="stThemeButton"],
[data-testid="stThemeSwitcher"],
[data-testid="stMainMenuDivider"],
[data-testid="stMainMenu"],
[data-testid="stDecoration"] {
  display: none !important;
}
[data-testid="stHeader"] {
  background: transparent;
  box-shadow: none;
  pointer-events: none;
}

/* Native generation status pill (Running... | STOP).
   Mounts only while a response is being generated. Anchored INSIDE the
   chat input box, immediately left of the send arrow, vertically centered
   on the input field. Provides the real Stop control that interrupts the
   running script. */
[data-testid="stStatusWidget"] {
  pointer-events: auto;
  position: fixed;
  bottom: 68px;
  right: 138px;
  z-index: 1000;
  display: flex;
  align-items: center;
  gap: 6px;
  height: 34px;
  background: var(--field);
  border: 1px solid var(--input-border);
  border-radius: 9px;
  padding: 0 10px;
  box-shadow: none;
}
[data-testid="stStatusWidget"] span,
[data-testid="stStatusWidget"] div {
  color: var(--stext);
  font-size: 0.7rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}
[data-testid="stStatusWidget"] button {
  all: unset;
  cursor: pointer;
  color: var(--ptext);
  font-size: 0.68rem;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  padding: 3px 8px;
  border-radius: 6px;
}
[data-testid="stStatusWidget"] button:hover {
  background: rgba(245, 245, 245, 0.08);
}

/* ---- Hide sidebar Material collapse/expand chevrons (raw icon-text leak) ---- */
[data-testid="stSidebarCollapseButton"],
[data-testid="stExpandSidebarButton"] {
  display: none !important;
}

/* ---- Main content: full remaining width, spacing after sidebar ---- */
[data-testid="stMainBlockContainer"] {
  max-width: none;
  padding: 1.6rem 2rem 4.5rem 2rem;
}

@media (max-width: 768px) {
  [data-testid="stSidebar"] { min-width: 100%; max-width: 100%; }
}
</style>
"""
st.markdown(INJECT_CSS.replace("__TRASH__", _TRASH_CSS_URI), unsafe_allow_html=True)


def _status_card(icon_svg: str, title: str, subtitle: str, dot_color: str) -> str:
    """Dark neutral system-status card with an inline vector icon + status dot."""
    return (
        '<div class="sys-card">'
        f'<span class="sys-icon">{icon_svg}</span>'
        f'<span class="sys-text"><div class="sys-title">{title}</div>'
        f'<div class="sys-sub">{subtitle}</div></span>'
        f'<span class="sys-dot" style="background:{dot_color};"></span>'
        "</div>"
    )


# =========================================================================
# SIDEBAR
# =========================================================================
with st.sidebar:
    st.markdown(
        """
        <div class="brand">
            <span class="brand-badge">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#EDEDF0" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4.2 7.5 L12 3 L19.8 7.5 L19.8 19.5 L12 21 L4.2 19.5 Z"/><circle cx="16" cy="15" r="2.4" fill="#EDEDF0"/></svg>
            </span>
            Agent OS
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-label">System Status</div>', unsafe_allow_html=True)

    # Logfire
    st.markdown(
        _status_card(ICON_LOG, "Logfire", LOGFIRE_STATUS, "#22C55E"),
        unsafe_allow_html=True,
    )

    # Session ID
    st.markdown(
        _status_card(
            ICON_MEMORY,
            "Session ID",
            st.session_state.session_id[:8],
            "#3B82F6",
        ),
        unsafe_allow_html=True,
    )

    # Clear history & memory (dark neutral button, red accent)
    st.button(
        "Clear history & memory\nPermanently remove all history and memories",
        key="clear_history",
        type="primary",
        width="stretch",
        on_click=clear_history,
        help="Wipe the current conversation and reset session memory.",
    )


_ANSWER_HEADER = (
    '<div class="ans-header"><span class="ans-dot"></span>'
    '<span class="ans-title">Answer Synthesized</span>{meta}</div>'
)
_DOING_HEADER = (
    '<div class="ans-header"><span class="spinner"></span>'
    '<span class="ans-title">Synthesizing answer</span></div>'
)


# =========================================================================
# MAIN CONTENT
# =========================================================================
st.markdown('<h1 class="main-title">Agent Workspace</h1>', unsafe_allow_html=True)
st.markdown("")


# --- Chat history ---
for message in st.session_state.messages:
    role = message["role"]
    avatar = AI_AVATAR if role == "assistant" else USER_AVATAR
    with st.chat_message(role, avatar=avatar):
        if role == "assistant":
            st.markdown(
                _ANSWER_HEADER.format(meta=""),
                unsafe_allow_html=True,
            )
        st.markdown(message["content"])


# --- Chat input + assistant response ---
if prompt := st.chat_input("Ask about your documentation..."):
    logfire.info(f"User query received: {prompt[:120]}")

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=AI_AVATAR):
        synth_holder = st.empty()
        synth_holder.markdown(_DOING_HEADER, unsafe_allow_html=True)

        try:
            # Distributed trace: call the RAG backend
            with logfire.span("Calling RAG Backend"):
                base_url = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
                url = f"{base_url}/query"
                payload = {"q": prompt, "thread_id": st.session_state.session_id}
                response = requests.post(url, json=payload, timeout=180)
                data = response.json()

            steps = data.get("thought_process", [])
            for step in steps:
                st.write(step)

            sources = data.get("sources", [])
        except Exception as e:
            logfire.error(f"UI-Backend Connection Failed: {e}")
            synth_holder.markdown(
                '<div class="ans-header"><span class="ans-dot" '
                'style="background:#EF4444;"></span>'
                '<span class="ans-title" style="color:#EF4444;">'
                "Connection Failed</span></div>",
                unsafe_allow_html=True,
            )
            st.error("Backend Offline.")
            st.stop()

        # Answer header with dynamic RAG source metadata (no hardcoded count)
        n_sources = len(sources)
        meta = (
            f'<span class="rag-meta">RAG &middot; {n_sources} sources</span>'
            if n_sources
            else ""
        )
        synth_holder.markdown(_ANSWER_HEADER.format(meta=meta), unsafe_allow_html=True)

        # Stream the final answer
        answer_placeholder = st.empty()
        full_answer = data.get("answer", "No response.")
        curr_text = ""
        for char in full_answer:
            curr_text += char
            answer_placeholder.markdown(curr_text)
            time.sleep(0.005)
        answer_placeholder.markdown(full_answer)

        st.session_state.messages.append({"role": "assistant", "content": full_answer})

        # Retrieved context (sources)
        if sources:
            with st.expander("View Retrieved Context (Sources)"):
                for i, source in enumerate(sources):
                    preview = source[:100].replace("\n", " ") + "..."
                    with st.expander(f"Chunk {i + 1}: {preview}"):
                        st.info(source)

        logfire.info("Chat cycle completed successfully.")
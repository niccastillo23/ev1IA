"""Web interface (Streamlit) for the Seguros Express assistant.

Minimalist UI with a light/dark theme toggle (clean typography, neutral
palette, single accent). Reuses the same agent / orchestrator + memory
pipeline as the CLI (`src/agent.py`).

Run with:  streamlit run app.py
"""

import uuid

import streamlit as st

from src.agent import Assistant
from src.config import (
    EMBEDDING_MODEL,
    LLM_MODEL,
    OLLAMA_MODEL,
    PROVIDER,
    USE_LONG_TERM,
)

st.set_page_config(
    page_title="Seguros Express · Asistente",
    page_icon="▪",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Palette per theme. The CSS below is written once using these variables.
THEME_VARS = {
    "dark": """
        --bg: #0b0c0e;
        --surface: #141517;
        --surface-2: #17191d;
        --surface-hover: #212328;
        --border: #26282e;
        --border-strong: #3a3d44;
        --text: #e7e8ea;
        --muted: #a1a6ad;
        --faint: #6f747c;
        --mark-bg: #e7e8ea;
        --mark-text: #0b0c0e;
    """,
    "light": """
        --bg: #ffffff;
        --surface: #fafafa;
        --surface-2: #ffffff;
        --surface-hover: #f6f6f7;
        --border: #ececec;
        --border-strong: #c9ccd1;
        --text: #111827;
        --muted: #6b7280;
        --faint: #9ca3af;
        --mark-bg: #111827;
        --mark-text: #ffffff;
    """,
}

CUSTOM_RULES = """
    html, body, [class*="css"], .stApp, [data-testid="stAppViewContainer"] {
        font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI",
            Roboto, "Helvetica Neue", Arial, sans-serif;
        background: var(--bg);
        color: var(--text);
    }

    /* Streamlit's own theme variables must follow our palette, otherwise native
       components (markdown, chat, widgets) keep the light base colours set in
       .streamlit/config.toml and text becomes unreadable in dark mode. */
    :root {
        --text-color: var(--text) !important;
        --background-color: var(--bg) !important;
        --secondary-background-color: var(--surface) !important;
    }
    [data-testid="stMarkdownContainer"],
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] span,
    [data-testid="stMarkdownContainer"] strong,
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li {
        color: var(--text);
    }
    [data-testid="stMarkdownContainer"] a {color: var(--text);}

    #MainMenu, footer, header[data-testid="stHeader"] {visibility: hidden; height: 0;}
    [data-testid="stDecoration"], [data-testid="stStatusWidget"] {display: none;}

    .block-container {max-width: 760px; padding-top: 2rem; padding-bottom: 7rem;}

    /* ----------------------------------------------------- layout bg */
    /* Streamlit paints some containers with the theme's base colour;
       force them to follow our variable so no white bars appear. */
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stMainBlockContainer"],
    [data-testid="stAppViewBlockContainer"] {
        background: var(--bg) !important;
    }
    [data-testid="stBottom"],
    [data-testid="stBottom"] > div,
    [data-testid="stBottomBlockContainer"] {
        background: var(--bg) !important;
    }

    /* ---------------------------------------------------------- header */
    .se-header {
        display: flex; align-items: center; justify-content: space-between;
        padding-bottom: 18px; border-bottom: 1px solid var(--border);
    }
    .se-brand {display: flex; align-items: center; gap: 12px;}
    .se-mark {
        width: 34px; height: 34px; border-radius: 9px;
        background: var(--mark-bg); color: var(--mark-text);
        display: flex; align-items: center; justify-content: center;
        font-size: 12px; font-weight: 600; letter-spacing: .5px;
    }
    .se-name {font-size: 15px; font-weight: 600; letter-spacing: -.01em; line-height: 1.1; color: var(--text);}
    .se-sub {font-size: 12px; color: var(--muted); margin-top: 2px;}
    .se-meta {font-size: 11px; color: var(--faint); text-align: right; line-height: 1.4;}

    /* ------------------------------------------------------- messages */
    [data-testid="stChatMessageAvatarUser"],
    [data-testid="stChatMessageAvatarAssistant"] {display: none !important;}
    [data-testid="stChatMessage"] {background: transparent; border: none; padding: 0.35rem 0;}
    .se-role {
        font-size: 11px; text-transform: uppercase; letter-spacing: .08em;
        color: var(--faint); font-weight: 600; margin-bottom: 4px;
    }

    /* --------------------------------------------------------- inputs */
    div.stButton > button {
        width: 100%; text-align: left; justify-content: flex-start;
        background: var(--surface-2); border: 1px solid var(--border); border-radius: 10px;
        padding: 10px 14px; color: var(--text); font-size: 14px; font-weight: 400;
        transition: border-color .15s ease, background .15s ease;
    }
    div.stButton > button:hover {
        border-color: var(--border-strong); background: var(--surface-hover); color: var(--text);
    }
    div.stButton > button:focus {box-shadow: none;}

    [data-testid="stChatInput"],
    [data-testid="stChatInput"] > div {
        background: var(--surface-2) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
    }
    [data-testid="stChatInput"] [data-baseweb="base-input"],
    [data-testid="stChatInput"] [data-baseweb="textarea"],
    [data-testid="stChatInput"] [data-baseweb="base-input"] > div {
        background: var(--surface-2) !important;
    }
    [data-testid="stChatInput"] textarea {
        color: var(--text) !important; background: transparent !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {color: var(--faint) !important;}

    [data-testid="stExpander"] {
        border: 1px solid var(--border); border-radius: 10px; background: var(--surface);
    }
    [data-testid="stExpander"] summary {font-size: 13px; color: var(--muted);}

    /* -------------------------------------------------------- sidebar */
    section[data-testid="stSidebar"] {
        background: var(--surface); border-right: 1px solid var(--border);
    }
    section[data-testid="stSidebar"], section[data-testid="stSidebar"] * {color: var(--text);}
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] * {color: var(--muted);}
    section[data-testid="stSidebar"] hr {border-color: var(--border);}

    .se-footer {
        text-align: center; color: var(--faint); font-size: 11px;
        margin-top: 28px; letter-spacing: .02em;
    }

    /* -------------------------------------------------- orchestration */
    [data-testid="stMain"] [data-testid="stWidgetLabel"] p {
        font-size: 11px; text-transform: uppercase; letter-spacing: .08em;
        color: var(--muted); font-weight: 600;
    }
    .se-orch-hint {font-size: 12px; color: var(--faint); text-align: right; line-height: 1.45;}
"""


def build_css(theme: str) -> str:
    return f"<style>:root {{{THEME_VARS[theme]}}}{CUSTOM_RULES}</style>"


@st.cache_resource(show_spinner="Inicializando el asistente…")
def get_assistant(mode: str) -> Assistant:
    """Create one Assistant per orchestration mode (cached per process)."""
    return Assistant(mode=mode, use_long_term=USE_LONG_TERM)


if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "theme" not in st.session_state:
    st.session_state.theme = "dark"
if "answers" not in st.session_state:
    st.session_state.answers = {}

st.markdown(
    build_css(st.session_state.theme), unsafe_allow_html=True
)


SUGGESTIONS = [
    "¿Cuál es el número para pedir grúa?",
    "¿Cuánto es el deducible por siniestro en pesos?",
    "¿Qué documentos necesito para un reembolso?",
    "¿Está cubierta la conducción en estado de ebriedad?",
]


def _section_title(doc: str) -> str:
    """First meaningful line of a fragment, skipping the '[Fragmento i]' tag."""
    for line in (doc or "").splitlines():
        line = line.strip()
        if line and not line.startswith("[Fragmento"):
            return line[:90]
    return ""


def render_sources(result: dict) -> None:
    """Show the routing decision, tools used and retrieved fragments."""
    route = result.get("route")
    plan = result.get("plan") or []
    tools = result.get("tools_used") or []
    docs = result.get("sources") or []
    used_uf = result.get("used_uf")
    if not (route or plan or tools or docs or used_uf):
        return
    with st.expander("Fuentes y trazabilidad"):
        if plan and len(plan) > 1:
            st.markdown(f"**Plan** · {' → '.join(plan)}")
        if route:
            st.markdown(f"**Especialista** · `{route}`")
        if tools:
            st.markdown(f"**Herramientas** · {', '.join(tools)}")
        for doc in docs:
            title = _section_title(doc)
            if title:
                st.markdown(f"**Manual** · {title}")
        if used_uf:
            st.markdown(f"**UF (mindicador.cl)** · {result.get('uf_info')}")


def handle_prompt(prompt: str, assistant: Assistant) -> None:
    """Run the pipeline for a new prompt and render the exchange."""
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown('<div class="se-role">Tú</div>', unsafe_allow_html=True)
        st.markdown(prompt)

    with st.chat_message("assistant"):
        st.markdown('<div class="se-role">Asistente</div>', unsafe_allow_html=True)
        cached = st.session_state.answers.get(prompt)
        if cached is not None:
            result = cached
        else:
            with st.spinner("Consultando…"):
                try:
                    result = assistant.ask(prompt, thread_id=st.session_state.thread_id)
                except Exception as exc:  # noqa: BLE001
                    result = {
                        "answer": f"No se pudo procesar la consulta: {exc}",
                        "route": "",
                        "plan": [],
                        "tools_used": [],
                        "sources": [],
                        "used_uf": False,
                        "uf_info": None,
                    }
            # Memoize identical questions to avoid repeated API calls in a demo.
            st.session_state.answers[prompt] = result
        st.markdown(result["answer"])
        render_sources(result)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "route": result.get("route", ""),
            "plan": result.get("plan", []),
            "tools_used": result.get("tools_used", []),
            "sources": result.get("sources", []),
            "used_uf": result.get("used_uf", False),
            "uf_info": result.get("uf_info"),
        }
    )


# ---------------------------------------------------------------- Header
st.markdown(
    """
    <div class="se-header">
        <div class="se-brand">
            <div class="se-mark">SE</div>
            <div>
                <div class="se-name">Seguros Express</div>
                <div class="se-sub">Asistente del asegurado</div>
            </div>
        </div>
        <div class="se-meta">
            RAG · agentes con herramientas<br/>memoria corta y larga
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------- Orchestration selector
_orch_left, _orch_right = st.columns([2, 1])
with _orch_left:
    mode = st.radio(
        "Orquestación",
        options=("multi", "single"),
        format_func=lambda m: "Multi-agente" if m == "multi" else "Agente único",
        horizontal=True,
        index=0,
    )
with _orch_right:
    st.markdown(
        '<div class="se-orch-hint">Multi-agente: supervisor + plan + ejecutor<br/>'
        'Agente único: un solo agente ReAct</div>',
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------- Sidebar
with st.sidebar:
    theme_label = st.radio(
        "Tema",
        options=("Oscuro", "Claro"),
        index=0 if st.session_state.theme == "dark" else 1,
        horizontal=True,
        label_visibility="collapsed",
    )
    new_theme = "dark" if theme_label == "Oscuro" else "light"
    if new_theme != st.session_state.theme:
        st.session_state.theme = new_theme
        st.rerun()

    st.divider()
    chat_model = OLLAMA_MODEL if PROVIDER == "ollama" else LLM_MODEL
    provider_label = "Ollama (local)" if PROVIDER == "ollama" else "NVIDIA NIM"
    st.caption(f"**Modelo**\n{chat_model} · {provider_label}")
    st.caption(f"**Embeddings**\n{EMBEDDING_MODEL}")
    st.caption("**Fuentes**\nManual del asegurado · API UF")
    st.caption("**Memoria**\nCorto plazo (sesión) · Largo plazo (semántica)")
    st.divider()
    if st.button("Limpiar conversación"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

assistant = get_assistant(mode)

# ------------------------------------------------------------ Chat render
if not st.session_state.messages:
    st.markdown(
        '<p style="color:var(--muted);font-size:14px;margin:22px 0 10px;">'
        "Haz una consulta sobre tu póliza o empieza con un ejemplo.</p>",
        unsafe_allow_html=True,
    )
    cols = st.columns(2)
    for index, suggestion in enumerate(SUGGESTIONS):
        if cols[index % 2].button(
            suggestion, use_container_width=True, key=f"suggestion_{index}"
        ):
            st.session_state.pending_prompt = suggestion
            st.rerun()

for message in st.session_state.messages:
    role_label = "Tú" if message["role"] == "user" else "Asistente"
    with st.chat_message(message["role"]):
        st.markdown(f'<div class="se-role">{role_label}</div>', unsafe_allow_html=True)
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(message)

# -------------------------------------------------------------- New input
prompt = st.chat_input("Escribe tu consulta…")
if st.session_state.pending_prompt:
    prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None

if prompt:
    handle_prompt(prompt, assistant)

st.markdown(
    '<div class="se-footer">Proyecto académico ISY0101 · '
    "Respuestas basadas únicamente en las fuentes disponibles.</div>",
    unsafe_allow_html=True,
)

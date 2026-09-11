"""Web interface (Streamlit) for the fleet operations assistant.

Local chat UI that reuses the same RAG + tools + memory pipeline
as the CLI (`src/agent.py`).

Run with:  streamlit run app.py
"""

import streamlit as st

from src.agent import answer_query, initialize_client
from src.memory import ConversationBufferWindowMemory
from src.rag_pipeline import build_retriever

st.set_page_config(
    page_title="Asistente de Flota — Logística Express",
    page_icon="🚚",
    layout="centered",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    #MainMenu, footer, [data-testid="stHeader"] {visibility: hidden;}

    .block-container {padding-top: 1.5rem; max-width: 860px;}

    .le-header {
        display: flex; align-items: center; gap: 18px;
        padding: 22px 26px; border-radius: 18px;
        background: linear-gradient(135deg, #0f3d26 0%, #1b7f4b 100%);
        color: #ffffff; margin-bottom: 6px;
        box-shadow: 0 8px 24px rgba(15, 61, 38, 0.18);
    }
    .le-header .le-logo {font-size: 2.4rem; line-height: 1;}
    .le-header h1 {font-size: 1.45rem; margin: 0; color: #ffffff; font-weight: 700;}
    .le-header p {margin: 2px 0 0 0; font-size: 0.9rem; color: #d8f0e2;}

    .le-tag {
        display: inline-block; margin-top: 10px; padding: 4px 12px;
        border-radius: 999px; background: #e7f4ec; color: #14613a;
        font-size: 0.75rem; font-weight: 600; letter-spacing: .3px;
    }

    [data-testid="stChatMessage"] {
        border-radius: 14px; padding: 4px 6px;
    }

    section[data-testid="stSidebar"] {
        background: #f4f9f6; border-right: 1px solid #e2efe8;
    }

    .le-footer {
        text-align: center; color: #7b8a83; font-size: 0.75rem;
        margin-top: 24px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_client():
    """Create the Groq client once per server process."""
    return initialize_client()


@st.cache_resource(show_spinner=False)
def get_retriever():
    """Load and index the manual once per server process."""
    return build_retriever()


def get_memory() -> ConversationBufferWindowMemory:
    """Return this browser session's sliding-window memory."""
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferWindowMemory(
            max_turns=5, persist_path=None
        )
    return st.session_state.memory


if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None


SUGGESTIONS = [
    "¿Cuál es el número para pedir grúa?",
    "¿Cuánto es el deducible del seguro?",
    "¿Cuántas horas puedo manejar seguido?",
    "¿Qué hago si se enciende el Check Engine?",
]


def render_sources(result: dict) -> None:
    """Show the retrieved fragments and external tool usage."""
    docs = result.get("sources") or []
    used_uf = result.get("used_uf")
    if not docs and not used_uf:
        return
    with st.expander("🔎 Fuentes consultadas"):
        for doc in docs:
            title = doc.splitlines()[0].strip() if doc else ""
            st.markdown(f"- **{title}**")
        if used_uf:
            st.markdown(f"- **UF externa (mindicador.cl):** {result.get('uf_info')}")


def handle_prompt(prompt: str) -> None:
    """Run the pipeline for a new prompt and render the exchange."""
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🚚"):
        with st.spinner("Consultando manual y fuentes…"):
            try:
                result = answer_query(
                    get_client(), get_retriever(), get_memory(), prompt
                )
            except Exception as exc:  # noqa: BLE001
                result = {
                    "answer": f"⚠️ Ocurrió un error al procesar tu consulta: {exc}",
                    "sources": [],
                    "used_uf": False,
                    "uf_info": None,
                }
        st.markdown(result["answer"])
        render_sources(result)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result.get("sources", []),
            "used_uf": result.get("used_uf", False),
            "uf_info": result.get("uf_info"),
        }
    )


# ---------------------------------------------------------------- Header
st.markdown(
    """
    <div class="le-header">
        <div class="le-logo">🚚</div>
        <div>
            <h1>Logística Express</h1>
            <p>Asistente Inteligente de Operaciones y Flota</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    '<span class="le-tag">RAG + UF en tiempo real · Memoria conversacional</span>',
    unsafe_allow_html=True,
)

# --------------------------------------------------------------- Sidebar
with st.sidebar:
    st.markdown("### 🚚 Logística Express")
    st.caption("Asistente de Operaciones y Flota")
    st.divider()
    st.markdown("**Modelo**\n\n`qwen/qwen3.6-27b` · vía Groq")
    st.markdown("**Memoria**\n\nVentana deslizante · 5 turnos")
    st.markdown("**Fuentes**\n\nManual de operaciones · API UF")
    st.divider()
    if st.button("🧹 Limpiar conversación", use_container_width=True):
        st.session_state.messages = []
        st.session_state.memory = ConversationBufferWindowMemory(
            max_turns=5, persist_path=None
        )
        st.rerun()

# ------------------------------------------------------------ Chat render
if not st.session_state.messages:
    st.markdown("#### ¿En qué puedo ayudarte hoy?")
    cols = st.columns(2)
    for index, suggestion in enumerate(SUGGESTIONS):
        if cols[index % 2].button(
            suggestion, use_container_width=True, key=f"suggestion_{index}"
        ):
            st.session_state.pending_prompt = suggestion
            st.rerun()

for message in st.session_state.messages:
    avatar = "🧑" if message["role"] == "user" else "🚚"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(message)

# -------------------------------------------------------------- New input
prompt = st.chat_input("Escribe tu consulta sobre operaciones y flota…")
if st.session_state.pending_prompt:
    prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None

if prompt:
    handle_prompt(prompt)

st.markdown(
    '<div class="le-footer">Proyecto académico ISY0101 · '
    "Las respuestas se basan únicamente en las fuentes disponibles.</div>",
    unsafe_allow_html=True,
)

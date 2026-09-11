"""Conversational agent for fleet operations support.

Uses Groq (via OpenAI-compatible API) with RAG and conversational memory.
Follows the pattern from the course notebook.
"""

import re
import time

from openai import OpenAI

from src.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
from src.memory import build_memory
from src.rag_pipeline import build_retriever
from src.tools import consultar_valor_uf_actual

SYSTEM_PROMPT = """\
/no_think
Eres el Asistente Inteligente de Seguros Express S.A.
Tu rol es apoyar a los asegurados con consultas sobre el manual del asegurado:
coberturas, siniestros, deducibles, reembolsos, primas y red de prestadores.

REGLAS ESTRICTAS:
1. Usa el contexto proporcionado para responder. Si la informacion no esta en el contexto, indica que no tienes esa informacion. NO inventes.
2. Responde siempre en espanol de forma clara, concisa y profesional.
3. Si la pregunta no esta relacionada con seguros, indica amablemente que solo puedes asistir con temas de la compania.
4. Responde de forma directa, sin mostrar tu razonamiento interno.
5. Si el usuario pide una cantidad de UF (por ejemplo "5 UF"), multiplica el valor de la UF entregado en el contexto por esa cantidad y muestra el resultado en pesos. NO inventes el valor de la UF: usa siempre el que viene en el contexto.
"""

THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)


def initialize_client():
    """Create Groq client via OpenAI-compatible API."""
    return OpenAI(
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
    )


def clean_response(content: str) -> str:
    """Remove reasoning blocks (closed or truncated) from the response."""
    if not content:
        return ""
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
    if "<think>" in content:
        content = content.split("<think>")[0]
    return content.strip()


def build_messages(query, context, conversation_history):
    """Build the message list including retrieved context and memory."""
    if context:
        prompt = f"""Contexto:
{context}

Pregunta: {query}

Responde basandote unicamente en el contexto proporcionado. Si el usuario pide una cantidad de UF (por ejemplo "5 UF"), multiplica el valor de la UF del contexto por esa cantidad. Si la informacion no esta en el contexto, indica que no tienes esa informacion. Responde directamente, sin razonamiento interno."""
    else:
        prompt = f"""Pregunta: {query}

Responde basandote en tu conocimiento general. Si no tienes informacion, indicarlo claramente. Responde directamente, sin razonamiento interno."""

    return (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + conversation_history
        + [{"role": "user", "content": prompt}]
    )


def generate_response(client, query, context, conversation_history=None, retries=3):
    """Call the LLM with retry logic for rate limits."""
    conversation_history = conversation_history or []
    messages = build_messages(query, context, conversation_history)

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                temperature=0.1,
                max_tokens=600,
                reasoning_effort="none",
            )
            content = response.choices[0].message.content or ""
            return clean_response(content)
        except TypeError:
            # Provider does not accept reasoning_effort; retry without it.
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                temperature=0.1,
                max_tokens=600,
            )
            content = response.choices[0].message.content or ""
            return clean_response(content)
        except Exception as e:
            if "429" in str(e) and attempt < retries - 1:
                wait = (attempt + 1) * 15
                print(f"  Rate limit. Reintentando en {wait}s... ({attempt + 2}/{retries})")
                time.sleep(wait)
            else:
                raise


def detect_uf_query(query: str) -> bool:
    """Detect if the query is about UF value or requires currency conversion."""
    uf_keywords = [
        "uf",
        "unidad de fomento",
        "valor uf",
        "cuanto vale la uf",
        "precio uf",
        "en pesos",
        "a pesos",
        "en clp",
        "en dinero",
        "equivalente en pesos",
    ]
    return any(kw in query.lower() for kw in uf_keywords)


def answer_query(client, retriever, memory, query: str) -> dict:
    """Full RAG + tool pipeline for a single user query.

    Retrieves manual context, optionally queries the external UF tool,
    generates an answer and updates the conversation memory.

    Returns:
        dict with keys: answer, sources, used_uf, uf_info.
    """
    relevant_docs = retriever(query, top_k=3)
    context = "\n".join(relevant_docs) if relevant_docs else ""

    used_uf = False
    uf_info = None
    if detect_uf_query(query):
        uf_info = consultar_valor_uf_actual()
        used_uf = True
        context = f"{context}\n\nInformacion economica: {uf_info}" if context else uf_info

    response = generate_response(client, query, context, memory.get_history())
    memory.add_user(query)
    memory.add_assistant(response)

    return {
        "answer": response,
        "sources": relevant_docs,
        "used_uf": used_uf,
        "uf_info": uf_info,
    }


def main():
    """Run the interactive CLI loop with conversational memory."""
    print("=" * 60)
    print("  Asistente Inteligente de Seguros - Seguros Express")
    print("=" * 60)
    print("Escribe 'salir' o 'exit' para terminar.\n")

    client = initialize_client()
    retriever = build_retriever()
    memory = build_memory()

    while True:
        try:
            user_input = input("  Asegurado: ").strip()
        except (EOFError, KeyboardInterrupt):
            memory.save()
            print("\n  Sesion finalizada.")
            break

        if user_input.lower() in ("salir", "exit"):
            memory.save()
            print("  Sesion finalizada.")
            break

        if not user_input:
            continue

        try:
            result = answer_query(client, retriever, memory, user_input)
            print(f"\n  Asistente: {result['answer']}\n")
        except Exception as exc:
            print(f"\n  Error: {exc}\n")


if __name__ == "__main__":
    main()

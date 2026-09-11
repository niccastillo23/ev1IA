"""Conversational agent for fleet operations support.

Builds a LangGraph agent with tool calling, conversational memory,
and a strict system prompt to minimize hallucinations.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain.agents import create_agent
from langchain_mistralai import ChatMistralAI
from langgraph.checkpoint.memory import MemorySaver

from src.config import LLM_API_KEY, LLM_MODEL
from src.tools import consultar_manual_operaciones, consultar_valor_uf_actual

SYSTEM_PROMPT = """\
Eres el Asistente Inteligente de Operaciones y Flota de Logística Express S.A.
Tu rol es apoyar a los conductores con consultas sobre el manual de operaciones,
protocolos de seguridad, mantenimiento, jornada laboral, combustible y talleres
autorizados.

REGLAS ESTRICTAS:
1. Consulta SIEMPRE el manual de operaciones usando la herramienta
   'consultar_manual_operaciones' antes de responder cualquier pregunta
   sobre procedimientos internos.
2. Si la información no se encuentra en las fuentes (manual o API externa),
   declara explícitamente que no se encuentra disponible. NO inventes ni
   alucines información.
3. Para consultas sobre valores económicos en UF, usa la herramienta
   'consultar_valor_uf_actual' para obtener el valor vigente.
4. Responde siempre en español de forma clara, concisa y profesional.
5. Si la pregunta no está relacionada con operaciones de flota, indica
   amablemente que solo puedes asistir con temas de la empresa.

Prioridad de consulta: primero el manual, luego la UF si aplica.\
"""


def build_agent():
    """Create and return the agent with tools and memory."""
    llm = ChatMistralAI(
        model=LLM_MODEL,
        api_key=LLM_API_KEY,
        temperature=0.1,
    )

    tools = [consultar_manual_operaciones, consultar_valor_uf_actual]

    memory = MemorySaver()

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=SystemMessage(content=SYSTEM_PROMPT),
        checkpointer=memory,
    )
    return agent


def main():
    """Run the interactive CLI loop with conversational memory."""
    print("=" * 60)
    print("  Asistente de Operaciones y Flota - Logística Express")
    print("=" * 60)
    print("Escribe 'salir' o 'exit' para terminar.\n")

    agent = build_agent()
    config = {"configurable": {"thread_id": "session-1"}}

    while True:
        try:
            user_input = input("🧑 Chofer: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Sesión finalizada.")
            break

        if user_input.lower() in ("salir", "exit"):
            print("👋 Sesión finalizada.")
            break

        if not user_input:
            continue

        try:
            result = agent.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
            )
            last_message = result["messages"][-1]
            print(f"\n🚛 Asistente: {last_message.content}\n")
        except Exception as exc:
            print(f"\n⚠️  Error: {exc}\n")


if __name__ == "__main__":
    main()

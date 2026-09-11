"""Conversational agent for fleet operations support.

Builds a LangChain agent with tool calling, conversational memory,
and a strict system prompt to minimize hallucinations.
"""

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_mistralai import ChatMistralAI

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
    """Create and return the agent executor with tools and memory."""
    llm = ChatMistralAI(
        model=LLM_MODEL,
        api_key=LLM_API_KEY,
        temperature=0.1,
    )

    tools = [consultar_manual_operaciones, consultar_valor_uf_actual]

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5,
    )
    return executor


def main():
    """Run the interactive CLI loop with conversational memory."""
    print("=" * 60)
    print("  Asistente de Operaciones y Flota - Logística Express")
    print("=" * 60)
    print("Escribe 'salir' o 'exit' para terminar.\n")

    executor = build_agent()
    chat_history = []

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
            result = executor.invoke({
                "input": user_input,
                "chat_history": chat_history,
            })
            print(f"\n🚛 Asistente: {result['output']}\n")
            chat_history.append(HumanMessage(content=user_input))
        except Exception as exc:
            print(f"\n⚠️  Error: {exc}\n")


if __name__ == "__main__":
    main()

"""High-level assistant façade for Seguros Express.

Wires together the ReAct agent / multi-agent orchestrator with short-term
(LangGraph checkpointer) and long-term (semantic) memory. Both the Streamlit
app and the CLI use this same façade, so they share identical behaviour.
"""

from src.agent_graph import build_agent, run_agent
from src.memory import LongTermMemory
from src.orchestrator import build_orchestrator, run_orchestrator

VALID_MODES = ("multi", "single")


class Assistant:
    """Conversational assistant with selectable orchestration mode.

    Args:
        mode: ``"multi"`` uses the supervisor + specialists orchestrator,
            ``"single"`` uses one ReAct agent with all tools.
        use_long_term: enable the semantic long-term memory.
        thread_id: session identifier for the short-term memory.
    """

    def __init__(self, mode: str = "multi", use_long_term: bool = True):
        if mode not in VALID_MODES:
            raise ValueError(f"mode debe ser uno de {VALID_MODES}")
        self.mode = mode
        self.long_term = LongTermMemory() if use_long_term else None
        self.agent = build_agent() if mode == "single" else None
        self.graph = build_orchestrator() if mode == "multi" else None

    def ask(self, query: str, thread_id: str = "default") -> dict:
        """Answer a query and return a result dict.

        Keys: answer, route, tools_used, sources, used_uf, uf_info.
        """
        if self.mode == "multi":
            return run_orchestrator(self.graph, query, thread_id, self.long_term)
        return run_agent(self.agent, query, thread_id, self.long_term)


def main():
    """Interactive CLI loop using the multi-agent assistant."""
    print("=" * 64)
    print("  Asistente Inteligente de Seguros - Seguros Express (multi-agente)")
    print("=" * 64)
    print("Escribe 'salir' o 'exit' para terminar.\n")

    assistant = Assistant(mode="multi")

    while True:
        try:
            user_input = input("  Asegurado: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Sesión finalizada.")
            break

        if user_input.lower() in ("salir", "exit"):
            print("  Sesión finalizada.")
            break
        if not user_input:
            continue

        try:
            result = assistant.ask(user_input)
            route = result.get("route")
            tools = result.get("tools_used") or []
            print(f"\n  Asistente [{route}] (herramientas: {', '.join(tools) or 'ninguna'}):")
            print(f"  {result['answer']}\n")
        except Exception as exc:  # noqa: BLE001
            print(f"\n  Error: {exc}\n")


if __name__ == "__main__":
    main()

"""Tests for the insurance assistant retrieval, tools and agents."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import src.agent_tools as agent_tools
from src.agent_graph import extract_metadata, sanitize_answer
from src.agent_tools import (
    _next_folio,
    _uf_number,
    consultar_siniestros,
    registrar_siniestro,
)
from src.memory import ConversationBufferWindowMemory
from src.orchestrator import _build_plan, _is_greeting, _keyword_route, _raw_query
from src.rag_pipeline import _fuse, build_retriever
from src.tools import consultar_manual_operaciones, consultar_valor_uf_actual


class TestRetrieval(unittest.TestCase):

    def setUp(self):
        # Lexical-only retriever keeps tests hermetic (no embeddings/network).
        self.retriever = build_retriever(use_embeddings=False)

    def test_accents_and_morphology_are_handled(self):
        docs = self.retriever("cual es el deducible por siniestro", top_k=3)
        joined = "\n".join(docs)
        self.assertIn("5 UF", joined)

    def test_synonyms_resolve_coverage_query(self):
        docs = self.retriever("cuanto cubre un choque", top_k=3)
        joined = "\n".join(docs).lower()
        self.assertIn("terceros", joined)

    def test_accentless_query_returns_tow_truck(self):
        docs = self.retriever("numero de grua", top_k=3)
        joined = "\n".join(docs)
        self.assertIn("800-500-100", joined)

    def test_unknown_topic_returns_no_results(self):
        docs = self.retriever("politica de viaticos", top_k=3)
        self.assertEqual(docs, [])


class TestConsultarManualOperaciones(unittest.TestCase):

    @patch("src.tools._get_retriever")
    def test_returns_results_for_valid_query(self, mock_retriever):
        mock_retriever.return_value = lambda q, top_k=3: [
            "Llamar al 800-500-100 en caso de siniestro."
        ]

        result = consultar_manual_operaciones("numero de grua")
        self.assertIn("800-500-100", result)

    @patch("src.tools._get_retriever")
    def test_returns_no_results_message_when_empty(self, mock_retriever):
        mock_retriever.return_value = lambda q, top_k=3: []

        result = consultar_manual_operaciones("tema inexistente")
        self.assertIn("No se encontraron resultados", result)


class TestConsultarValorUFActual(unittest.TestCase):

    @patch("src.tools.requests.get")
    def test_returns_formatted_uf_value(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "uf": {
                "valor": 38500.50,
                "fecha": "2025-01-15T04:00:00.000Z",
            }
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = consultar_valor_uf_actual()
        self.assertIn("$38,500.50", result)
        self.assertIn("CLP", result)

    @patch("src.tools.requests.get")
    def test_handles_timeout_gracefully(self, mock_get):
        import requests as req
        mock_get.side_effect = req.exceptions.Timeout()

        result = consultar_valor_uf_actual()
        self.assertIn("timeout", result.lower())

    @patch("src.tools.requests.get")
    def test_handles_connection_error_gracefully(self, mock_get):
        import requests as req
        mock_get.side_effect = req.exceptions.ConnectionError("refused")

        result = consultar_valor_uf_actual()
        self.assertIn("Error", result)


class TestConversationBufferWindowMemory(unittest.TestCase):

    def test_trims_to_window_size(self):
        memory = ConversationBufferWindowMemory(max_turns=2)
        for i in range(5):
            memory.add_user(f"pregunta {i}")
            memory.add_assistant(f"respuesta {i}")

        history = memory.get_history()
        self.assertEqual(len(history), 4)
        self.assertEqual(history[0]["content"], "pregunta 3")
        self.assertEqual(history[-1]["content"], "respuesta 4")

    def test_clear_resets_buffer(self):
        memory = ConversationBufferWindowMemory(max_turns=3)
        memory.add_user("hola")
        memory.add_assistant("hola, ¿en qué ayudo?")
        memory.clear()
        self.assertEqual(memory.get_history(), [])

    def test_save_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "session.json")
            memory = ConversationBufferWindowMemory(max_turns=3, persist_path=path)
            memory.add_user("pregunta")
            memory.add_assistant("respuesta")
            memory.save()

            restored = ConversationBufferWindowMemory(max_turns=3, persist_path=path)
            restored.load()
            self.assertEqual(restored.get_history(), memory.get_history())


class TestFusion(unittest.TestCase):

    def test_prefers_lexical_and_deduplicates(self):
        lexical = ["SECCIÓN 2: SINIESTROS\nDeducible 5 UF."]
        vector = ["SECCIÓN 2: SINIESTROS\nDeducible 5 UF.", "3.1 Coberturas\nDaños a terceros."]
        fused = _fuse(lexical, vector, top_k=3)
        self.assertEqual(len(fused), 2)
        self.assertEqual(fused[0], lexical[0])

    def test_fills_slots_with_vector(self):
        fused = _fuse([], ["a", "b"], top_k=2)
        self.assertEqual(fused, ["a", "b"])


class TestRouting(unittest.TestCase):

    def test_keyword_routing(self):
        self.assertEqual(_keyword_route("cuanto cubre un choque"), "coberturas")
        self.assertEqual(_keyword_route("numero de grua"), "siniestros")
        self.assertEqual(_keyword_route("documentos para reembolso"), "reembolsos")
        self.assertEqual(_keyword_route("deducible en pesos"), "financiero")

    def test_keyword_routing_returns_none_for_unknown(self):
        self.assertIsNone(_keyword_route("receta de chocolate"))

    def test_greeting_detection(self):
        self.assertTrue(_is_greeting("Hola, buenos días"))
        self.assertTrue(_is_greeting("hola"))
        self.assertFalse(_is_greeting("cuento cubre el choque"))

    def test_raw_query_strips_injected_memory(self):
        injected = (
            "[Contexto de conversaciones previas del asegurado]\n"
            "- 5 UF = $205.000 CLP en pesos\n\n"
            "[Consulta actual]\n¿Cubren la conducción en estado de ebriedad?"
        )
        self.assertEqual(
            _raw_query(injected), "¿Cubren la conducción en estado de ebriedad?"
        )
        self.assertEqual(_raw_query("consulta simple"), "consulta simple")


class TestUfParser(unittest.TestCase):

    def test_parses_chilean_format(self):
        raw = "Valor UF al 2025-01-15: $38,500.50 CLP"
        self.assertEqual(_uf_number(raw), 38500.50)

    def test_returns_none_without_number(self):
        self.assertIsNone(_uf_number("Error al consultar el valor de la UF"))


class TestExtractMetadata(unittest.TestCase):

    def test_collects_tools_and_sources(self):
        class Msg:
            def __init__(self, type, name="", content="", tool_calls=None):
                self.type = type
                self.name = name
                self.content = content
                self.tool_calls = tool_calls

        messages = [
            Msg("ai", tool_calls=[{"name": "buscar_manual"}]),
            Msg("tool", name="buscar_manual", content="Telefono 800-500-100"),
            Msg("ai", content="Listo."),
        ]
        meta = extract_metadata(messages)
        self.assertIn("buscar_manual", meta["tools_used"])
        self.assertEqual(len(meta["sources"]), 1)
        self.assertFalse(meta["used_uf"])

    def test_detects_uf_tool(self):
        class Msg:
            type = "tool"
            name = "convertir_uf_a_clp"
            content = "5 UF = $205,000 CLP"
            tool_calls = None

        meta = extract_metadata([Msg()])
        self.assertTrue(meta["used_uf"])
        self.assertIn("205", meta["uf_info"])


class TestSanitizeAnswer(unittest.TestCase):

    def test_fixes_corrupted_uf_term(self):
        for bad in ["5 UF (Unidades de Folicación)", "5 UF (Unidades de Fenglobales)"]:
            fixed = sanitize_answer(bad)
            self.assertIn("Unidades de Fomento", fixed)
            self.assertNotIn("Folicación", fixed)
            self.assertNotIn("Fenglobales", fixed)

    def test_keeps_correct_term(self):
        text = "El deducible es 5 UF (Unidades de Fomento)."
        self.assertEqual(sanitize_answer(text), text)


class TestRegistrarSiniestro(unittest.TestCase):
    """Write-side tool: persists a siniestro and can be queried back."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._orig = agent_tools.SINIESTROS_DB_PATH
        agent_tools.SINIESTROS_DB_PATH = Path(self._tmp.name) / "siniestros.json"

    def tearDown(self):
        agent_tools.SINIESTROS_DB_PATH = self._orig
        self._tmp.cleanup()

    def test_registrar_returns_folio_and_persists(self):
        out = registrar_siniestro.invoke(
            {"patente": "abcd12", "tipo": "choque", "descripcion": "colisión leve", "ciudad": "Santiago"}
        )
        self.assertIn("SIN-", out)
        self.assertTrue(agent_tools.SINIESTROS_DB_PATH.exists())
        saved = agent_tools._load_siniestros()
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0]["patente"], "ABCD12")

    def test_consultar_lists_registered(self):
        registrar_siniestro.invoke(
            {"patente": "ABCD12", "tipo": "robo", "descripcion": "espejo", "ciudad": "Valparaíso"}
        )
        out = consultar_siniestros.invoke({"patente": "abcd12"})
        self.assertIn("ABCD12", out)
        self.assertIn("SIN-", out)

    def test_consultar_without_records(self):
        out = consultar_siniestros.invoke({"patente": "ZZZZ99"})
        self.assertIn("No hay siniestros", out)

    def test_folios_increment(self):
        self.assertTrue(_next_folio([]).endswith("-0001"))
        first = registrar_siniestro.invoke(
            {"patente": "AAAA11", "tipo": "choque", "descripcion": "x", "ciudad": ""}
        )
        second = registrar_siniestro.invoke(
            {"patente": "AAAA11", "tipo": "choque", "descripcion": "y", "ciudad": ""}
        )
        self.assertIn("-0001", first)
        self.assertIn("-0002", second)


class TestPlanner(unittest.TestCase):
    """The planner sequences specialists for compound (multi-step) requests."""

    def test_simple_query_is_single_step(self):
        self.assertEqual(_build_plan("coberturas", "que cubre un choque"), ["coberturas"])
        self.assertEqual(_build_plan("financiero", "deducible en pesos"), ["financiero"])

    def test_write_only_query(self):
        self.assertEqual(_build_plan("siniestros", "registra un choque"), ["siniestros"])

    def test_compound_query_is_two_steps(self):
        plan = _build_plan(
            "siniestros", "registra el choque y dime cuanto pagare de deducible en pesos"
        )
        self.assertEqual(plan, ["siniestros", "financiero"])


if __name__ == "__main__":
    unittest.main()

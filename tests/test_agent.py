"""Tests for the insurance assistant tools and RAG pipeline."""

import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from src.memory import ConversationBufferWindowMemory
from src.rag_pipeline import build_retriever
from src.tools import consultar_manual_operaciones, consultar_valor_uf_actual


class TestRetrieval(unittest.TestCase):

    def setUp(self):
        self.retriever = build_retriever()

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


if __name__ == "__main__":
    unittest.main()

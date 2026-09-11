"""Tests for the fleet operations agent tools and RAG pipeline."""

import unittest
from unittest.mock import patch, MagicMock

from src.tools import consultar_manual_operaciones, consultar_valor_uf_actual


class TestConsultarManualOperaciones(unittest.TestCase):

    @patch("src.tools._get_retriever")
    def test_returns_results_for_valid_query(self, mock_retriever):
        mock_retriever.return_value = lambda q, top_k=3: [
            "Llamar al 800-500-100 en caso de falla mecanica."
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


if __name__ == "__main__":
    unittest.main()

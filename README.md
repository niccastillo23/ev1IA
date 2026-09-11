# Asistente Inteligente de Operaciones y Flota — Logística Express

> Proyecto académico evaluado: Sistema de IA conversacional con LangChain, RAG híbrido, Tool Calling y memoria conversacional para soporte a conductores de flota logística.

## Descripción

Este proyecto implementa un asistente conversacional que apoya a los conductores de **Logística Express S.A.** con consultas sobre el manual de operaciones: fallas mecánicas, protocolo de siniestros, jornada laboral, mantenimiento preventivo, política de combustible y talleres autorizados. Además, puede consultar en tiempo real el valor de la UF (Unidad de Fomento) desde la API de mindicador.cl.

### Tecnologías

- **LangChain** — Framework de orquestación de agentes
- **Mistral AI** — LLM (tool calling) + embeddings
- **ChromaDB** — Vectorstore para RAG semántico
- **Tool Calling** — El agente decide qué herramienta invocar
- **Memoria conversacional** — Historial de mensajes por sesión

## Arquitectura

```
Chofer → CLI → Agente (Mistral) ↔ Memoria Conversacional
                      │
                      ├── consultar_manual_operaciones → ChromaDB (RAG)
                      └── consultar_valor_uf_actual → mindicador.cl (API)
```

Ver [docs/arquitectura.md](docs/arquitectura.md) para el diagrama completo en Mermaid.

## Requisitos Previos

- Python 3.10+
- API key de Mistral AI ([console.mistral.ai](https://console.mistral.ai/))

## Instalación

```bash
# 1. Clonar el repositorio
git clone <repo-url>
cd ev1IA

# 2. Crear entorno virtual
python -m venv .venv
source .venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar variables de entorno
cp .env.example .env
# Editar .env y agregar tu LLM_API_KEY
```

### Archivo `.env`

```env
LLM_API_KEY=tu-api-key-aqui
LLM_MODEL=mistral-small-latest
LLM_EMBEDDING_MODEL=mistral-embed
CHROMA_COLLECTION_NAME=fleet_operations
CHROMA_PERSIST_DIR=.chroma
```

## Ejecución

```bash
python src/agent.py
```

## Casos de Prueba

| # | Tipo | Pregunta de Ejemplo | Herramienta Esperada | Resultado Esperado |
|---|------|---------------------|---------------------|-------------------|
| 1 | **Consulta interna** | "¿Cuál es el número para pedir grúa?" | `consultar_manual_operaciones` | 800-500-100, cobertura hasta 120 km |
| 2 | **Memoria multicanal** | "¿Cuánto es el deducible?" → "¿Y cuánto es en pesos?" | `consultar_manual_operaciones` → `consultar_valor_uf_actual` | 5 UF → valor en CLP calculado |
| 3 | **Cálculo externo** | "¿Cuánto vale la UF hoy?" | `consultar_valor_uf_actual` | Valor actualizado desde mindicador.cl |
| 4 | **Anti-alucinación** | "¿Cuál es la política de viáticos?" | `consultar_manual_operaciones` | "No se encuentra en el manual" |
| 5 | **Seguridad** | "¿Qué hago si se enciende el Check Engine?" | `consultar_manual_operaciones` | Detener marcha inmediatamente, llamar 800-500-100 |
| 6 | **Jornada** | "¿Cuántas horas puedo manejar seguido?" | `consultar_manual_operaciones` | Máximo 5 horas, pausa de 30 min |

## Ejecutar Tests

```bash
python -m pytest tests/ -v
```

## Estructura del Proyecto

```
├── data/
│   └── manual_operaciones_logistica.txt   # Manual de operaciones
├── docs/
│   └── arquitectura.md                    # Documentación técnica
├── src/
│   ├── __init__.py
│   ├── config.py                          # Configuración y variables de entorno
│   ├── rag_pipeline.py                    # Pipeline RAG con ChromaDB
│   ├── tools.py                           # Herramientas LangChain
│   └── agent.py                           # Agente conversacional + CLI
├── tests/
│   └── test_agent.py                      # Tests unitarios
├── .gitignore
├── requirements.txt
└── README.md
```

## Licencia

Proyecto académico — Logística Express S.A.

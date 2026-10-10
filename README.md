# Asistente Inteligente de Seguros — Seguros Express S.A.

> Proyecto académico (ISY0101 · Ingeniería de Soluciones con IA). Asistente
> conversacional con **RAG sobre embeddings**, **agente(s) con herramientas** y
> **memoria de corto y largo plazo** para el soporte a asegurados de una
> compañía de seguros de vehículos.

## Descripción

El asistente atiende a los **asegurados de Seguros Express S.A.** con consultas
sobre el manual del asegurado: asistencia en ruta y grúa, protocolo de
siniestros, coberturas y exclusiones, reembolsos, primas y vigencia, y red de
prestadores. Además consulta en tiempo real el valor de la **UF** desde
mindicador.cl para resolver deducibles y montos en pesos.

El sistema evoluciona el RAG de la Evaluación Parcial 1 (recuperación léxica) a
una arquitectura de agentes de la Evaluación Parcial 2:

- **Recuperación híbrida**: embeddings (Mistral + Chroma) **+** búsqueda léxica.
- **Agente ReAct** con *tool calling* (LangChain v1 + LangGraph).
- **Multi-agente**: un **supervisor** que enruta a especialistas.
- **Memoria**: corto plazo (checkpointer) y largo plazo (semántica en Chroma).

## Tecnologías

| Componente | Tecnología |
|---|---|
| Chat (LLM) | Mistral `mistral-small-latest` (API compatible con OpenAI) |
| Embeddings | Mistral `mistral-embed` (1024 dims) |
| Base vectorial | Chroma (persistente) |
| RAG | Chunking de LangChain + retriever híbrido (vector + léxico) |
| Agentes | LangChain v1 (`create_agent`) + LangGraph (`MemorySaver`) |
| Orquestación | Patrón supervisor + especialistas (LangGraph) |
| Memoria largo plazo | Chroma + embeddings (recuperación semántica) |
| Interfaz | Streamlit (web) + CLI |
| Herramienta externa | API mindicador.cl (valor UF) |

## Arquitectura

```
Asegurado → Web (Streamlit) / CLI → Assistant (fachada)
                                     │
                    ┌────────────────┴──────────────────┐
                    │                                   │
             Multi-agente (default)               Agente único
             Supervisor (router)                   ReAct + tools
                    │
   ┌────────────┬───┴────────┬──────────────┐
 Coberturas  Siniestros   Reembolsos     Financiero        General
   │            │             │             │                │
   └────────────┴─────────────┴─────────────┴────────────────┘
                    │
        ┌───────────┴────────────┐
   RAG híbrido                Tool UF
 (embeddings + léxico)     (mindicador.cl)
        │
  manual_operaciones_seguros.txt
```

Ver [docs/arquitectura.md](docs/arquitectura.md) para el diagrama Mermaid
detallado, el flujo de orquestación y las decisiones de diseño.

## Instalación

Requisitos: **Python 3.10+** y una **API key de Mistral**
([console.mistral.ai/api-keys](https://console.mistral.ai/api-keys), gratuita).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # y completa LLM_API_KEY
```

### Archivo `.env`

```env
LLM_BASE_URL="https://api.mistral.ai/v1"
LLM_API_KEY="tu-api-key-de-mistral"
LLM_MODEL="mistral-small-latest"
LLM_MODEL_SMALL="ministral-8b-latest"

EMBEDDING_MODEL="mistral-embed"
VECTOR_DB_PATH="data/chroma"
CHUNK_SIZE="700"
CHUNK_OVERLAP="120"
RETRIEVAL_TOP_K="4"
VECTOR_SCORE_THRESHOLD="0.68"

MEMORY_MAX_TURNS="5"
MEMORY_PERSIST_PATH=".memory/session.json"
LONGTERM_DB_PATH="data/chroma_memory"
```

## Ejecución

### Opción 1 — Interfaz web (Streamlit, recomendada)

```bash
streamlit run app.py
```

Abre `http://localhost:8501`. Incluye burbujas de chat, preguntas rápidas,
selector de **orquestación** (multi-agente / agente único) y un panel de
**Trazabilidad** que muestra el especialista elegido, las herramientas usadas y
las fuentes consultadas.

### Opción 2 — Línea de comandos (CLI)

```bash
PYTHONPATH=. python src/agent.py
```

```
  Asegurado: ¿Cuál es el número para pedir grúa?
  Asistente [siniestros] (herramientas: buscar_manual):
  El número de la Central de Asistencia es 800-500-100.
```

## RAG con embeddings

`src/vectorstore.py` construye (una sola vez y de forma persistente) un índice
Chroma sobre el manual usando `mistral-embed`:

1. El manual se divide en secciones y luego en *chunks* con solape
   (`RecursiveCharacterTextSplitter`).
2. Cada *chunk* se embebe y se guarda en `data/chroma/`. El índice se
   reconstruye solo si el manual cambia (verificación por *fingerprint*).
3. La búsqueda devuelve fragmentos con su **score de relevancia**; los que no
   alcanzan `VECTOR_SCORE_THRESHOLD` se descartan (anti-alucinación).

`src/rag_pipeline.py` combina el vector con la **búsqueda léxica** original
(normalización de acentos, stopwords, raíces y sinónimos) mediante *fusión*: los
aciertos léxicos tienen prioridad (códigos exactos como `800-500-100` o `5 UF`)
y el vector rellena los cupos restantes. Si ninguna estrategia encuentra
contenido, el retriever devuelve vacío y el agente responde que no tiene la
información.

## Agentes y orquestación

### Herramientas (`src/agent_tools.py`)

| Herramienta | Uso |
|---|---|
| `buscar_manual(consulta)` | RAG híbrido sobre el manual del asegurado |
| `consultar_valor_uf_actual()` | Valor vigente de la UF (mindicador.cl) |
| `convertir_uf_a_clp(cantidad_uf)` | Convierte UF → CLP con el valor vigente |
| `consultar_red_talleres(ciudad)` | Talleres autorizados por ciudad |

### Agente único (`src/agent_graph.py`)

Agente ReAct construido con `langchain.agents.create_agent`, con el conjunto
completo de herramientas y memoria de corto plazo vía `MemorySaver`.

### Multi-agente — supervisor (`src/orchestrator.py`)

Patrón **supervisor + especialistas** en LangGraph:

```
START → supervisor → {coberturas | siniestros | reembolsos | financiero | general} → END
```

- El **supervisor** clasifica la consulta con un *router híbrido*: primero
  reglas deterministas por palabras clave (fiable y explicable) y, ante
  ambigüedad, un LLM con salida estructurada (`RouteDecision`).
- Cada **especialista** es un agente ReAct con un *system prompt* y un
  subconjunto de herramientas adaptados a su dominio.
- Todos comparten el mismo estado conversacional, por lo que el enrutamiento
  considera el hilo previo (los seguimientos cortos como *"¿y eso en pesos?"*
  conservan el contexto).

## Memoria

| Tipo | Implementación | Archivo |
|---|---|---|
| **Corto plazo** | Checkpointer de LangGraph (`MemorySaver`) por `thread_id` | `src/agent_graph.py`, `src/orchestrator.py` |
| **Corto plazo (buffers)** | `ConversationBufferWindowMemory` (ventana deslizante persistida) | `src/memory.py` |
| **Largo plazo** | `LongTermMemory`: resúmenes embebidos en Chroma, recuperados por similitud | `src/memory.py` |

Al responder, el asistente recupera de la memoria de largo plazo los
intercambios semánticamente relevantes y los inyecta como contexto; tras la
respuesta, guarda el nuevo intercambio.

## Casos de Prueba

| # | Tipo | Pregunta de Ejemplo | Resultado Esperado |
|---|------|---------------------|-------------------|
| 1 | Consulta interna | "¿Cuál es el número para pedir grúa?" | 800-500-100 |
| 2 | Memoria multi-turno | "¿Cuánto es el deducible?" → "¿Y eso en pesos?" | 5 UF → CLP con UF vigente |
| 3 | Cálculo externo | "¿Cuánto vale la UF hoy?" | Valor actualizado (mindicador.cl) |
| 4 | Anti-alucinación | "¿Cuál es la política de viáticos?" | "No tengo esa información" |
| 5 | Coberturas | "¿Cuánto cubre un choque?" | Daños a terceros/propios, gastos médicos hasta 200 UF |
| 6 | Exclusiones | "¿Cubren la conducción en estado de ebriedad?" | No, es exclusión |
| 7 | Tolerancia a acentos | "cual es el deducible por siniestro" | 5 UF por evento |

## Ejecutar Tests

```bash
PYTHONPATH=. python -m unittest discover -s tests -v
```

Los tests de recuperación usan `build_retriever(use_embeddings=False)` para ser
herméticos (sin red ni API key).

## Estructura del Proyecto

```
├── data/
│   └── manual_operaciones_seguros.txt   # Manual del asegurado
│   └── chroma/                          # Índice vectorial (generado)
│   └── chroma_memory/                   # Memoria de largo plazo (generada)
├── docs/
│   └── arquitectura.md                  # Documentación técnica (Mermaid)
├── app.py                               # Interfaz web (Streamlit)
├── src/
│   ├── config.py                        # Configuración y variables de entorno
│   ├── vectorstore.py                   # Embeddings Mistral + Chroma
│   ├── rag_pipeline.py                  # Recuperación híbrida (vector + léxico)
│   ├── tools.py                         # Funciones base (manual + UF)
│   ├── agent_tools.py                   # Herramientas LangChain del agente
│   ├── memory.py                        # Memoria corto y largo plazo
│   ├── agent_graph.py                   # Agente ReAct (LangChain + LangGraph)
│   ├── orchestrator.py                  # Supervisor multi-agente
│   └── agent.py                         # Fachada Assistant + CLI
├── tests/
│   └── test_agent.py                    # Tests unitarios
├── .env.example
├── requirements.txt
└── README.md
```

## Licencia

Proyecto académico — Seguros Express S.A.

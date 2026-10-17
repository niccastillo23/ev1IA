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

- **Recuperación híbrida**: embeddings (Chroma) **+** búsqueda léxica.
- **Planificación** (`plan → execute`): un **supervisor** clasifica y arma un
  plan de pasos; un **ejecutor determinista** invoca las herramientas en orden y
  un paso final de composición redacta la respuesta fundamentada.
- **Herramientas de consulta y de escritura**: además del RAG y la UF, el
  asistente **registra siniestros** y los consulta (persistencia en disco).
- **Agente ReAct** con *tool calling* (LangChain v1 + LangGraph) como modo
  alternativo de agente único.
- **Memoria**: corto plazo (checkpointer) y largo plazo (semántica en Chroma).
- **Chat con proveedor conmutable**: local (Ollama) o remoto (NVIDIA NIM).

## Tecnologías

| Componente | Tecnología |
|---|---|
| Chat (LLM) | Local **Ollama** `qwen2.5:14b-instruct` (por defecto) o remoto **NVIDIA NIM** `openai/gpt-oss-20b` |
| Embeddings | NVIDIA NIM `nvidia/llama-nemotron-embed-vl-1b-v2` (2048 dims, API OpenAI-compatible) |
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
       Supervisor (router + plan)                 ReAct + tools
                    │                                   │
             Ejecutor determinista                      │
        (invoca herramientas en orden)                  │
                    │                                   │
             Composición (LLM)  ◄────── respuesta ──────┘
                    │
   ┌────────────────┼─────────────────┬─────────────────┐
 RAG híbrido     Tool UF          Red de talleres   Registro siniestros
(emb. + léxico) (mindicador.cl)   (por ciudad)     (data/siniestros.json)
   │
 manual_operaciones_seguros.txt
```

Ver [docs/arquitectura.md](docs/arquitectura.md) para el diagrama Mermaid
detallado, el flujo de orquestación y las decisiones de diseño.

## Instalación

Requisitos: **Python 3.10+** y una **API key de NVIDIA NIM**
([build.nvidia.com](https://build.nvidia.com), gratuita, formato `nvapi-...`).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # y completa LLM_API_KEY
```

### Archivo `.env`

```env
# Proveedor del chat: "ollama" (local) o "nvidia" (remoto)
PROVIDER="ollama"
OLLAMA_BASE_URL="http://localhost:11434"
OLLAMA_MODEL="qwen2.5:14b-instruct"
OLLAMA_NUM_CTX="8192"

# Chat remoto (se usa si PROVIDER="nvidia")
LLM_BASE_URL="https://integrate.api.nvidia.com/v1"
LLM_API_KEY="nvapi-..."
LLM_MODEL="openai/gpt-oss-20b"
LLM_MODEL_SMALL="openai/gpt-oss-20b"

EMBEDDING_MODEL="nvidia/llama-nemotron-embed-vl-1b-v2"
VECTOR_DB_PATH="data/chroma"
CHUNK_SIZE="700"
CHUNK_OVERLAP="120"
RETRIEVAL_TOP_K="4"
VECTOR_SCORE_THRESHOLD="0.30"

MAX_TOKENS="700"
AGENT_RECURSION_LIMIT="6"
USE_LLM_ROUTER="false"
USE_LONG_TERM="false"

MEMORY_MAX_TURNS="5"
MEMORY_PERSIST_PATH=".memory/session.json"
LONGTERM_DB_PATH="data/chroma_memory"
```

> **Chat local (opcional, más rápido).** Con `PROVIDER="ollama"` el chat corre en
> tu equipo vía [Ollama](https://ollama.com) (`brew install ollama` +
> `ollama pull qwen2.5:14b-instruct`), sin red ni coste. Los **embeddings** siempre usan
> el endpoint OpenAI-compatible (NVIDIA NIM).

> **Topes de costo.** `MAX_TOKENS` y `AGENT_RECURSION_LIMIT` acotan el gasto por
> consulta (evitan respuestas largas y bucles de herramientas); `USE_LLM_ROUTER`
> y `USE_LONG_TERM` permiten desactivar llamadas extra durante la demo.

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
Chroma sobre el manual usando un endpoint de embeddings OpenAI-compatible
(`EMBEDDING_MODEL`), en espacio **coseno**:

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
| `registrar_siniestro(patente, tipo, descripcion, ciudad)` | **Escritura**: registra un siniestro y devuelve un folio `SIN-YYYYMMDD-NNNN` (persistente) |
| `consultar_siniestros(patente)` | Consulta los siniestros registrados de una patente |

### Agente único (`src/agent_graph.py`)

Agente ReAct construido con `langchain.agents.create_agent`, con el conjunto
completo de herramientas y memoria de corto plazo vía `MemorySaver`.

### Multi-agente — supervisor (`src/orchestrator.py`)

Patrón **plan → execute** con ejecución determinista en LangGraph:

```
START → supervisor (clasifica + arma el plan) → executor → END
```

- El **supervisor** clasifica la consulta con un *router híbrido*: primero
  reglas deterministas por palabras clave (fiable y explicable) y, ante
  ambigüedad, un LLM con salida estructurada (`RouteDecision`). Además arma un
  **plan** (p. ej. una consulta compuesta "registra un choque y dime el
  deducible en pesos" produce `["siniestros", "financiero"]`).
- El **ejecutor** invoca las herramientas en el orden del plan **por código**
  (no por *tool calling*), lo que garantiza fiabilidad y baja latencia con
  modelos locales; luego una llamada de **composición** redacta la respuesta
  final solo con la información recuperada (anti-alucinación).
- Todos comparten el mismo estado conversacional, por lo que el enrutamiento y
  la ejecución consideran el hilo previo (los seguimientos cortos como *"¿y eso
  en pesos?"* conservan el contexto y resuelven el referente).

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
| 8 | Escritura + plan compuesto | "Registra un choque de la patente ABCD12 y dime el deducible en pesos" | Folio `SIN-...` **+** 5 UF → CLP (plan `["siniestros","financiero"]`) |

## Ejecutar Tests

```bash
PYTHONPATH=. python -m unittest discover -s tests -v
```

Los tests de recuperación usan `build_retriever(use_embeddings=False)` para ser
herméticos (sin red ni API key).

## Estructura del Proyecto

```
├── data/
│   ├── manual_operaciones_seguros.txt   # Manual del asegurado
│   ├── siniestros.json                  # Registro de siniestros (herramienta de escritura)
│   ├── chroma/                          # Índice vectorial (generado)
│   └── chroma_memory/                   # Memoria de largo plazo (generada)
├── docs/
│   ├── arquitectura.md                  # Documentación técnica (Mermaid)
│   ├── build_presentacion.py            # Genera la presentación (PPTX)
│   ├── presentacion_ep2.pptx            # Presentación EP2 (4 diapositivas)
│   ├── build_informe.py                 # Genera el scaffold del informe (DOCX)
│   └── informe_ep2.docx                 # Informe técnico EP2 (scaffold)
├── app.py                               # Interfaz web (Streamlit)
├── src/
│   ├── config.py                        # Configuración y variables de entorno
│   ├── vectorstore.py                   # Embeddings (OpenAI-compatible) + Chroma coseno
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

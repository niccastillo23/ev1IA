# Asistente Inteligente de Seguros — Seguros Express S.A.

> Proyecto académico evaluado: Asistente conversacional con RAG, integración de herramientas externas y memoria conversacional para el soporte a asegurados de una compañía de seguros de vehículos.

## Descripción

Este proyecto implementa un asistente conversacional que apoya a los **asegurados de Seguros Express S.A.** con consultas sobre el manual del asegurado: asistencia en ruta y grúa, protocolo de siniestros, coberturas y exclusiones, reembolsos de gastos, primas y vigencia de la póliza, y red de prestadores autorizados. Además, puede consultar en tiempo real el valor de la UF (Unidad de Fomento) desde la API de mindicador.cl para resolver deducibles y montos en pesos.

### Tecnologías

- **Groq** — Inferencia del LLM (Qwen3) vía API compatible con OpenAI
- **OpenAI SDK** — Cliente utilizado contra el endpoint de Groq
- **RAG** — Recuperación léxica desde el manual (normalización de acentos, stopwords, raíces y sinónimos)
- **Herramientas externas** — Consulta de la UF a mindicador.cl
- **Memoria conversacional** — Buffer de ventana deslizante (`ConversationBufferWindowMemory`)

## Arquitectura

```
Asegurado → Web (Streamlit) / CLI → Agente (Qwen3 vía Groq) ↔ Memoria Conversacional
                      │
                      ├── Retriever → manual_operaciones_seguros.txt (RAG)
                      └── consultar_valor_uf_actual → mindicador.cl (API)
```

Ver [docs/arquitectura.md](docs/arquitectura.md) para el diagrama completo en Mermaid.

## Requisitos Previos

- Python 3.10+
- API key de Groq ([console.groq.com](https://console.groq.com/))

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
# Editar .env y agregar tu LLM_API_KEY de Groq
```

### Archivo `.env`

```env
LLM_API_KEY=tu-api-key-de-groq
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_MODEL=qwen/qwen3.6-27b
LLM_MODEL_SMALL=qwen/qwen3.8-27b

# Memoria conversacional (ventana deslizante)
MEMORY_MAX_TURNS=5
MEMORY_PERSIST_PATH=.memory/session.json
```

## Memoria Conversacional

El asistente utiliza una **memoria de buffer de ventana deslizante**
(`ConversationBufferWindowMemory`, en `src/memory.py`). Conserva los últimos
`MEMORY_MAX_TURNS` intercambios (5 por defecto) y descarta los más antiguos,
lo que permite conversaciones multi-turno sin que el contexto crezca sin límite.
La sesión se guarda en `.memory/session.json` y se recupera al reiniciar.

| Variable | Descripción | Default |
|---|---|---|
| `MEMORY_MAX_TURNS` | Turnos conservados (usuario + asistente) | `5` |
| `MEMORY_PERSIST_PATH` | Archivo de persistencia de la sesión | `.memory/session.json` |

## Recuperación (RAG)

`src/rag_pipeline.py` implementa recuperación léxica sin dependencias externas:
normaliza acentos (`deducible` → `Deducible`, `poliza` → `póliza`), elimina
stopwords, aplica coincidencia por raíz (`reemb` ≈ `reembolso`/`reembolsan`) y
amplía con sinónimos del dominio (`siniestro → accidente`, `cobertura → cubre`,
`grúa → auxilio`). Cuando ninguna palabra de contenido coincide, no devuelve
contexto y el asistente responde que no tiene la información.

## Ejecución

### Opción 1 — Interfaz web (Streamlit, recomendada)

```bash
streamlit run app.py
```

Abre `http://localhost:8501` en el navegador. Incluye burbujas de chat,
preguntas rápidas, panel de **Fuentes consultadas** (fragmentos del manual y
llamada a la UF) y botón para limpiar la conversación.

### Opción 2 — Línea de comandos (CLI)

```bash
PYTHONPATH=. python src/agent.py
```

Estructura de la conversación:

```
  Asegurado: ¿Cuál es el número para pedir grúa?
  Asistente: El número ... es 800-500-100.
  Asegurado: salir
```

## Casos de Prueba

| # | Tipo | Pregunta de Ejemplo | Resultado Esperado |
|---|------|---------------------|-------------------|
| 1 | **Consulta interna** | "¿Cuál es el número para pedir grúa?" | 800-500-100 |
| 2 | **Memoria multicanal** | "¿Cuánto es el deducible del seguro?" → "¿Y cuánto es eso en pesos?" | 5 UF → conversión a CLP con valor UF vigente |
| 3 | **Cálculo externo** | "¿Cuánto vale la UF hoy?" | Valor actualizado desde mindicador.cl |
| 4 | **Anti-alucinación** | "¿Cuál es la política de viáticos?" | "No tengo esa información en el contexto" |
| 5 | **Coberturas** | "¿Cuánto cubre un choque?" | Daños a terceros, daños propios, gastos médicos hasta 200 UF |
| 6 | **Exclusiones** | "¿Cubren la conducción en estado de ebriedad?" | No, es una exclusión de la póliza |
| 7 | **Tolerancia a acentos** | "cual es el deducible por siniestro" (sin tildes) | 5 UF por evento |

### Salida de referencia (casos 1–4)

```
  Asegurado: ¿Cuál es el número para pedir grúa?
  Asistente: El número para contactar la Central de Asistencia ... es el 800-500-100.

  Asegurado: ¿Cuánto es el deducible del seguro?
  Asistente: El deducible del seguro es de 5 UF por evento.

  Asegurado: ¿Y cuánto es eso en pesos?
  Asistente: ... el monto en pesos es de $204,509.70 CLP.

  Asegurado: ¿Cuál es la política de viáticos?
  Asistente: No tengo esa información en el contexto proporcionado.
```

## Ejecutar Tests

```bash
PYTHONPATH=. python -m unittest discover -s tests -v
```

## Estructura del Proyecto

```
├── data/
│   └── manual_operaciones_seguros.txt    # Manual del asegurado
├── docs/
│   └── arquitectura.md                    # Documentación técnica
├── app.py                                 # Interfaz web (Streamlit)
├── .streamlit/
│   └── config.toml                        # Tema visual de la app web
├── src/
│   ├── __init__.py
│   ├── config.py                          # Configuración y variables de entorno
│   ├── memory.py                          # Memoria (buffer de ventana deslizante)
│   ├── rag_pipeline.py                    # Recuperación de contexto (RAG)
│   ├── tools.py                           # Herramientas (manual + UF)
│   └── agent.py                           # Pipeline del agente + CLI
├── tests/
│   └── test_agent.py                      # Tests unitarios
├── .gitignore
├── .env.example
├── requirements.txt
└── README.md
```

## Licencia

Proyecto académico — Seguros Express S.A.

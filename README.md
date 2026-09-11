# Asistente Inteligente de Operaciones y Flota — Logística Express

> Proyecto académico evaluado: Asistente conversacional con RAG, integración de herramientas externas y memoria conversacional para soporte a conductores de flota logística.

## Descripción

Este proyecto implementa un asistente conversacional que apoya a los conductores de **Logística Express S.A.** con consultas sobre el manual de operaciones: fallas mecánicas, protocolo de siniestros, jornada laboral, mantenimiento preventivo, política de combustible y talleres autorizados. Además, puede consultar en tiempo real el valor de la UF (Unidad de Fomento) desde la API de mindicador.cl.

### Tecnologías

- **Groq** — Inferencia del LLM (Qwen3) vía API compatible con OpenAI
- **OpenAI SDK** — Cliente utilizado contra el endpoint de Groq
- **RAG** — Recuperación léxica desde el manual (normalización de acentos, stopwords, raíces y sinónimos)
- **Herramientas externas** — Consulta de la UF a mindicador.cl
- **Memoria conversacional** — Buffer de ventana deslizante (`ConversationBufferWindowMemory`)

## Arquitectura

```
Chofer → CLI → Agente (Qwen3 vía Groq) ↔ Memoria Conversacional
                      │
                      ├── Retriever → manual_operaciones_logistica.txt (RAG)
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
normaliza acentos (`limite` → `Límites`), elimina stopwords, aplica coincidencia
por raíz (`conduc` ≈ `conducir`/`conducción`) y amplía con sinónimos del dominio
(`conducir → manejo`, `grúa → auxilio`). Cuando ninguna palabra de contenido
coincide, no devuelve contexto y el asistente responde que no tiene la información.

## Ejecución

```bash
PYTHONPATH=. python src/agent.py
```

Estructura de la conversación:

```
  Chofer: ¿Cuál es el número para pedir grúa?
  Asistente: El número ... es 800-500-100.
  Chofer: salir
```

## Casos de Prueba

| # | Tipo | Pregunta de Ejemplo | Resultado Esperado |
|---|------|---------------------|-------------------|
| 1 | **Consulta interna** | "¿Cuál es el número para pedir grúa?" | 800-500-100 |
| 2 | **Memoria multicanal** | "¿Cuánto es el deducible del seguro?" → "¿Y cuánto es eso en pesos?" | 5 UF → conversión a CLP con valor UF vigente |
| 3 | **Cálculo externo** | "¿Cuánto vale la UF hoy?" | Valor actualizado desde mindicador.cl |
| 4 | **Anti-alucinación** | "¿Cuál es la política de viáticos?" | "No tengo esa información en el contexto" |
| 5 | **Seguridad** | "¿Qué hago si se enciende el Check Engine?" | Detener marcha inmediatamente, llamar 800-500-100 |
| 6 | **Jornada** | "¿Cuántas horas puedo manejar seguido?" | Máximo 5 horas, pausa de 30 min |
| 7 | **Tolerancia a acentos** | "cual es el limite para conducir" (sin tildes) | 5 horas continuas, jornada máxima 12 h |

### Salida de referencia (casos 1–4)

```
  Chofer: ¿Cuál es el número para pedir grúa?
  Asistente: El número para contactar a la Central de Operaciones ... es el 800-500-100.

  Chofer: ¿Cuánto es el deducible del seguro?
  Asistente: El deducible del seguro es de 5 UF por evento.

  Chofer: ¿Y cuánto es eso en pesos?
  Asistente: ... el monto en pesos es de $204,509.70 CLP.

  Chofer: ¿Cuál es la política de viáticos?
  Asistente: No tengo esa información en el contexto proporcionado.
```

## Ejecutar Tests

```bash
PYTHONPATH=. python -m unittest discover -s tests -v
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
│   ├── memory.py                          # Memoria (buffer de ventana deslizante)
│   ├── rag_pipeline.py                    # Recuperación de contexto (RAG)
│   ├── tools.py                           # Herramientas (manual + UF)
│   └── agent.py                           # Agente conversacional + CLI
├── tests/
│   └── test_agent.py                      # Tests unitarios
├── .gitignore
├── .env.example
├── requirements.txt
└── README.md
```

## Licencia

Proyecto académico — Logística Express S.A.

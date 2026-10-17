"""Genera el scaffold del informe técnico EP2 (Asistente Inteligente de Seguros).

Produce: docs/informe_ep2.docx  (Word, <=5 páginas, listo para editar)

IMPORTANTE (rúbrica): las secciones marcadas como
"[COMPLETAR — redactar tú]" NO deben escribirse con IA
(conclusiones, justificación de componentes y reflexión personal).

Uso:
    .venv/bin/python docs/build_informe.py
"""

from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm, Pt, RGBColor

AUTOR = "Nicolás Castillo"
RAMO = "ISY0101 · Ingeniería de Soluciones con IA"
TITULO = "Informe Técnico EP2 — Asistente Inteligente de Seguros"
SUBTITULO = "Seguros Express S.A. · RAG con embeddings, agentes con herramientas, planificación y memoria"
FECHA = "Evaluación Parcial 2 · 2026"

INK = RGBColor(0x11, 0x18, 0x27)
MUTED = RGBColor(0x6B, 0x72, 0x80)
RED = RGBColor(0xB4, 0x23, 0x18)
ACCENT = RGBColor(0x4B, 0x55, 0x63)


def _base(doc):
    style = doc.styles["Normal"]
    style.font.name = "Arial"
    style.font.size = Pt(11)
    pf = style.paragraph_format
    pf.space_after = Pt(6)
    pf.line_spacing = 1.15
    for section in doc.sections:
        section.top_margin = Cm(2.0)
        section.bottom_margin = Cm(2.0)
        section.left_margin = Cm(2.2)
        section.right_margin = Cm(2.2)


def h1(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run(text)
    r.bold = True
    r.font.size = Pt(14)
    r.font.color.rgb = INK
    return p


def h2(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(8)
    r = p.add_run(text)
    r.bold = True
    r.font.size = Pt(12)
    r.font.color.rgb = ACCENT
    return p


def para(doc, text):
    return doc.add_paragraph(text)


def bullet(doc, text, level=0):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.left_indent = Cm(0.6 + 0.5 * level)
    p.paragraph_format.space_after = Pt(2)
    p.add_run(text)
    return p


def todo(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(3)
    r = p.add_run("[COMPLETAR — redactar tú] ")
    r.bold = True
    r.font.color.rgb = RED
    r2 = p.add_run(text)
    r2.italic = True
    r2.font.color.rgb = MUTED
    return p


def mono(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(6)
    r = p.add_run(text)
    r.font.name = "Consolas"
    r.font.size = Pt(9)
    r.font.color.rgb = MUTED
    return p


def portada(doc):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(90)
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run(TITULO)
    r.bold = True
    r.font.size = Pt(22)

    p = doc.add_paragraph()
    r = p.add_run(SUBTITULO)
    r.font.size = Pt(13)
    r.font.color.rgb = MUTED

    doc.add_paragraph()
    for txt, bold in [(AUTOR, True), (RAMO, False), (FECHA, False)]:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(2)
        r = p.add_run(txt)
        r.bold = bold
        r.font.size = Pt(12)

    doc.add_page_break()


def build():
    doc = Document()
    _base(doc)
    portada(doc)

    h1(doc, "1. Introducción")
    para(doc, "El presente informe documenta la evolución del Asistente Inteligente de "
              "Seguros de Seguros Express S.A., desarrollado para la Evaluación Parcial 1 "
              "como un pipeline RAG y extendido en la Evaluación Parcial 2 a una "
              "arquitectura de agentes con herramientas de consulta y escritura, "
              "planificación y memoria.")
    para(doc, "El asistente responde consultas de asegurados sobre el manual (grúa y "
              "asistencia en ruta, siniestros, coberturas y exclusiones, reembolsos, "
              "primas y vigencia, y red de prestadores), consulta el valor vigente de la "
              "UF en línea y puede registrar siniestros, entregando siempre respuestas "
              "fundamentadas en las fuentes recuperadas.")
    todo(doc, "Ajusta la introducción a tu voz: problema/objetivo y alcance en 1-2 párrafos.")

    h1(doc, "2. Arquitectura de la solución")
    para(doc, "El sistema separa la obtención de datos (herramientas) de la redacción de "
              "la respuesta (composición con el LLM). El flujo general es:")
    mono(doc,
         "Asegurado -> Web (Streamlit) / CLI -> Assistant (fachada)\n"
         "   modo multi: Supervisor (clasifica + plan) -> Ejecutor determinista -> Compose (LLM)\n"
         "   modo single: Agente ReAct unico (tool calling)\n"
         "Herramientas: buscar_manual · UF (mindicador.cl) · red de talleres · registrar_siniestro\n"
         "Recuperacion: Chroma (embeddings 2048 dims) + capa lexica (fusion)")
    todo(doc, "Insertar aquí el diagrama de arquitectura (Mermaid de docs/arquitectura.md o "
              "una captura de la app). Los diagramas deben ser TUYOS.")
    para(doc, "Componentes y tecnologías:")
    for t in [
        "Chat (LLM): Ollama local qwen2.5:14b-instruct (por defecto) o NVIDIA NIM remoto openai/gpt-oss-20b.",
        "Embeddings: NVIDIA NIM nvidia/llama-nemotron-embed-vl-1b-v2 (2048 dims), espacio coseno.",
        "Base vectorial: Chroma persistente (data/chroma).",
        "Agentes: LangChain v1 (create_agent) + LangGraph (MemorySaver).",
        "Orquestación: patrón plan -> execute (supervisor + ejecutor determinista).",
        "Interfaz: Streamlit (web, con tema claro/oscuro) y CLI.",
    ]:
        bullet(doc, t)

    h1(doc, "3. Agentes y orquestación")
    para(doc, "En el modo multi-agente, el supervisor (1) clasifica la consulta con un router "
              "híbrido —reglas deterministas por palabras clave y, ante ambigüedad, un LLM "
              "con salida estructurada— y (2) arma un plan de pasos. Una consulta compuesta "
              "como «registra un choque de la patente ABCD12 y dime el deducible en pesos» "
              "produce el plan [siniestros, financiero].")
    para(doc, "El ejecutor determinista invoca las herramientas en el orden del plan por "
              "código (no delega el tool calling al LLM). Esto da fiabilidad y baja latencia "
              "con modelos locales, y hace la traza reproducible. Finalmente, una única "
              "llamada de composición redacta la respuesta usando solo los datos recuperados.")

    h1(doc, "4. Herramientas: consulta y escritura")
    for t in [
        "buscar_manual(consulta): RAG híbrido sobre el manual del asegurado.",
        "consultar_valor_uf_actual(): valor vigente de la UF (mindicador.cl).",
        "convertir_uf_a_clp(cantidad_uf): conversión UF -> CLP.",
        "consultar_red_talleres(ciudad): talleres autorizados por ciudad.",
        "registrar_siniestro(patente, tipo, descripcion, ciudad) [ESCRITURA]: registra un "
        "siniestro y devuelve un folio SIN-YYYYMMDD-NNNN persistido en data/siniestros.json.",
        "consultar_siniestros(patente): consulta los siniestros registrados.",
    ]:
        bullet(doc, t)

    h1(doc, "5. Memoria y recuperación semántica")
    para(doc, "La memoria de corto plazo (checkpointer de LangGraph por thread_id) da "
              "continuidad en la sesión: los seguimientos cortos como «¿y eso en pesos?» "
              "resuelven su referente con el turno anterior. La memoria de largo plazo "
              "recupera intercambios semánticamente relevantes (Chroma + embeddings) y los "
              "inyecta como contexto. La recuperación del manual combina embeddings (Chroma, "
              "coseno) con una capa léxica y descarta fragmentos bajo el umbral "
              "(VECTOR_SCORE_THRESHOLD=0.30) para evitar alucinaciones.")

    h1(doc, "6. Comportamiento adaptativo")
    para(doc, "El sistema adapta su comportamiento según la consulta (ver ejemplo en cada caso):")
    for t in [
        "Consulta directa de un dato exacto (grúa) -> un solo paso de RAG.",
        "Consulta compuesta -> plan multi-paso [siniestros, financiero] y ejecución en orden.",
        "Seguimiento de contexto -> reutiliza el turno previo para resolver el referente.",
        "Fuera de dominio -> responde «no cuento con esa información» (anti-alucinación).",
        "Pregunta sensible sobre póliza (ebriedad) -> responde factualmente sobre la exclusión.",
    ]:
        bullet(doc, t)
    todo(doc, "Explica TÚ por qué estas decisiones adaptativas son adecuadas e incluye evidencia "
              "(capturas o trazas). Este punto es evaluado (IE7).")

    h1(doc, "7. Decisiones de diseño y justificación de componentes")
    todo(doc, "Redacta TÚ la justificación técnica de cada componente (IE9): elección de "
              "LangGraph frente a CrewAI (compatibilidad con Python 3.14), RAG híbrido, "
              "ejecutor determinista vs. tool calling, memoria en dos niveles, modelo local "
              "(privacidad, coste, latencia) y separación entre obtención de datos y composición.")

    h1(doc, "8. Pruebas y evidencia")
    para(doc, "La batería de pruebas unitarias (31 pruebas, unittest) cubre recuperación, "
              "herramientas, memoria, planificador, escritura y saneamiento de respuestas. "
              "Comando:")
    mono(doc, "PYTHONPATH=. .venv/bin/python -m unittest discover -s tests -v")
    para(doc, "Casos demostrativos:")
    for t in [
        "«¿Cuál es el número para pedir grúa?» -> 800-500-100.",
        "«¿Cuánto es el deducible?» -> «¿Y eso en pesos?» -> 5 UF -> CLP con la UF vigente.",
        "«¿Cuál es la política de viáticos?» -> «No cuento con esa información».",
        "«¿Cubren la conducción en estado de ebriedad?» -> exclusión (no cubierta).",
        "«Registra un choque de ABCD12 y dime el deducible en pesos» -> folio SIN-… + 5 UF -> CLP.",
    ]:
        bullet(doc, t)
    todo(doc, "Adjunta evidencia: capturas de la app (panel de trazabilidad), salida de los "
              "tests y, si aplica, el repositorio en GitHub con el README.")

    h1(doc, "9. Conclusiones y reflexión personal")
    todo(doc, "PROHIBIDO ESCRIBIRLO CON IA. Redacta tú las conclusiones y la reflexión "
              "individual sobre el proceso de aprendizaje.")

    h1(doc, "10. Declaración de uso de IA")
    todo(doc, "Confirma y ajusta esta declaración: se utilizó IA como apoyo para generar y "
              "depurar código, redactar documentación técnica y diagramas; las conclusiones, "
              "la justificación de componentes y la reflexión personal fueron redactadas "
              "íntegramente por el estudiante.")

    h1(doc, "11. Referencias (APA 7.ª ed.)")
    for ref in [
        "Anthropic. (2026). Claude [Modelo de lenguaje grande]. https://www.anthropic.com",
        "Chroma. (2026). Chroma: The AI-native open-source embedding database. https://www.trychroma.com",
        "LangChain. (2026). LangChain: Build context-aware reasoning applications. https://python.langchain.com",
        "LangGraph. (2026). LangGraph: Build resilient agents. https://langchain-ai.github.io/langgraph/",
        "Mindicador. (2026). Indicadores económicos: Unidad de Fomento (UF). https://mindicador.cl",
        "NVIDIA. (2026). NVIDIA NIM microservices. https://build.nvidia.com",
        "Ollama. (2026). Ollama: Run large language models locally. https://ollama.com",
        "Qwen Team. (2026). Qwen2.5 technical report. Alibaba Cloud.",
        "Streamlit. (2026). Streamlit: A faster way to build and share data apps. https://streamlit.io",
    ]:
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Cm(0.75)
        p.paragraph_format.first_line_indent = Cm(-0.75)
        p.paragraph_format.space_after = Pt(3)
        r = p.add_run(ref)
        r.font.size = Pt(10)
    todo(doc, "Verifica y completa las referencias según las fuentes que realmente usaste.")

    out = Path(__file__).resolve().parent / "informe_ep2.docx"
    doc.save(str(out))
    print(f"OK -> {out}")


if __name__ == "__main__":
    build()

"""Genera la presentación PPTX de la EP2 (Asistente de Seguros Seguros Express).

Estructura simple inspirada en la presentación de la EP1:
- Portada + 3 diapositivas.
- Cada diapositiva: título en pregunta, subtítulo "Seguros Express — ..." y
  4 ítems con frase líder en negrita.
- "¿Cómo funciona?" incluye un diagrama de flujo (formas + flechas).

Uso:
    .venv/bin/python docs/build_presentacion.py

Produce: docs/presentacion_ep2.pptx
"""

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

# --------------------------------------------------------------------------- #
# Tema (claro, académico) — misma paleta neutra que la app
# --------------------------------------------------------------------------- #
BG = RGBColor(0xFF, 0xFF, 0xFF)
SURFACE = RGBColor(0xF6, 0xF6, 0xF7)
BORDER = RGBColor(0xE2, 0xE4, 0xE8)
INK = RGBColor(0x11, 0x18, 0x27)
MUTED = RGBColor(0x4B, 0x55, 0x63)
FAINT = RGBColor(0x9C, 0xA3, 0xAF)
ACCENT = RGBColor(0x4B, 0x55, 0x63)
ACCENT_SOFT = RGBColor(0xEE, 0xF0, 0xF3)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

FONT = "Arial"
W, H = Inches(13.333), Inches(7.5)
MARGIN = Inches(0.7)

AUTOR = "Nicolás Castillo"
RAMO = "ISY0101 · Ingeniería de Soluciones con IA"
FECHA = "Evaluación Parcial 2 · 2026"


def _set_font(run, size, color=INK, bold=False, italic=False, name=FONT):
    run.font.size = Pt(size)
    run.font.color.rgb = color
    run.font.bold = bold
    run.font.italic = italic
    run.font.name = name


def blank_slide(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    bg = slide.background.fill
    bg.solid()
    bg.fore_color.rgb = BG
    return slide


def textbox(slide, x, y, w, h, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = 0
    tf.margin_right = 0
    tf.margin_top = 0
    tf.margin_bottom = 0
    tf.paragraphs[0].alignment = align
    return tb, tf


def add_title(slide, text, subtitle=None, kicker=None):
    top = Inches(0.62)
    if kicker:
        _, tfk = textbox(slide, MARGIN, top, W - 2 * MARGIN, Inches(0.3))
        rk = tfk.paragraphs[0].add_run()
        rk.text = kicker.upper()
        _set_font(rk, 11, ACCENT, bold=True)
        top = Inches(0.95)
    _, tf = textbox(slide, MARGIN, top, W - 2 * MARGIN, Inches(0.7))
    r = tf.paragraphs[0].add_run()
    r.text = text
    _set_font(r, 30, INK, bold=True)
    line_y = Inches(1.55)
    if subtitle:
        p = tf.add_paragraph()
        p.space_before = Pt(6)
        r2 = p.add_run()
        r2.text = subtitle
        _set_font(r2, 15, ACCENT, bold=True)
        line_y = Inches(2.0)
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, MARGIN, line_y, W - 2 * MARGIN, Emu(9525))
    line.fill.solid()
    line.fill.fore_color.rgb = BORDER
    line.line.fill.background()
    line.shadow.inherit = False
    return line_y


def add_footer(slide, index):
    _, tf = textbox(slide, MARGIN, Inches(7.02), W - 2 * MARGIN, Inches(0.3))
    r = tf.paragraphs[0].add_run()
    r.text = "Asistente Inteligente de Seguros · Seguros Express  —  EP1 → EP2"
    _set_font(r, 9, FAINT)
    _, tf2 = textbox(slide, W - MARGIN - Inches(1), Inches(7.02), Inches(1), Inches(0.3),
                     align=PP_ALIGN.RIGHT)
    r2 = tf2.paragraphs[0].add_run()
    r2.text = str(index)
    _set_font(r2, 9, FAINT)


def add_notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


def lead_bullets(slide, items, x, y, w, h, size=16, gap=16):
    """items: lista de (lead, rest). Renderiza '• Lead: rest' con lead en negrita."""
    _, tf = textbox(slide, x, y, w, h)
    first = True
    for lead, rest in items:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.space_after = Pt(gap)
        rb = p.add_run()
        rb.text = "•  "
        _set_font(rb, size, ACCENT, bold=True)
        rl = p.add_run()
        rl.text = f"{lead}: "
        _set_font(rl, size, INK, bold=True)
        rr = p.add_run()
        rr.text = rest
        _set_font(rr, size, MUTED)
    return tf


def box(slide, x, y, w, h, text, fill=SURFACE, line=BORDER, size=12, bold=True, color=INK):
    shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill
    shp.line.color.rgb = line
    shp.line.width = Pt(1)
    shp.shadow.inherit = False
    tf = shp.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = text
    _set_font(r, size, color, bold=bold)
    return shp


def arrow_right(slide, x, cy, w=Inches(0.3)):
    shp = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, x, cy - Inches(0.09), w, Inches(0.18))
    shp.fill.solid()
    shp.fill.fore_color.rgb = FAINT
    shp.line.fill.background()
    shp.shadow.inherit = False
    return shp


def add_table(slide, headers, rows, x, y, w, h, col_widths=None, font_size=13):
    shape = slide.shapes.add_table(len(rows) + 1, len(headers), x, y, w, h)
    table = shape.table
    table.first_row = True
    if col_widths:
        total = sum(col_widths)
        for i, cw in enumerate(col_widths):
            table.columns[i].width = Emu(int(w * cw / total))

    def fill_cell(cell, text, size, color, bold, fill):
        cell.fill.solid()
        cell.fill.fore_color.rgb = fill
        cell.margin_left = Inches(0.12)
        cell.margin_right = Inches(0.08)
        cell.margin_top = Inches(0.04)
        cell.margin_bottom = Inches(0.04)
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        tf = cell.text_frame
        tf.word_wrap = True
        r = tf.paragraphs[0].add_run()
        r.text = text
        _set_font(r, size, color, bold=bold)

    for c, header in enumerate(headers):
        fill_cell(table.cell(0, c), header, font_size, WHITE, True, ACCENT)
    for r, row in enumerate(rows, start=1):
        base = WHITE if r % 2 else SURFACE
        for c, value in enumerate(row):
            fill_cell(table.cell(r, c), value, font_size,
                      INK if c == 0 else MUTED, bold=(c == 0), fill=base)
    return table


# --------------------------------------------------------------------------- #
# Slides (portada + 3)
# --------------------------------------------------------------------------- #
def slide_portada(prs):
    s = blank_slide(prs)
    box(s, MARGIN, Inches(2.05), Inches(0.9), Inches(0.9), "SE",
        fill=INK, line=INK, size=22, color=WHITE)
    _, tf = textbox(s, MARGIN, Inches(3.2), W - 2 * MARGIN, Inches(1.5))
    r = tf.paragraphs[0].add_run()
    r.text = "Asistente Inteligente de Seguros"
    _set_font(r, 40, INK, bold=True)
    p = tf.add_paragraph()
    r = p.add_run()
    r.text = "Seguros Express S.A. · RAG, agentes con herramientas y memoria"
    _set_font(r, 19, MUTED)
    p2 = tf.add_paragraph()
    p2.space_before = Pt(8)
    r = p2.add_run()
    r.text = "Evolución EP1 → EP2"
    _set_font(r, 18, ACCENT, bold=True)

    _, tf = textbox(s, MARGIN, Inches(6.0), W - 2 * MARGIN, Inches(0.8))
    for i, txt in enumerate([RAMO, f"{AUTOR}   ·   {FECHA}"]):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        r = p.add_run()
        r.text = txt
        _set_font(r, 12, MUTED if i == 0 else INK, bold=(i == 1))
    add_notes(s, (
        "En una frase: construimos un asistente de seguros que responde SOLO con el "
        "manual del asegurado. En la EP1 era un RAG léxico; en la EP2 agregamos "
        "recuperación con embeddings, agentes con herramientas, memoria y registro de "
        "siniestros. Voy a mostrar qué es, cómo funciona y su valor."
    ))


def slide_evolucion(prs):
    s = blank_slide(prs)
    add_title(s, "Evolución EP1 → EP2",
              subtitle="Seguros Express — Qué cambió y por qué")
    rows = [
        ("Recuperación", "Léxica", "Híbrida: embeddings + léxico"),
        ("Arquitectura", "Pipeline RAG", "Agentes (multi-agente + único)"),
        ("Herramientas", "2 (consulta)", "6 (consulta + escritura)"),
        ("Planificación", "No", "Plan → execute"),
        ("Memoria", "5 turnos", "Corto + largo plazo"),
        ("Modelo", "Qwen3 (Groq)", "Qwen2.5 local / NVIDIA NIM"),
    ]
    add_table(s, ["", "EP1", "EP2"], rows, MARGIN, Inches(2.2),
              W - 2 * MARGIN, Inches(3.85), col_widths=[2.4, 2.8, 6.4], font_size=13)
    _, tf = textbox(s, MARGIN, Inches(6.3), W - 2 * MARGIN, Inches(0.5))
    r = tf.paragraphs[0].add_run()
    r.text = "De responder con el manual  →  a agentes que consultan, escriben, planifican y recuerdan."
    _set_font(r, 14, ACCENT, bold=True)
    add_footer(s, 2)
    add_notes(s, (
        "En una frase: la EP1 era un RAG léxico que respondía con el manual; la EP2 son "
        "agentes. Sumamos recuperación híbrida con embeddings, orquestación de agentes "
        "(multi-agente y único), seis herramientas incluida la escritura, planificación "
        "plan→execute y memoria en dos niveles. El modelo pasó de Qwen3 remoto (Groq) a "
        "Qwen2.5 local, por privacidad, coste y funcionamiento offline."
    ))


def slide_que_es(prs):
    s = blank_slide(prs)
    add_title(s, "¿Qué es el Asistente de IA?",
              subtitle="Seguros Express — Asistente Virtual con IA")
    lead_bullets(s, [
        ("Atención conversacional", "Atiende a los asegurados de «Seguros Express» de forma directa y clara."),
        ("Respuestas sobre el manual", "Resuelve dudas frecuentes del contrato y la póliza usando el manual oficial."),
        ("Cálculo de deducibles", "Convierte montos en UF a pesos chilenos con el valor vigente (mindicador.cl)."),
        ("Registro de siniestros", "Ante un siniestro, lo deja registrado y entrega un folio de seguimiento."),
    ], MARGIN, Inches(2.4), W - 2 * MARGIN, Inches(4.2), size=17, gap=20)
    add_footer(s, 3)
    add_notes(s, (
        "En una frase: es un asistente de seguros que responde solo con el manual. En la "
        "EP1 era un RAG léxico; en la EP2 suma recuperación con embeddings, agentes con "
        "herramientas y memoria. Además de responder, ahora puede registrar siniestros y "
        "entregar un folio."
    ))


def slide_como_funciona(prs):
    s = blank_slide(prs)
    add_title(s, "¿Cómo funciona el Sistema? (Agentes)",
              subtitle="Seguros Express — Arquitectura de Agentes y Herramientas")

    # Diagrama de flujo (formas + flechas)
    y = Inches(2.3)
    bw, bh = Inches(2.05), Inches(0.9)
    gap = Inches(0.35)
    x = MARGIN
    steps = [
        ("Asegurado", SURFACE, BORDER, INK),
        ("Supervisor\nclasifica + plan", WHITE, BORDER, INK),
        ("Ejecutor\nherramientas", WHITE, BORDER, INK),
        ("Composición\nQwen2.5 (local)", WHITE, BORDER, INK),
        ("Respuesta\nfundamentada", ACCENT_SOFT, ACCENT, INK),
    ]
    for i, (text, fill, line, color) in enumerate(steps):
        box(s, x, y, bw, bh, text, fill=fill, line=line, size=11, color=color)
        if i < len(steps) - 1:
            arrow_right(s, x + bw + Inches(0.02), y + bh / 2, w=gap - Inches(0.06))
        x += bw + gap

    lead_bullets(s, [
        ("Recuperación híbrida", "Embeddings (Chroma) para el significado + búsqueda léxica para códigos exactos como 800-500-100."),
        ("Planificación por pasos", "El supervisor clasifica la consulta y arma un plan (p. ej. registrar un siniestro + calcular el deducible)."),
        ("Ejecución con herramientas", "Un ejecutor las invoca en orden: manual, UF, red de talleres y registro de siniestros."),
        ("Respuesta con Qwen2.5", "El modelo redacta usando solo el contexto hallado y mantiene memoria de los últimos 5 turnos."),
    ], MARGIN, Inches(3.5), W - 2 * MARGIN, Inches(3.4), size=13.5, gap=11)

    add_footer(s, 4)
    add_notes(s, (
        "El flujo: entra la consulta, un supervisor la clasifica y arma un plan de pasos, "
        "un ejecutor invoca las herramientas por código (manual, UF, talleres y escritura) "
        "y una composición con Qwen2.5 redacta con el contexto hallado. La recuperación es "
        "híbrida (embeddings + léxico) y la memoria de corto plazo conserva los últimos 5 "
        "turnos."
    ))


def slide_valor(prs):
    s = blank_slide(prs)
    add_title(s, "Valor y Confiabilidad",
              subtitle="Seguros Express — Beneficios Principales")
    lead_bullets(s, [
        ("Respuestas fundamentadas", "Todo lo que dice el asistente está respaldado por el manual oficial."),
        ("Información trazable", "Muestra el plan, las herramientas usadas y las fuentes del manual."),
        ("Sin invención de datos", "Si la respuesta no está en el manual, indica claramente que no la posee."),
        ("Transparencia total", "Evita alucinaciones y entrega máxima confianza al cliente."),
    ], MARGIN, Inches(2.4), W - 2 * MARGIN, Inches(4.2), size=17, gap=20)
    add_footer(s, 5)
    add_notes(s, (
        "El valor: todo está respaldado por el manual (fundamentado), la traza permite "
        "verificar los datos, si la respuesta no está en el manual lo dice (sin inventar) y "
        "eso entrega transparencia y confianza. Quedo atento a preguntas."
    ))


def build():
    prs = Presentation()
    prs.slide_width = W
    prs.slide_height = H

    slide_portada(prs)
    slide_evolucion(prs)
    slide_que_es(prs)
    slide_como_funciona(prs)
    slide_valor(prs)

    out = Path(__file__).resolve().parent / "presentacion_ep2.pptx"
    prs.save(str(out))
    print(f"OK -> {out}  ({len(prs.slides)} slides)")


if __name__ == "__main__":
    build()

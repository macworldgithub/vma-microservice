"""
main.py — VMA Backend
=====================
Virtual Meeting Assistant  |  OmniSuiteAI / Patterson Cheney Automotive Group

All configuration (including the OpenAI API key) is loaded from environment
variables via config.py — never passed in request bodies.

Endpoints
---------
  GET  /health           Service liveness check
  POST /analyse          Full structured JSON report
  POST /analyse/stream   Streaming JSON (token-by-token via GPT-4o)
  POST /report/pdf       Branded A4 PDF download
  POST /report/text      Plain-text report + raw JSON  (useful for email / logging)
"""

import io
import json
import re
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, Response
from pydantic import BaseModel
from openai import AsyncOpenAI

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable,
)

import config as cfg

# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="VMA Backend — OmniSuiteAI",
    version="1.0.0",
    description="Virtual Meeting Assistant: transcript analysis + branded PDF reports.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cfg.ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single shared AsyncOpenAI client — key loaded from env at startup.
_openai_client = AsyncOpenAI(api_key=cfg.OPENAI_API_KEY)

# ── Brand palette ──────────────────────────────────────────────────────────────

BRAND_TEAL   = colors.HexColor("#00B4D8")
BRAND_DARK   = colors.HexColor("#0A0A0A")
BRAND_GRAY   = colors.HexColor("#6B7280")
BRAND_LIGHT  = colors.HexColor("#F3F4F6")
BRAND_BORDER = colors.HexColor("#E5E7EB")
RED_BG       = colors.HexColor("#FEF2F2")
RED_TEXT     = colors.HexColor("#991B1B")
AMBER_BG     = colors.HexColor("#FFFBEB")
AMBER_TEXT   = colors.HexColor("#92400E")
GREEN_BG     = colors.HexColor("#F0FDF4")
GREEN_TEXT   = colors.HexColor("#166534")
BLUE_BG      = colors.HexColor("#EFF6FF")

# ── Request model  (no api_key field — key lives in env only) ─────────────────

class TranscriptRequest(BaseModel):
    transcript:    str
    meeting_title: Optional[str] = "Meeting"
    meeting_date:  Optional[str] = None
    organisation:  Optional[str] = None   # falls back to cfg.ORGANISATION

# ── OpenAI system prompt ───────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are the OmniSuiteAI Virtual Meeting Assistant — expert at analysing
automotive dealership meeting transcripts and producing structured intelligence reports.

Return a single valid JSON object with EXACTLY these fields — no markdown fences, no preamble:

{
  "meeting_title": "string",
  "meeting_date": "string",
  "duration_estimate": "string (e.g. '45 minutes')",
  "participants": [
    {"name": "string", "role": "string", "speaking_time": "string (e.g. '~30%')"}
  ],
  "executive_summary": "string (3-5 sentences)",
  "key_decisions": ["string"],
  "action_items": [
    {"owner": "string", "task": "string", "deadline": "string", "priority": "High|Medium|Low"}
  ],
  "risks_flagged": ["string"],
  "follow_up_questions": ["string"],
  "next_meeting": {
    "suggested_date": "string",
    "suggested_time": "string",
    "agenda_items": ["string"]
  },
  "next_steps": [
    {"step": "string", "owner": "string", "timeline": "string"}
  ],
  "detailed_notes": "string (comprehensive paragraph-form notes covering all discussion points)",
  "sentiment": "Positive|Neutral|Negative|Mixed",
  "generated_at": "string (ISO 8601 UTC timestamp)"
}

Rules:
- Extract ALL named speakers; infer roles from context (Sales Manager, GM, Finance Director, etc.)
- Action items must have specific owners and realistic deadlines
- Next steps must be ordered by urgency
- For automotive/dealership context flag: sales figures, inventory, finance penetration,
  service turnaround, compliance (ASIC / APPs), and campaign metrics
- Return ONLY the raw JSON object — nothing else
"""

# ── Internal helpers ───────────────────────────────────────────────────────────

def _resolve_org(req: TranscriptRequest) -> str:
    """Use request-level org if provided, otherwise fall back to env default."""
    return (req.organisation or "").strip() or cfg.ORGANISATION


def _clean_json(raw: str) -> dict:
    """Strip any accidental markdown fences and parse JSON."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def _analyse(req: TranscriptRequest) -> dict:
    """
    Call GPT-4o with the transcript and return a validated dict.
    Raises HTTPException on OpenAI or JSON-parse errors.
    """
    date_hint = req.meeting_date or datetime.now().strftime("%d %B %Y")
    org       = _resolve_org(req)

    user_msg = (
        f"Meeting Title: {req.meeting_title}\n"
        f"Meeting Date: {date_hint}\n"
        f"Organisation: {org}\n\n"
        f"TRANSCRIPT:\n{req.transcript}"
    )

    try:
        response = await _openai_client.chat.completions.create(
            model=cfg.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            temperature=cfg.OPENAI_TEMPERATURE,
            max_tokens=cfg.OPENAI_MAX_TOKENS,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"OpenAI error: {exc}")

    raw = response.choices[0].message.content or ""
    try:
        data = _clean_json(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse model response as JSON: {exc} | raw snippet: {raw[:300]}",
        )

    data.setdefault("generated_at", _now_iso())
    data.setdefault("meeting_title", req.meeting_title)
    return data


def _priority_colors(priority: str):
    p = (priority or "").upper()
    if p == "HIGH":   return RED_BG,   RED_TEXT
    if p == "MEDIUM": return AMBER_BG, AMBER_TEXT
    return GREEN_BG, GREEN_TEXT


# ── PDF builder ────────────────────────────────────────────────────────────────

def build_pdf(data: dict, organisation: str) -> bytes:
    """
    Render a fully branded A4 PDF report from the structured meeting data dict.
    Returns raw PDF bytes ready to send as an HTTP response.
    """
    buffer = io.BytesIO()
    PAGE_W, PAGE_H = A4
    MARGIN = 20 * mm

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=28 * mm,
        bottomMargin=22 * mm,
        title=data.get("meeting_title", "Meeting Report"),
        author="OmniSuiteAI VMA",
    )
    CW = PAGE_W - 2 * MARGIN   # usable content width

    # ── Paragraph style factory ────────────────────────────────────────────────

    def S(name, **kw):
        return ParagraphStyle(name, **kw)

    sTitle    = S("sTitle",    fontSize=22, leading=28, fontName="Helvetica-Bold",
                               textColor=BRAND_DARK,  spaceAfter=4)
    sSectionH = S("sSectionH", fontSize=10, leading=14, fontName="Helvetica-Bold",
                               textColor=BRAND_TEAL,  spaceBefore=12, spaceAfter=5)
    sBody     = S("sBody",     fontSize=9.5, leading=14, fontName="Helvetica",
                               textColor=BRAND_DARK,  spaceAfter=3)
    sBodyBold = S("sBodyBold", fontSize=9.5, leading=14, fontName="Helvetica-Bold",
                               textColor=BRAND_DARK,  spaceAfter=2)
    sSmall    = S("sSmall",    fontSize=8.5, leading=12, fontName="Helvetica",
                               textColor=BRAND_GRAY)
    sBullet   = S("sBullet",   fontSize=9.5, leading=14, fontName="Helvetica",
                               textColor=BRAND_DARK,  spaceAfter=3,
                               leftIndent=12, firstLineIndent=-12)
    sCenter   = S("sCenter",   fontSize=9,   leading=14, fontName="Helvetica",
                               textColor=BRAND_DARK,  alignment=TA_CENTER)
    sNotes    = S("sNotes",    fontSize=9,   leading=15, fontName="Helvetica",
                               textColor=BRAND_DARK,  spaceAfter=4)

    # ── Per-page header & footer ───────────────────────────────────────────────

    def on_page(canvas, doc):
        canvas.saveState()
        # Header bar
        canvas.setFillColor(BRAND_DARK)
        canvas.rect(0, PAGE_H - 18 * mm, PAGE_W, 18 * mm, fill=1, stroke=0)
        # Teal accent line
        canvas.setFillColor(BRAND_TEAL)
        canvas.rect(0, PAGE_H - 19.5 * mm, PAGE_W, 1.5 * mm, fill=1, stroke=0)
        # Brand name
        canvas.setFillColor(colors.white)
        canvas.setFont("Helvetica-Bold", 11)
        canvas.drawString(MARGIN, PAGE_H - 11.5 * mm, "OmniSuiteAI")
        canvas.setFont("Helvetica", 9)
        canvas.setFillColor(BRAND_TEAL)
        canvas.drawString(MARGIN + 66, PAGE_H - 11.5 * mm, "Virtual Meeting Assistant")
        # Page number
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.white)
        canvas.drawRightString(PAGE_W - MARGIN, PAGE_H - 11.5 * mm, f"Page {doc.page}")
        # Footer bar
        canvas.setFillColor(BRAND_BORDER)
        canvas.rect(0, 0, PAGE_W, 13 * mm, fill=1, stroke=0)
        canvas.setFillColor(BRAND_GRAY)
        canvas.setFont("Helvetica", 7.5)
        gen_at = data.get("generated_at", "")[:19].replace("T", " ")
        canvas.drawString(MARGIN, 7.5 * mm,
                          f"Generated {gen_at} UTC  |  CONFIDENTIAL — Internal Use Only")
        canvas.drawRightString(PAGE_W - MARGIN, 7.5 * mm, organisation)
        canvas.restoreState()

    # ── Story helpers ──────────────────────────────────────────────────────────

    story = []

    def hr(col=BRAND_BORDER, t=0.5):
        return HRFlowable(width="100%", thickness=t, color=col,
                          spaceAfter=5, spaceBefore=2)

    def section(title: str):
        story.append(Spacer(1, 1 * mm))
        story.append(Paragraph(title.upper(), sSectionH))
        story.append(hr(BRAND_TEAL, 0.8))

    def card(paragraphs: list, bg, border):
        """Wrap a list of Paragraphs in a shaded rounded-ish card."""
        tbl = Table([[p] for p in paragraphs], colWidths=[CW])
        tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), bg),
            ("BOX",           (0, 0), (-1, -1), 0.5, border),
            ("LEFTPADDING",   (0, 0), (-1, -1), 10),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
            ("TOPPADDING",    (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ]))
        return tbl

    def info_table(rows: list):
        """Two-column label / value table — no borders, tight padding."""
        tbl = Table(rows, colWidths=[38 * mm, CW - 38 * mm])
        tbl.setStyle(TableStyle([
            ("FONTNAME",      (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTNAME",      (1, 0), (1, -1), "Helvetica"),
            ("FONTSIZE",      (0, 0), (-1, -1), 9),
            ("TEXTCOLOR",     (0, 0), (0, -1), BRAND_GRAY),
            ("TEXTCOLOR",     (1, 0), (1, -1), BRAND_DARK),
            ("TOPPADDING",    (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ]))
        return tbl

    def striped_table(header: list, rows: list, col_widths: list,
                      priority_col: int = -1, priority_data: list = None):
        """
        Dark header + alternating white/light-gray rows.
        If priority_col >= 0, colour that column cell per priority value.
        """
        all_rows = [header] + rows
        tbl = Table(all_rows, colWidths=col_widths, repeatRows=1)
        ts = [
            ("BACKGROUND",    (0, 0), (-1, 0),  BRAND_DARK),
            ("TEXTCOLOR",     (0, 0), (-1, 0),  colors.white),
            ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
            ("FONTSIZE",      (0, 0), (-1, 0),  8.5),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, BRAND_LIGHT]),
            ("FONTSIZE",      (0, 1), (-1, -1), 9),
            ("TOPPADDING",    (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING",   (0, 0), (-1, -1), 6),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
            ("BOX",           (0, 0), (-1, -1), 0.5, BRAND_BORDER),
            ("INNERGRID",     (0, 0), (-1, -1), 0.3, BRAND_BORDER),
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ]
        if priority_col >= 0 and priority_data:
            for row_idx, priority_val in enumerate(priority_data, start=1):
                bg, _ = _priority_colors(priority_val)
                ts.append(("BACKGROUND", (priority_col, row_idx),
                            (priority_col, row_idx), bg))
        tbl.setStyle(TableStyle(ts))
        return tbl

    # ── 1. Title card ──────────────────────────────────────────────────────────

    title_tbl = Table(
        [[Paragraph(data.get("meeting_title", "Meeting Report"), sTitle)]],
        colWidths=[CW],
    )
    title_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), BRAND_LIGHT),
        ("BOX",           (0, 0), (-1, -1), 0.5, BRAND_BORDER),
        ("LEFTPADDING",   (0, 0), (-1, -1), 12),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 12),
        ("TOPPADDING",    (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(title_tbl)
    story.append(Spacer(1, 4 * mm))

    story.append(info_table([
        ["Organisation", organisation],
        ["Meeting Date",  data.get("meeting_date",      "—")],
        ["Duration",      data.get("duration_estimate", "—")],
        ["Participants",  str(len(data.get("participants", [])))],
        ["Sentiment",     data.get("sentiment",         "—")],
    ]))
    story.append(Spacer(1, 4 * mm))

    # ── 2. Executive Summary ───────────────────────────────────────────────────

    section("Executive Summary")
    story.append(card(
        [Paragraph(data.get("executive_summary", "No summary available."), sBody)],
        BLUE_BG, colors.HexColor("#BFDBFE"),
    ))

    # ── 3. Participants ────────────────────────────────────────────────────────

    participants = data.get("participants", [])
    if participants:
        section("Participants")
        c1 = CW * 0.30
        c3 = CW * 0.20
        c2 = CW - c1 - c3
        rows = [
            [Paragraph(p.get("name", "—"), sBodyBold),
             Paragraph(p.get("role", "—"), sBody),
             Paragraph(p.get("speaking_time", "—"), sCenter)]
            for p in participants
        ]
        story.append(striped_table(
            header=["Name", "Role", "Speaking Time"],
            rows=rows,
            col_widths=[c1, c2, c3],
        ))

    # ── 4. Key Decisions ──────────────────────────────────────────────────────

    decisions = data.get("key_decisions", [])
    if decisions:
        section("Key Decisions")
        for d in decisions:
            story.append(Paragraph(f"<b>&#10003;</b>  {d}", sBullet))

    # ── 5. Action Items ────────────────────────────────────────────────────────

    action_items = data.get("action_items", [])
    if action_items:
        section("Action Items")
        n_w  = 8  * mm
        p_w  = 22 * mm
        d_w  = 30 * mm
        o_w  = 28 * mm
        t_w  = CW - n_w - p_w - d_w - o_w
        rows = [
            [Paragraph(str(i),                      sCenter),
             Paragraph(ai.get("task",     "—"),     sBody),
             Paragraph(ai.get("owner",    "—"),     sBodyBold),
             Paragraph(ai.get("deadline", "—"),     sSmall),
             Paragraph(ai.get("priority", "Medium"), sCenter)]
            for i, ai in enumerate(action_items, 1)
        ]
        story.append(striped_table(
            header=["#", "Task", "Owner", "Deadline", "Priority"],
            rows=rows,
            col_widths=[n_w, t_w, o_w, d_w, p_w],
            priority_col=4,
            priority_data=[ai.get("priority", "Medium") for ai in action_items],
        ))

    # ── 6. Risks Flagged ──────────────────────────────────────────────────────

    risks = data.get("risks_flagged", [])
    if risks:
        section("Risks Flagged")
        for r in risks:
            story.append(card(
                [Paragraph(f"<b>&#9888;</b>  {r}", sBody)],
                AMBER_BG, colors.HexColor("#FCD34D"),
            ))
            story.append(Spacer(1, 2))

    # ── 7. Follow-up Questions ─────────────────────────────────────────────────

    fup = data.get("follow_up_questions", [])
    if fup:
        section("Follow-up Questions")
        for q in fup:
            story.append(Paragraph(f"?  {q}", sBullet))

    # ── 8. Next Steps ─────────────────────────────────────────────────────────

    next_steps = data.get("next_steps", [])
    if next_steps:
        section("Next Steps")
        n_w   = 8  * mm
        o_w2  = 32 * mm
        tl_w  = 32 * mm
        st_w  = CW - n_w - o_w2 - tl_w
        rows = [
            [Paragraph(str(i),                  sCenter),
             Paragraph(ns.get("step",     "—"), sBody),
             Paragraph(ns.get("owner",    "—"), sBodyBold),
             Paragraph(ns.get("timeline", "—"), sSmall)]
            for i, ns in enumerate(next_steps, 1)
        ]
        story.append(striped_table(
            header=["#", "Step", "Owner", "Timeline"],
            rows=rows,
            col_widths=[n_w, st_w, o_w2, tl_w],
        ))

    # ── 9. Next Meeting ────────────────────────────────────────────────────────

    nm = data.get("next_meeting", {})
    if nm:
        section("Next Meeting")
        meta = []
        if nm.get("suggested_date"): meta.append(["Date", nm["suggested_date"]])
        if nm.get("suggested_time"): meta.append(["Time", nm["suggested_time"]])
        if meta:
            story.append(info_table(meta))
            story.append(Spacer(1, 3))
        agenda = nm.get("agenda_items", [])
        if agenda:
            story.append(Paragraph("Proposed Agenda", sBodyBold))
            for item in agenda:
                story.append(Paragraph(f"&#8226;  {item}", sBullet))

    # ── 10. Detailed Notes ─────────────────────────────────────────────────────

    notes = data.get("detailed_notes", "")
    if notes:
        section("Detailed Notes")
        story.append(card([Paragraph(notes, sNotes)], BRAND_LIGHT, BRAND_BORDER))

    # ── Build ──────────────────────────────────────────────────────────────────

    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    buffer.seek(0)
    return buffer.read()


# ── Plain-text report builder ──────────────────────────────────────────────────

def build_text_report(data: dict, organisation: str) -> str:
    """Render the meeting data as a plain-text string (email / logging)."""
    SEP  = "=" * 72
    DIV  = "─" * 72

    def section(title):
        return f"\n{DIV}\n  {title}\n{DIV}"

    lines = [
        SEP,
        "  MEETING INTELLIGENCE REPORT — OmniSuiteAI VMA",
        f"  {data.get('meeting_title', '').upper()}",
        SEP,
        f"  Organisation : {organisation}",
        f"  Date         : {data.get('meeting_date',      '—')}",
        f"  Duration     : {data.get('duration_estimate', '—')}",
        f"  Sentiment    : {data.get('sentiment',         '—')}",
        f"  Generated    : {data.get('generated_at',      '—')}",
        "",
        section("PARTICIPANTS"),
    ]
    for p in data.get("participants", []):
        lines.append(
            f"  • {p.get('name','?'):<25} {p.get('role',''):<30} {p.get('speaking_time','')}"
        )

    lines.append(section("EXECUTIVE SUMMARY"))
    lines.append(f"  {data.get('executive_summary', '')}")

    lines.append(section("KEY DECISIONS"))
    for d in data.get("key_decisions", []):
        lines.append(f"  [✓]  {d}")

    lines.append(section("ACTION ITEMS"))
    for i, ai in enumerate(data.get("action_items", []), 1):
        lines += [
            f"  [{i}] {ai.get('task', '')}",
            f"       Owner    : {ai.get('owner',    'TBD')}",
            f"       Deadline : {ai.get('deadline', 'TBD')}",
            f"       Priority : {ai.get('priority', 'Medium')}",
            "",
        ]

    if data.get("risks_flagged"):
        lines.append(section("RISKS FLAGGED"))
        for r in data["risks_flagged"]:
            lines.append(f"  [!]  {r}")

    if data.get("follow_up_questions"):
        lines.append(section("FOLLOW-UP QUESTIONS"))
        for q in data["follow_up_questions"]:
            lines.append(f"  [?]  {q}")

    lines.append(section("NEXT STEPS"))
    for i, ns in enumerate(data.get("next_steps", []), 1):
        lines += [
            f"  [{i}] {ns.get('step', '')}",
            f"       Owner    : {ns.get('owner',    'TBD')}",
            f"       Timeline : {ns.get('timeline', 'TBD')}",
            "",
        ]

    nm = data.get("next_meeting", {})
    lines.append(section("NEXT MEETING"))
    lines.append(f"  Date  : {nm.get('suggested_date', 'TBD')}")
    lines.append(f"  Time  : {nm.get('suggested_time', 'TBD')}")
    lines.append("  Proposed Agenda:")
    for item in nm.get("agenda_items", []):
        lines.append(f"    •  {item}")

    lines.append(section("DETAILED NOTES"))
    lines.append(f"  {data.get('detailed_notes', '')}")
    lines += ["", SEP, "  — Generated by OmniSuiteAI Virtual Meeting Assistant —", SEP]

    return "\n".join(lines)


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Ops"])
async def health():
    """Service liveness check."""
    return {
        "status":  "ok",
        "service": "VMA Backend",
        "version": "1.0.0",
        "model":   cfg.OPENAI_MODEL,
    }


@app.post("/analyse", tags=["Analysis"])
async def analyse(req: TranscriptRequest):
    """
    Analyse a meeting transcript and return a fully structured JSON report.

    Fields returned: meeting_title, meeting_date, duration_estimate,
    participants, executive_summary, key_decisions, action_items,
    risks_flagged, follow_up_questions, next_meeting, next_steps,
    detailed_notes, sentiment, generated_at.
    """
    if not req.transcript.strip():
        raise HTTPException(400, "transcript cannot be empty.")
    data = await _analyse(req)
    return JSONResponse(content=data)


@app.post("/analyse/stream", tags=["Analysis"])
async def analyse_stream(req: TranscriptRequest):
    """
    Same as /analyse but streams raw JSON tokens from GPT-4o as they arrive
    (Content-Type: text/plain).  Collect the full stream, then JSON.parse().
    """
    if not req.transcript.strip():
        raise HTTPException(400, "transcript cannot be empty.")

    date_hint = req.meeting_date or datetime.now().strftime("%d %B %Y")
    org       = _resolve_org(req)

    user_msg = (
        f"Meeting Title: {req.meeting_title}\n"
        f"Meeting Date: {date_hint}\n"
        f"Organisation: {org}\n\n"
        f"TRANSCRIPT:\n{req.transcript}"
    )

    async def generator():
        try:
            stream = await _openai_client.chat.completions.create(
                model=cfg.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                temperature=cfg.OPENAI_TEMPERATURE,
                max_tokens=cfg.OPENAI_MAX_TOKENS,
                stream=True,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except Exception as exc:
            yield f'\n{{"error": "{exc}"}}'

    return StreamingResponse(generator(), media_type="text/plain")


@app.post("/report/pdf", tags=["Reports"])
async def report_pdf(req: TranscriptRequest):
    """
    Analyse transcript and return a **branded A4 PDF** as a file attachment.

    The PDF includes: title card, participant table, executive summary,
    key decisions, colour-coded action items, risk cards, follow-up questions,
    next steps table, next meeting block, and detailed notes.
    OmniSuiteAI header + confidentiality footer on every page.

    curl example:
        curl -X POST http://localhost:8000/report/pdf \\
             -H "Content-Type: application/json" \\
             -d '{"transcript":"...","meeting_title":"Q3 Review"}' \\
             --output report.pdf
    """
    if not req.transcript.strip():
        raise HTTPException(400, "transcript cannot be empty.")

    org  = _resolve_org(req)
    data = await _analyse(req)
    pdf  = build_pdf(data, org)

    safe = re.sub(r"[^\w\-]", "_", data.get("meeting_title", "report"))[:60]
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="VMA_Report_{safe}.pdf"'},
    )


@app.post("/report/text", tags=["Reports"])
async def report_text(req: TranscriptRequest):
    """
    Analyse transcript and return a **plain-text** formatted report
    alongside the raw JSON data object.  Useful for email bodies or audit logs.

    Response shape: { "report": "...", "data": { ... } }
    """
    if not req.transcript.strip():
        raise HTTPException(400, "transcript cannot be empty.")

    org  = _resolve_org(req)
    data = await _analyse(req)
    return JSONResponse(content={"report": build_text_report(data, org), "data": data})
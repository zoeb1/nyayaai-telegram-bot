# ==========================================
# NYAYAAI TELEGRAM BOT — FULL VERSION
# WITH GOOGLE CSE + KANOON FALLBACK + PDF
# NOW WITH HELP MENU, FAQ, LOGGING
# ==========================================

import os
import re
import uuid
import json
import logging
import fitz
import pdfkit
import requests
from datetime import datetime
from bs4 import BeautifulSoup

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)

from openai import OpenAI


# ============================
# LOGGING
# ============================

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger("NyayaAI")
logger.info("Starting NyayaAI bot...")


# ============================
# DEBUG ENV PRINT
# ============================

print("ENV DEBUG:")
print("GOOGLE_API_KEY =", os.getenv("GOOGLE_API_KEY"))
print("GOOGLE_CSE_ID =", os.getenv("GOOGLE_CSE_ID"))
print("TELEGRAM_TOKEN =", os.getenv("TELEGRAM_TOKEN"))
print("------------------------------")


# ============================
# LOAD ENV VARIABLES
# ============================

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

client = OpenAI(api_key=OPENAI_API_KEY)


# ============================
# UTILITIES
# ============================

def extract_text_from_pdf(path):
    try:
        doc = fitz.open(path)
        text = "".join(page.get_text() for page in doc)
        return text.strip()
    except Exception as e:
        logger.error(f"PDF extraction error: {e}")
        return ""


def safe_get(url, headers=None, timeout=15):
    try:
        res = requests.get(url, headers=headers or {"User-Agent": "Mozilla/5.0"}, timeout=timeout)
        return res.text
    except Exception as e:
        logger.error(f"HTTP error: {e}")
        return ""


# ============================
# GOOGLE CUSTOM SEARCH
# ============================

def google_legal_search(query, max_results=5):
    if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
        logger.error("Google keys missing")
        return []

    search_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_API_KEY,
        "cx": GOOGLE_CSE_ID,
        "q": f"{query} judgment case law",
        "num": max_results,
    }

    try:
        r = requests.get(search_url, params=params, timeout=10)
        data = r.json()

        if "items" not in data:
            return []

        results = []
        for item in data["items"]:
            link = item.get("link", "")
            if any(x in link for x in [
                "indiankanoon.org",
                "main.sci.gov.in",
                "judgment",
                "highcourt",
                "court"
            ]):
                results.append(link)

        return results[:max_results]

    except Exception as e:
        logger.error(f"Google Search Error: {e}")
        return []


# ============================
# FALLBACK: INDIANKANOON SCRAPER
# ============================

def search_cases_kanoon(query, max_results=2):
    url = f"https://indiankanoon.org/search/?formInput={query.replace(' ', '+')}"
    try:
        html = safe_get(url)
        soup = BeautifulSoup(html, "html.parser")

        links = []
        for a in soup.find_all("a", href=True):
            if a["href"].startswith("/doc/"):
                full = "https://indiankanoon.org" + a["href"]
                if full not in links:
                    links.append(full)
                if len(links) >= max_results:
                    break
        return links

    except Exception as e:
        logger.error(f"Kanoon Error: {e}")
        return []


# ============================
# FETCH CASE TEXT
# ============================

def fetch_case_text(url):
    try:
        html = safe_get(url)
        soup = BeautifulSoup(html, "html.parser")
        paras = soup.find_all("p")
        return "\n".join(p.get_text(strip=True) for p in paras)[:10000]
    except:
        return ""


# ============================
# AI HELPERS
# ============================

async def correct_query(q):
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Correct legal query. Fix IPC/CrPC numbers."},
            {"role": "user", "content": q}
        ]
    )
    return resp.choices[0].message.content.strip()


async def convert_fir_to_query(text):
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Summarise FIR into 5–8 word legal search query."},
            {"role": "user", "content": text}
        ]
    )
    return resp.choices[0].message.content.strip()


async def suggest_better_query(bad):
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Rewrite as strong legal search query."},
            {"role": "user", "content": bad}
        ]
    )
    return resp.choices[0].message.content.strip()


async def summarize_case(text, query):
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Summaries should include TITLE, SUMMARY, AI_SCORE:0-100"},
            {"role": "user", "content": text}
        ]
    )
    out = resp.choices[0].message.content

    score = 0
    for line in out.split("\n"):
        if "AI_SCORE" in line:
            try:
                score = int(line.split(":")[1])
            except:
                pass

    return out, score


async def generate_ilac_note(query, text):
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Write ILAC: Issue, Law, Application, Conclusion."},
            {"role": "user", "content": text}
        ]
    )
    return resp.choices[0].message.content


async def generate_arguments(query, text):
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Write PETITIONER, RESPONDENT, COUNTER arguments."},
            {"role": "user", "content": text}
        ]
    )
    return resp.choices[0].message.content


def extract_citations(text):
    case_pat = r"[A-Z][A-Za-z .]+ vs\.? [A-Z][A-Za-z .]+"
    statute_pat = r"(IPC\s*\d+|CrPC\s*\d+|Section\s*\d+)"

    cases = list(dict.fromkeys(re.findall(case_pat, text)))
    statutes = list(dict.fromkeys(re.findall(statute_pat, text, re.IGNORECASE)))

    out = "📌 CITED CASES:\n"
    out += "\n".join(f"- {c}" for c in cases) if cases else "None"
    out += "\n\n📌 STATUTES:\n"
    out += "\n".join(f"- {s}" for s in statutes) if statutes else "None"
    return out


async def compare_case_with_query(query, text):
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Compare legal query vs case."},
            {"role": "user", "content": text}
        ]
    )
    return resp.choices[0].message.content


async def extract_legal_issues(q):
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Extract 3–6 legal issues."},
            {"role": "user", "content": q}
        ]
    )
    return resp.choices[0].message.content.strip()


def compute_relevance_score(ai, query, text):
    q = query.lower()
    t = text.lower()

    kw = sum(5 for w in q.split() if w in t)
    kw = min(kw, 100)

    boost = 0
    for n in re.findall(r"\b\d+\b", q):
        if n in t:
            boost += 15
    boost = min(boost, 30)

    return min(int((0.6 * ai) + (0.3 * kw) + boost), 100)


# ============================
# HELP MENU + FAQ
# ============================

FAQ_BUTTONS = [
    ("What can NyayaAI do?", "FAQ_CAP"),
    ("How to convert FIR?", "FAQ_FIR"),
    ("How to generate PDF?", "FAQ_PDF"),
    ("Accuracy", "FAQ_ACC"),
    ("Pricing", "FAQ_PRICE"),
    ("Contact", "FAQ_CONTACT"),
    ("Close", "FAQ_CLOSE"),
]


def build_faq_keyboard():
    rows = []
    row = []
    for t, key in FAQ_BUTTONS:
        row.append(InlineKeyboardButton(t, callback_data=key))
        if len(row) == 2:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    return InlineKeyboardMarkup(rows)


async def help_command(update, context):
    msg = "📘 *NyayaAI Help Menu*\nSelect a topic:"
    await update.message.reply_text(msg, parse_mode="Markdown", reply_markup=build_faq_keyboard())


async def faq_button_handler(update, context):
    q = update.callback_query
    await q.answer()
    key = q.data

    if key == "FAQ_CLOSE":
        return await q.edit_message_text("Menu closed.")

    text = "Unknown option."

    if key == "FAQ_CAP":
        text = (
            "*NyayaAI can do:*\n"
            "- Supreme + High Court case search\n"
            "- Auto-correct queries\n"
            "- Summaries + relevance scores\n"
            "- ILAC notes\n"
            "- Arguments\n"
            "- Citations\n"
            "- Compare cases (/compare)\n"
            "- Extract issues (/issues)\n"
            "- FIR → Query → Case Law (/uploadfir)\n"
            "- Full PDF report (/pdf)\n"
        )

    if key == "FAQ_FIR":
        text = (
            "*FIR Conversion:* Reply to PDF → `/uploadfir` → Bot extracts FIR, builds query, generates research PDF."
        )

    if key == "FAQ_PDF":
        text = "*Generate full PDF report:*\nUse: `/pdf IPC 420 cheating`"

    if key == "FAQ_ACC":
        text = "*Accuracy:* Summaries & scores are AI-based; use for research only."

    if key == "FAQ_PRICE":
        text = "*Pricing:* Free limited version. Paid plan coming soon."

    if key == "FAQ_CONTACT":
        text = "Email: **zoebsadeqa@gmail.com**"

    back = InlineKeyboardMarkup([
        [
            InlineKeyboardButton("Back", callback_data="HELP_MENU"),
            InlineKeyboardButton("Close", callback_data="FAQ_CLOSE")
        ]
    ])

    await q.edit_message_text(text, parse_mode="Markdown", reply_markup=back)


# ============================
# START COMMAND UPDATED
# ============================

async def start_command(update, context):
    msg = (
        "👋 *Welcome to NyayaAI — India’s AI Legal Research Bot*\n\n"
        "NyayaAI helps you instantly:\n"
        "• Search Supreme & High Court cases\n"
        "• Auto-correct legal queries\n"
        "• Summaries + relevance\n"
        "• ILAC\n"
        "• Arguments\n"
        "• Citations\n"
        "• Compare cases (/compare)\n"
        "• Extract legal issues (/issues)\n"
        "• Generate full PDF legal research (/pdf)\n"
        "• FIR → Query → Case Law (/uploadfir)\n\n"
        "👇 Press the button for help."
    )

    kb = [[InlineKeyboardButton("Help & FAQ", callback_data="HELP_MENU")]]
    await update.message.reply_text(msg, parse_mode="Markdown", reply_markup=InlineKeyboardMarkup(kb))


# (PDF, compare, issues, uploadfir, reply handlers continue below)
# KEEP YOUR EXISTING PDF/JUDGMENT HANDLERS — NO CHANGE


# ============================
# PDF BUILDER
# ============================

def build_html(query, results, ilac, arguments, citations, qna):
    now = datetime.now().strftime("%d %b %Y %I:%M %p")

    header = """
    <div style='background:#1F2A44;color:white;padding:20px;border-radius:8px;margin-bottom:25px;'>
        <div style='font-size:36px;font-weight:800;'>NYAYAAI</div>
        <div style='font-size:16px;color:#D0D4D9;'>India’s AI Legal Research Assistant</div>
    </div>
    """

    body = f"<b>Query:</b> {query}<br><b>Generated:</b> {now}<hr/>"

    for i, r in enumerate(results, 1):
        body += f"""
        <h2>Case {i}</h2>
        <p><b>Link:</b> {r['link']}</p>
        <pre>{r['summary']}</pre>
        <p>AI Score: {r['ai_score']} — Relevance: {r['relevance']}%</p><hr/>
        """

    body += f"<h2>ILAC</h2><pre>{ilac}</pre><hr/>"
    body += f"<h2>Arguments</h2><pre>{arguments}</pre><hr/>"
    body += f"<h2>Citations</h2><pre>{citations}</pre><hr/>"

    if qna:
        body += f"<h2>Q&A</h2><pre>{qna}</pre><hr/>"

    footer = "<div style='text-align:center;font-size:12px;color:gray'>Generated by NyayaAI © 2025</div>"

    return f"<html><body style='font-family:Arial;padding:20px'>{header}{body}{footer}</body></html>"


# ============================
# PDF COMMAND
# ============================

async def pdf_command(update, context):
    if not context.args:
        return await update.message.reply_text("Usage: /pdf <query>")

    raw = " ".join(context.args)
    q = await correct_query(raw)

    await update.message.reply_text("🔎 Searching...")

    links = google_legal_search(q, max_results=2) or search_cases_kanoon(q, max_results=2)

    if not links:
        better = await suggest_better_query(q)
        return await update.message.reply_text(f"No cases found. Try `{better}`", parse_mode="Markdown")

    results = []
    first_text = ""

    for link in links:
        t = fetch_case_text(link)
        first_text = first_text or t
        s, ai = await summarize_case(t, q)
        rel = compute_relevance_score(ai, q, t)
        results.append({"link": link, "summary": s, "ai_score": ai, "relevance": rel})

    ilac = await generate_ilac_note(q, first_text)
    arguments = await generate_arguments(q, first_text)
    citations = extract_citations(first_text)

    qna = ""

    await update.message.reply_text("📄 Generating PDF...")

    os.makedirs("pdfs", exist_ok=True)
    filename = f"pdfs/{uuid.uuid4().hex}.pdf"
    html = build_html(q, results, ilac, arguments, citations, qna)
    pdfkit.from_string(html, filename)

    await update.message.reply_document(open(filename, "rb"))


# ============================
# COMPARE COMMAND
# ============================

async def compare_command(update, context):
    raw = " ".join(context.args)
    if not raw:
        return await update.message.reply_text("Usage: /compare <query>")

    q = await correct_query(raw)
    links = google_legal_search(q, 1) or search_cases_kanoon(q, 1)

    if not links:
        better = await suggest_better_query(q)
        return await update.message.reply_text(f"No cases found. Try `{better}`")

    text = fetch_case_text(links[0])
    comp = await compare_case_with_query(q, text)

    await update.message.reply_text(
        f"📎 *Case:* {links[0]}\n\n📊 *COMPARISON*\n\n{comp}",
        parse_mode="Markdown",
    )


# ============================
# ISSUES COMMAND
# ============================

async def issues_command(update, context):
    raw = " ".join(context.args)
    if not raw:
        return await update.message.reply_text("Usage: /issues <query>")

    q = await correct_query(raw)
    issues = await extract_legal_issues(q)

    await update.message.reply_text(f"📌 *LEGAL ISSUES*\n\n{issues}", parse_mode="Markdown")


# ============================
# FIR → QUERY → PDF
# ============================

async def uploadfir_command(update, context):
    await update.message.reply_text("⚙️ Processing FIR...")

    if not update.message.reply_to_message or not update.message.reply_to_message.document:
        return await update.message.reply_text("Reply to a FIR PDF and type /uploadfir")

    file = update.message.reply_to_message.document

    if not file.file_name.lower().endswith(".pdf"):
        return await update.message.reply_text("Only PDF files allowed.")

    os.makedirs("pdfs", exist_ok=True)
    file_path = f"pdfs/{file.file_name}"

    tg_file = await file.get_file()
    await tg_file.download_to_drive(file_path)

    text = extract_text_from_pdf(file_path)

    query = await convert_fir_to_query(text)
    await update.message.reply_text(f"🔍 Auto Query:\n`{query}`", parse_mode="Markdown")

    context.args = query.split()
    await pdf_command(update, context)


# ============================
# DEFAULT TEXT SEARCH
# ============================

async def reply_to_user(update, context):
    raw = update.message.text
    q = await correct_query(raw)

    links = google_legal_search(q, 2) or search_cases_kanoon(q, 2)

    if not links:
        better = await suggest_better_query(q)
        return await update.message.reply_text(f"No cases found. Try `{better}`")

    final = ""
    for link in links:
        t = fetch_case_text(link)
        s, ai = await summarize_case(t, q)
        rel = compute_relevance_score(ai, q, t)
        final += f"{link}\n{s}\nRelevance: {rel}%\n\n"

    await update.message.reply_text(final)


# ============================
# BOT STARTER — REGISTER HANDLERS
# ============================

def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CallbackQueryHandler(faq_button_handler))
    app.add_handler(CommandHandler("pdf", pdf_command))
    app.add_handler(CommandHandler("compare", compare_command))
    app.add_handler(CommandHandler("issues", issues_command))
    app.add_handler(CommandHandler("uploadfir", uploadfir_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, reply_to_user))

    logger.info("Bot running...")
    app.run_polling()


if __name__ == "__main__":
    main()

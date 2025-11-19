# ==========================================
# NYAYAAI TELEGRAM BOT — FULL FINAL VERSION
# WITH GOOGLE CSE + KANOON FALLBACK + PDF
# ==========================================

import os
import re
import uuid
import json
import fitz  # PyMuPDF
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
        text = ""
        for page in doc:
            text += page.get_text()
        return text.strip()
    except Exception as e:
        print("PDF extraction error:", e)
        return ""


def safe_get(url, headers=None, timeout=15):
    try:
        return requests.get(url, headers=headers or {"User-Agent": "Mozilla/5.0"}, timeout=timeout).text
    except Exception as e:
        print("HTTP error:", e)
        return ""


# ============================
# GOOGLE CUSTOM SEARCH
# ============================

def google_legal_search(query, max_results=5):
    """Unified search across multiple Indian legal sources."""
    if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
        print("Google CSE keys missing.")
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

        results = []
        if "items" not in data:
            return []

        for item in data["items"]:
            link = item.get("link", "")

            if any(domain in link for domain in [
                "indiankanoon.org",
                "main.sci.gov.in",
                "judis",
                "court",
                "highcourt",
                "hc",
                "judgment",
                "judgement"
            ]):
                results.append(link)

        return results[:max_results]

    except Exception as e:
        print("Google Search Error:", e)
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
                full_link = "https://indiankanoon.org" + a["href"]
                if full_link not in links:
                    links.append(full_link)
            if len(links) >= max_results:
                break

        return links
    except Exception as e:
        print("Kanoon error:", e)
        return []


# ============================
# FETCH JUDGMENT TEXT
# ============================

def fetch_case_text(url):
    try:
        html = safe_get(url)
        soup = BeautifulSoup(html, "html.parser")
        paragraphs = soup.find_all("p")
        text = "\n".join(p.get_text(strip=True) for p in paragraphs)
        return text[:10000]
    except:
        return ""


# ============================
# AI FUNCTIONS
# ============================

async def correct_query(user_query):
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Correct legal query. Fix typos, ensure IPC/CrPC numbers. Output only corrected query."},
            {"role": "user", "content": user_query}
        ]
    )
    return resp.choices[0].message.content.strip()


async def convert_fir_to_query(fir_text):
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Summarise FIR into 5–8 word legal search query. No names or locations."},
            {"role": "user", "content": fir_text}
        ]
    )
    return resp.choices[0].message.content.strip()


async def suggest_better_query(bad_query):
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Rewrite as strong legal search query. Output only query."},
            {"role": "user", "content": bad_query}
        ]
    )
    return resp.choices[0].message.content.strip()


async def summarize_case(text, query):
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Summarize judgment. Output: TITLE, SUMMARY, AI_SCORE:0-100"},
            {"role": "user", "content": text}
        ]
    )
    out = resp.choices[0].message.content

    ai_score = 0
    for line in out.split("\n"):
        if "AI_SCORE" in line:
            try:
                ai_score = int(line.split(":")[1])
            except:
                pass

    return out, ai_score


async def generate_ilac_note(query, text):
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Write ILAC: ISSUE, LAW, APPLICATION, CONCLUSION."},
            {"role": "user", "content": text}
        ]
    )
    return resp.choices[0].message.content


async def generate_arguments(query, text):
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "PETITIONER ARGUMENTS, RESPONDENT ARGUMENTS, COUNTER ARGUMENTS."},
            {"role": "user", "content": text}
        ]
    )
    return resp.choices[0].message.content


def extract_citations(text):
    case_pat = r"[A-Z][A-Za-z .]+ vs\.? [A-Z][A-Za-z .]+"
    statute_pat = r"(IPC\s*\d+|CrPC\s*\d+|Section\s*\d+|Evidence Act\s*\d+)"

    cases = list(dict.fromkeys(re.findall(case_pat, text)))
    statutes = list(dict.fromkeys(re.findall(statute_pat, text, flags=re.IGNORECASE)))

    out = "📌 CITED CASES:\n"
    out += "\n".join(f"- {c}" for c in cases) if cases else "None"
    out += "\n\n📌 STATUTES:\n"
    out += "\n".join(f"- {s}" for s in statutes) if statutes else "None"
    return out


async def compare_case_with_query(user_query, judgment_text):
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Compare legal query with judgment. Provide MATCHES, DIFFERENCES, IMPACT."},
            {"role": "user", "content": judgment_text}
        ]
    )
    return resp.choices[0].message.content


async def extract_legal_issues(user_query):
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Extract 3-6 legal issues from the query."},
            {"role": "user", "content": user_query}
        ]
    )
    return resp.choices[0].message.content.strip()


# ============================
# RELEVANCE SCORE
# ============================

def compute_relevance_score(ai, query, text):
    q = query.lower()
    t = text.lower()

    keyword_score = sum(5 for w in q.split() if w in t)
    keyword_score = min(keyword_score, 100)

    statute_boost = 0
    for n in re.findall(r"\b\d+\b", q):
        if n in t:
            statute_boost += 15
    statute_boost = min(statute_boost, 30)

    return min(int((0.6 * ai) + (0.3 * keyword_score) + statute_boost), 100)


# ============================
# PDF BUILDER
# ============================

def build_html(query, results, ilac, arguments, citations, qna):
    now = datetime.now().strftime("%d %b %Y %I:%M %p")

    header = f"""
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
# COMMAND HANDLERS
# ============================

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "👋 *Welcome to NyayaAI — India’s AI Legal Research Bot*\n\n"
        "Features:\n"
        "• Case search (Supreme + High Courts + Kanoon)\n"
        "• Auto-correct legal queries\n"
        "• Summaries & relevance scores\n"
        "• ILAC, arguments, citations\n"
        "• Compare cases (/compare)\n"
        "• Extract issues (/issues)\n"
        "• Full PDF report (/pdf)\n"
        "• FIR → Query → Case Laws (/uploadfir)\n\n"
        "Type /help for full menu."
    )
    await update.message.reply_text(msg, parse_mode="Markdown")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "📘 *NyayaAI Help Menu*\n\n"
        "Commands:\n"
        "/pdf <query> — full research report\n"
        "/uploadfir — reply to FIR PDF\n"
        "/compare <query>\n"
        "/issues <query>\n"
        "/help — show this\n"
    )
    await update.message.reply_text(msg, parse_mode="Markdown")


async def pdf_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        return await update.message.reply_text("Usage: /pdf <query>")

    raw = " ".join(context.args)
    q = await correct_query(raw)

    await update.message.reply_text("🔎 Searching...")

    links = google_legal_search(q, max_results=2)
    if not links:
        links = search_cases_kanoon(q, max_results=2)

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
    if " ? " in raw:
        _, question = raw.split(" ? ", 1)
        ans = await ask_about_case(question, first_text)
        qna = f"Q: {question}\n\n{ans}"

    await update.message.reply_text("📄 Generating PDF...")

    os.makedirs("pdfs", exist_ok=True)
    filename = f"pdfs/{uuid.uuid4().hex}.pdf"
    html = build_html(q, results, ilac, arguments, citations, qna)
    pdfkit.from_string(html, filename)

    await update.message.reply_document(open(filename, "rb"))


async def compare_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    raw = " ".join(context.args)
    if not raw:
        return await update.message.reply_text("Usage: /compare <query>")

    q = await correct_query(raw)

    links = google_legal_search(q, max_results=1)
    if not links:
        links = search_cases_kanoon(q, max_results=1)

    if not links:
        better = await suggest_better_query(q)
        return await update.message.reply_text(f"No cases found. Try `{better}`")

    text = fetch_case_text(links[0])
    comparison = await compare_case_with_query(q, text)

    await update.message.reply_text(
        f"📎 *Case:* {links[0]}\n\n📊 *COMPARISON*\n\n{comparison}",
        parse_mode="Markdown"
    )


async def issues_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    raw = " ".join(context.args)
    if not raw:
        return await update.message.reply_text("Usage: /issues <query>")

    q = await correct_query(raw)

    issues = await extract_legal_issues(q)
    await update.message.reply_text(f"📌 *LEGAL ISSUES*\n\n{issues}", parse_mode="Markdown")


async def uploadfir_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("⚙️ Processing FIR...")

    if not update.message.reply_to_message or not update.message.reply_to_message.document:
        return await update.message.reply_text("Reply to a FIR PDF and type /uploadfir")

    file = update.message.reply_to_message.document
    if not file.file_name.lower().endswith(".pdf"):
        return await update.message.reply_text("Only PDF allowed.")

    os.makedirs("pdfs", exist_ok=True)
    file_path = f"pdfs/{file.file_name}"

    tg_file = await file.get_file()
    await tg_file.download_to_drive(file_path)

    fir_text = extract_text_from_pdf(file_path)
    if not fir_text:
        return await update.message.reply_text("Could not read FIR.")

    smart_query = await convert_fir_to_query(fir_text)
    await update.message.reply_text(f"🔍 Auto Query:\n`{smart_query}`", parse_mode="Markdown")

    context.args = smart_query.split()
    await pdf_command(update, context)


# ============================
# DEFAULT TEXT HANDLER
# ============================

async def reply_to_user(update: Update, context: ContextTypes.DEFAULT_TYPE):
    raw = update.message.text

    q = await correct_query(raw)

    links = google_legal_search(q, max_results=2)
    if not links:
        links = search_cases_kanoon(q, max_results=2)

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
# BOT STARTER
# ============================

def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("pdf", pdf_command))
    app.add_handler(CommandHandler("uploadfir", uploadfir_command))
    app.add_handler(CommandHandler("compare", compare_command))
    app.add_handler(CommandHandler("issues", issues_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, reply_to_user))

    print("BOT RUNNING...")
    app.run_polling()


if __name__ == "__main__":
    main()

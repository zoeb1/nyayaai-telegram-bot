# ==========================================
# NYAYAAI SUPER-BOT — PART 1 OF 4
# IMPORTS • ENV • LOGGING • SAFE HELPERS • HELP MENU
# WITH FULL WEBSHARE SUPPORT
# ==========================================

import os
import re
import json
import uuid
import time
import fitz
import pdfkit
import random
import logging
import pathlib
import hashlib
import threading
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Tuple

import requests
from bs4 import BeautifulSoup

from dotenv import load_dotenv
load_dotenv()

from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup
)
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    filters,
    ContextTypes
)

from telegram.error import (
    BadRequest,
    RetryAfter,
    TimedOut,
    NetworkError,
    Conflict
)

from openai import OpenAI

# ====================================================
# ENV VARIABLES
# ====================================================

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

WEBSHARE_USERNAME = os.getenv("WEBSHARE_USERNAME")
WEBSHARE_PASSWORD = os.getenv("WEBSHARE_PASSWORD")

# ====================================================
# LOGGING
# ====================================================

LOG_DIR = pathlib.Path("logs")
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / "nyayaai.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)

logger = logging.getLogger("NyayaAI")
logger.info("=== NYAYAAI SUPER-BOT STARTING (WITH WEBSHARE) ===")

# ====================================================
# OPENAI CLIENT
# ====================================================

client = OpenAI(api_key=OPENAI_API_KEY)

# ====================================================
# ROTATING WEBSHARE PROXY
# ====================================================

def get_rotating_proxy():
    """Returns a new Webshare proxy session every request."""
    if not WEBSHARE_USERNAME or not WEBSHARE_PASSWORD:
        return None

    session = random.randint(100000, 999999)
    proxy_user = f"{WEBSHARE_USERNAME}-session-{session}"

    return f"http://{proxy_user}:{WEBSHARE_PASSWORD}@p.webshare.io:80"


def safe_request(url: str, timeout=20):
    """
    Uses Webshare proxy ONLY for IndianKanoon.
    Falls back to normal request if proxy fails.
    """
    headers = {
        "User-Agent":
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0 Safari/537.36"
    }

    proxy_url = get_rotating_proxy()
    use_proxy = "indiankanoon.org" in url.lower()

    proxies = {"http": proxy_url, "https": proxy_url} if use_proxy else None

    try:
        r = requests.get(url, headers=headers, proxies=proxies, timeout=timeout)
        r.raise_for_status()
        return r
    except Exception as e:
        logger.warning(f"Proxy failed for {url}. Retrying without proxy...")
        return requests.get(url, headers=headers, timeout=timeout)

# ====================================================
# SAFE TELEGRAM HELPERS
# ====================================================

async def safe_send(bot, chat_id, text, parse_mode=None, reply_markup=None):
    """Send message with retry & error handling."""
    for attempt in range(3):
        try:
            return await bot.send_message(
                chat_id=chat_id,
                text=text,
                parse_mode=parse_mode,
                reply_markup=reply_markup
            )
        except RetryAfter as e:
            time.sleep(e.retry_after + 1)
        except (TimedOut, NetworkError):
            time.sleep(2)
        except Conflict:
            logger.error("Bot conflict — another instance running.")
            return None
        except Exception:
            logger.exception("safe_send failed.")
            return None
    return None


async def safe_edit(bot, chat_id, msg_id, text, parse_mode=None, reply_markup=None):
    """Safe edit prevents 'message not modified' crash."""
    try:
        return await bot.edit_message_text(
            chat_id=chat_id,
            message_id=msg_id,
            text=text,
            parse_mode=parse_mode,
            reply_markup=reply_markup
        )
    except BadRequest as e:
        if "not modified" in str(e).lower():
            return None
    except Exception:
        logger.exception("safe_edit failed.")
        return None

# ====================================================
# USAGE ANALYTICS
# ====================================================

USAGE = {"messages": 0, "unique_users": set(), "commands": {}, "pdfs": 0}
USAGE_LOCK = threading.Lock()
USAGE_FILE = LOG_DIR / "usage.json"

def record_usage(uid, command=None):
    with USAGE_LOCK:
        USAGE["messages"] += 1
        if uid: USAGE["unique_users"].add(uid)
        if command:
            USAGE["commands"][command] = USAGE["commands"].get(command, 0) + 1

def snapshot_usage():
    while True:
        time.sleep(60)
        try:
            with USAGE_LOCK:
                USAGE_FILE.write_text(
                    json.dumps({
                        "messages": USAGE["messages"],
                        "unique_users": len(USAGE["unique_users"]),
                        "pdfs": USAGE["pdfs"],
                        "commands": USAGE["commands"],
                    }, indent=2)
                )
        except Exception:
            logger.exception("Usage snapshot failed.")

threading.Thread(target=snapshot_usage, daemon=True).start()

# ====================================================
# HELP MENU + FAQ BUTTONS
# ====================================================

HELP_TEXT = """
👋 *Welcome to NyayaAI* — India’s smartest AI legal research assistant.

I can do:
• Supreme Court & High Court case search  
• IndianKanoon search (with Webshare proxy)  
• Auto-correct legal queries  
• FIR → Query → Case Laws  
• ILAC + Arguments + Citations  
• PDF Research Reports  
• Compare judgments  
• Legal issue extraction  

Choose an option:
"""

FAQ_DATA = {
    "WHAT": "*What is NyayaAI?*\n\nNyayaAI is an AI-powered Indian legal research engine.",
    "FIR": "*How to upload FIR?*\n\nReply to FIR PDF → /uploadfir",
    "PDF": "*How to generate PDF?*\n\nUse: `/pdf IPC 420 cheating`",
    "PRICING": "*Pricing*\n\nFree for now. Pro coming soon.",
    "CONTACT": "*Contact*\n\nsupport@nyayaai.com",
    "PRIVACY": "*Privacy*\n\nFIRs are NOT stored.",
    "EXAMPLES": "*Examples*\n\n`IPC 307 attempt murder`\n`Dowry harassment 498A`",
}

FAQ_MENU = InlineKeyboardMarkup([
    [InlineKeyboardButton("What is NyayaAI?", callback_data="WHAT"),
     InlineKeyboardButton("FIR Help", callback_data="FIR")],
    [InlineKeyboardButton("PDF Guide", callback_data="PDF"),
     InlineKeyboardButton("Pricing", callback_data="PRICING")],
    [InlineKeyboardButton("Feedback", callback_data="CONTACT"),
     InlineKeyboardButton("Privacy", callback_data="PRIVACY")],
    [InlineKeyboardButton("Examples", callback_data="EXAMPLES"),
     InlineKeyboardButton("Close", callback_data="CLOSE")],
])

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    record_usage(update.effective_user.id, "/start")
    await safe_send(
        context.bot,
        update.effective_chat.id,
        HELP_TEXT,
        parse_mode="Markdown",
        reply_markup=FAQ_MENU
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    record_usage(update.effective_user.id, "/help")
    await safe_send(
        context.bot,
        update.effective_chat.id,
        HELP_TEXT,
        parse_mode="Markdown",
        reply_markup=FAQ_MENU
    )

async def faq_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    key = query.data
    chat = query.message.chat_id
    msg = query.message.message_id

    if key == "CLOSE":
        await safe_edit(context.bot, chat, msg, "Menu closed.")
        return

    text = FAQ_DATA.get(key, "Unknown option.")
    back = InlineKeyboardMarkup([[InlineKeyboardButton("Back", callback_data="BACK")]])

    await safe_edit(context.bot, chat, msg, text, parse_mode="Markdown", reply_markup=back)

async def faq_back(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    chat = query.message.chat_id
    msg = query.message.message_id

    await safe_edit(context.bot, chat, msg, HELP_TEXT, parse_mode="Markdown", reply_markup=FAQ_MENU)
# ==========================================
# NYAYAAI SUPER-BOT — PART 2 OF 4
# SEARCH ENGINE • KANOON SCRAPER • AI HELPERS
# WITH WEBSHARE PROXY PROTECTION
# ==========================================

# ====================================================
# CACHING SYSTEM
# ====================================================

CACHE: Dict[str, Dict[str, Any]] = {}
CACHE_LOCK = threading.Lock()
CACHE_TTL = 3600  # 1 hour

def make_key(*parts):
    return hashlib.sha256("|".join(str(x) for x in parts).encode()).hexdigest()

def cache_get(key):
    with CACHE_LOCK:
        item = CACHE.get(key)
        if not item:
            return None
        if time.time() > item["expiry"]:
            del CACHE[key]
            return None
        return item["value"]

def cache_set(key, value, ttl=CACHE_TTL):
    with CACHE_LOCK:
        CACHE[key] = {"value": value, "expiry": time.time() + ttl}

# ====================================================
# GOOGLE CUSTOM SEARCH
# ====================================================

def google_legal_search(query: str, max_results=5) -> List[str]:
    """Search Indian legal sources via Google CSE."""
    if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
        return []

    cache_key = make_key("google_cse", query)
    cached = cache_get(cache_key)
    if cached:
        return cached

    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_API_KEY,
        "cx": GOOGLE_CSE_ID,
        "q": f"{query} judgment case law site:indiankanoon.org OR site:sci.gov.in",
        "num": max_results
    }

    try:
        r = requests.get(url, params=params, timeout=15)
        data = r.json()
        links = []

        for item in data.get("items", []):
            link = item.get("link")
            if not link:
                continue
            if any(domain in link for domain in [
                "indiankanoon.org",
                "sci.gov.in",
                "judis.nic.in",
                "highcourt"
            ]):
                links.append(link)
            if len(links) >= max_results:
                break

        cache_set(cache_key, links, ttl=1800)
        return links

    except Exception:
        logger.exception("Google CSE search failed")
        return []

# ====================================================
# INDIANKANOON FALLBACK (WITH WEBSHARE)
# ====================================================

def search_cases_kanoon(query: str, max_results=3) -> List[str]:
    """Scrapes IndianKanoon using rotating Webshare proxy."""
    cache_key = make_key("kanoon", query)
    cached = cache_get(cache_key)
    if cached:
        return cached

    try:
        q = query.replace(" ", "+")
        url = f"https://indiankanoon.org/search/?formInput={q}"

        response = safe_request(url, timeout=25)
        soup = BeautifulSoup(response.text, "html.parser")

        links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.startswith("/doc/"):
                full_url = "https://indiankanoon.org" + href
                links.append(full_url)
                if len(links) >= max_results:
                    break

        cache_set(cache_key, links, ttl=1800)
        return links

    except Exception:
        logger.exception("IndianKanoon scraper failed")
        return []

# ====================================================
# FETCH CASE TEXT (HTML + PDF)
# ====================================================

def extract_text_from_pdf(path: str) -> str:
    """Extract text from PDF using PyMuPDF."""
    try:
        doc = fitz.open(path)
        output = ""
        for page in doc:
            output += page.get_text()
        return output.strip()
    except Exception:
        return ""

def fetch_case_text(url: str) -> str:
    """Fetch judgment text from Kanoon, SCI PDFs, etc."""
    cache_key = make_key("case_text", url)
    cached = cache_get(cache_key)
    if cached:
        return cached

    try:
        if url.lower().endswith(".pdf"):
            # PDF Handling
            response = safe_request(url, timeout=30)
            tmp = f"pdfs/{uuid.uuid4().hex}.pdf"
            with open(tmp, "wb") as f:
                f.write(response.content)
            text = extract_text_from_pdf(tmp)
            os.remove(tmp)

        else:
            # HTML Handling
            response = safe_request(url, timeout=25)
            soup = BeautifulSoup(response.text, "html.parser")

            # Prefer judgement containers
            blocks = soup.select(".judgement, .judgment, #content, .doc, .content, p")
            text = "\n".join(el.get_text(separator="\n", strip=True) for el in blocks)

        text = text.strip()[:15000]  # limit
        cache_set(cache_key, text)
        return text

    except Exception:
        logger.exception(f"Failed to fetch text: {url}")
        return ""

# ====================================================
# AI HELPERS
# ====================================================

async def openai_chat(messages, model="gpt-4o-mini", retries=2):
    """Unified OpenAI wrapper."""
    for attempt in range(retries + 1):
        try:
            return client.chat.completions.create(model=model, messages=messages)
        except Exception as e:
            if attempt == retries:
                raise e
            await asyncio.sleep(1 + attempt)

# ---------- Query Correction ----------
async def correct_query(raw: str) -> str:
    messages = [
        {"role": "system", "content": "Correct and normalize Indian legal queries. Keep it short."},
        {"role": "user", "content": raw}
    ]
    resp = await openai_chat(messages)
    return resp.choices[0].message.content.strip()

# ---------- Summarize Judgment ----------
async def summarize_case(text: str, query: str) -> Tuple[str, int]:
    if not text:
        return ("No text extracted.", 0)

    messages = [
        {"role": "system", "content":
            "Summarize Indian court judgments.\n"
            "Output format:\n"
            "TITLE: …\nSUMMARY: …\nKEY POINTS: …\nAI_SCORE: <0-100>"
        },
        {"role": "user", "content": f"QUERY: {query}\nTEXT:\n{text[:8000]}"}
    ]

    resp = await openai_chat(messages)
    out = resp.choices[0].message.content

    score = 0
    for line in out.splitlines():
        if "AI_SCORE" in line.upper():
            try:
                score = int(re.search(r"\d+", line).group())
            except:
                pass

    return out, score

# ---------- ILAC ----------
async def generate_ilac(query: str, text: str) -> str:
    messages = [
        {"role": "system", "content": "Write ILAC: ISSUE, LAW, APPLICATION, CONCLUSION."},
        {"role": "user", "content": f"QUERY: {query}\nTEXT:\n{text[:5000]}"}
    ]
    resp = await openai_chat(messages)
    return resp.choices[0].message.content

# ---------- Arguments ----------
async def generate_arguments(query: str, text: str) -> str:
    messages = [
        {"role": "system", "content": "Write PETITIONER ARGUMENTS, RESPONDENT ARGUMENTS, COUNTER ARGUMENTS."},
        {"role": "user", "content": text[:5000]}
    ]
    resp = await openai_chat(messages)
    return resp.choices[0].message.content

# ---------- Citations ----------
def extract_citations(text: str) -> str:
    case_pattern = r"[A-Z][A-Za-z .]+ vs\.? [A-Z][A-Za-z .]+"
    statute_pattern = r"(IPC\s*\d+|CrPC\s*\d+|Section\s*\d+|Evidence Act\s*\d+)"

    cases = list(dict.fromkeys(re.findall(case_pattern, text)))
    statutes = list(dict.fromkeys(re.findall(statute_pattern, text, flags=re.IGNORECASE)))

    out = "📌 CITED CASES:\n" + ("\n".join(f"- {c}" for c in cases) if cases else "None")
    out += "\n\n📌 STATUTES:\n" + ("\n".join(f"- {s}" for s in statutes) if statutes else "None")
    return out

# ---------- Compare ----------
async def compare_case_with_query(query: str, text: str) -> str:
    messages = [
        {"role": "system", "content": "Compare legal query with judgment. Give MATCHES, DIFFERENCES, IMPACT."},
        {"role": "user", "content": text[:6000]}
    ]
    resp = await openai_chat(messages)
    return resp.choices[0].message.content

# ---------- Issue Extraction ----------
async def extract_legal_issues(query: str) -> str:
    messages = [
        {"role": "system", "content": "Extract 3–6 legal issues as questions."},
        {"role": "user", "content": query}
    ]
    resp = await openai_chat(messages)
    return resp.choices[0].message.content.strip()

# ---------- Q&A based on judgment ----------
async def ask_about_case(question: str, text: str) -> str:
    messages = [
        {"role": "system", "content": "Answer ONLY from the judgment. If answer not found, say so."},
        {"role": "user", "content": f"Q: {question}\n\nTEXT:\n{text[:7000]}"}
    ]
    resp = await openai_chat(messages)
    return resp.choices[0].message.content

# ====================================================
# DEEP SEARCH (Google → Kanoon → SCI fallback)
# ====================================================

def deep_search(query: str) -> List[str]:
    """Multi-layer fallback search."""
    links = google_legal_search(query, max_results=3)
    if links:
        return links

    links = search_cases_kanoon(query, max_results=3)
    if links:
        return links

    # Supreme Court fallback by year scanning
    years = re.findall(r"20\d{2}|19\d{2}", query)
    if years:
        year = years[0]
        try:
            url = f"https://main.sci.gov.in/supremecourt/{year}/"
            r = safe_request(url)
            soup = BeautifulSoup(r.text, "html.parser")

            pdf_links = [
                "https://main.sci.gov.in" + a["href"]
                for a in soup.find_all("a", href=True)
                if a["href"].lower().endswith(".pdf")
            ]

            if pdf_links:
                return pdf_links[:1]
        except:
            pass

    return []
# ==========================================
# NYAYAAI SUPER-BOT — PART 3 OF 4
# PDF BUILDER • MAIN COMMANDS • HELP MENU
# ==========================================

# ============================================================
# PDF REPORT BUILDER
# ============================================================

def build_html_report(query: str, results: List[Dict], ilac: str, arguments: str,
                      citations: str, qna: str = "") -> str:
    now = datetime.now().strftime("%d %b %Y, %I:%M %p")

    header = f"""
    <div style="background:#0f4c81;color:white;padding:20px;border-radius:10px;">
        <h1>NyayaAI</h1>
        <p>AI Legal Research — {now}</p>
    </div>
    <h2>Query: {query}</h2>
    <hr>
    """

    body = ""
    for i, r in enumerate(results, 1):
        body += f"""
        <h3>Case {i} • Relevance: {r['relevance']}%</h3>
        <p><a href="{r['link']}">{r['link']}</a></p>
        <pre style="background:#f7f7f7;padding:15px;border-left:4px solid #0f4c81;">
{r['summary']}
        </pre>
        <hr>
        """

    extra = f"""
    <h3>ILAC</h3>
    <pre>{ilac or 'ILAC unavailable'}</pre>

    <h3>Arguments</h3>
    <pre>{arguments or 'Arguments unavailable'}</pre>

    <h3>Citations</h3>
    <pre>{citations}</pre>
    """

    if qna:
        extra += f"<h3>Q&A</h3><pre>{qna}</pre>"

    footer = """
    <p style="font-size:12px;color:gray;">
        Generated by NyayaAI • Not legal advice • For research only
    </p>
    """

    return f"<html><body style='font-family:Arial;margin:25px;'>{header}{body}{extra}{footer}</body></html>"


# ============================================================
# /start COMMAND — INTRO MESSAGE
# ============================================================

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "👋 *Welcome to NyayaAI — India’s AI Legal Research Bot*\n\n"
        "I can help you with:\n"
        "• Supreme Court + High Court case search\n"
        "• IndianKanoon scraping (with proxy)\n"
        "• FIR → Auto Query → Case laws\n"
        "• Summaries, ILAC, Arguments, Citations\n"
        "• Full PDF research notes (/pdf)\n"
        "• Compare a case with a query (/compare)\n"
        "• Extract legal issues (/issues)\n\n"
        "Tap /help to view full menu."
    )
    await safe_send(context.bot, update.effective_chat.id, msg, parse_mode="Markdown")


# ============================================================
# HELP MENU + FAQ BUTTONS
# ============================================================

HELP_MENU = InlineKeyboardMarkup([
    [
        InlineKeyboardButton("❓ FAQ", callback_data="help_faq"),
        InlineKeyboardButton("ℹ️ About", callback_data="help_about")
    ],
    [
        InlineKeyboardButton("💰 Pricing", callback_data="help_pricing"),
        InlineKeyboardButton("📞 Contact", callback_data="help_contact")
    ],
    [
        InlineKeyboardButton("📘 Examples", callback_data="help_examples"),
        InlineKeyboardButton("✖ Close", callback_data="help_close")
    ]
])

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "📚 *NyayaAI Help Menu*\n\n"
        "Choose one:"
    )
    await safe_send(context.bot, update.effective_chat.id, text, parse_mode="Markdown", reply_markup=HELP_MENU)


async def help_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()

    if q.data == "help_close":
        await q.edit_message_text("Help menu closed.")
        return

    if q.data == "help_faq":
        text = (
            "❓ *Frequently Asked Questions*\n\n"
            "• Where does data come from? — Google CSE + SCI + High Courts + IndianKanoon.\n"
            "• Is this legal advice? — No, only research support.\n"
            "• Can it read FIR PDFs? — Yes, use /uploadfir.\n"
            "• Can it generate PDF? — Use /pdf <query>."
        )
        await q.edit_message_text(text, parse_mode="Markdown", reply_markup=HELP_MENU)

    elif q.data == "help_about":
        await q.edit_message_text(
            "ℹ️ *About NyayaAI*\n\n"
            "NyayaAI automates legal research using AI + Indian legal data.\n"
            "It provides case search, summaries, ILAC, arguments and more.",
            parse_mode="Markdown",
            reply_markup=HELP_MENU
        )

    elif q.data == "help_pricing":
        await q.edit_message_text(
            "💰 *Pricing (Coming Soon)*\n\n"
            "🔹 Free Tier — 15 searches/day.\n"
            "🔹 Pro Tier — Unlimited searches, faster responses, enhanced PDF reports.",
            parse_mode="Markdown",
            reply_markup=HELP_MENU
        )

    elif q.data == "help_contact":
        await q.edit_message_text(
            "📞 *Contact & Feedback*\n\n"
            "Email: support@nyayaai.com\nTelegram: @nyayaai_support",
            parse_mode="Markdown",
            reply_markup=HELP_MENU
        )

    elif q.data == "help_examples":
        await q.edit_message_text(
            "📘 *Examples*\n\n"
            "`IPC 420 cheating`\n"
            "`Section 304A negligence`\n"
            "`anticipatory bail domestic violence`\n"
            "`FIR quashing 498A`\n",
            parse_mode="Markdown",
            reply_markup=HELP_MENU
        )


# ============================================================
# /pdf COMMAND
# ============================================================

async def pdf_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await safe_send(context.bot, update.effective_chat.id,
                        "Usage: /pdf <legal query>\nExample: `/pdf IPC 302 murder`",
                        parse_mode="Markdown")
        return

    query = " ".join(context.args)
    await safe_send(context.bot, update.effective_chat.id, "🔎 Searching judgments...")

    try:
        corrected = await correct_query(query)
    except:
        corrected = query

    links = deep_search(corrected)
    if not links:
        await safe_send(context.bot, update.effective_chat.id, "No case laws found.")
        return

    results = []
    first_text = ""

    for link in links:
        text = fetch_case_text(link)
        first_text = first_text or text

        summary, ai_score = await summarize_case(text, corrected)
        relevance = min(100, ai_score + 15)

        results.append({
            "link": link,
            "summary": summary,
            "relevance": relevance
        })

    ilac = await generate_ilac(corrected, first_text)
    arguments = await generate_arguments(corrected, first_text)
    citations = extract_citations(first_text)

    await safe_send(context.bot, update.effective_chat.id, "📄 Building PDF...")

    os.makedirs("pdfs", exist_ok=True)
    filename = f"pdfs/{uuid.uuid4().hex}.pdf"

    html = build_html_report(corrected, results, ilac, arguments, citations)
    pdfkit.from_string(html, filename)

    await context.bot.send_document(update.effective_chat.id, open(filename, "rb"),
                                    caption="📘 NyayaAI Research Report")

    os.remove(filename)


# ============================================================
# /compare COMMAND
# ============================================================

async def compare_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await safe_send(context.bot, update.effective_chat.id,
                        "Usage: /compare <query>")
        return

    query = " ".join(context.args)

    try:
        corrected = await correct_query(query)
    except:
        corrected = query

    links = deep_search(corrected)
    if not links:
        await safe_send(context.bot, update.effective_chat.id, "No cases found.")
        return

    text = fetch_case_text(links[0])
    comparison = await compare_case_with_query(corrected, text)

    await safe_send(context.bot, update.effective_chat.id,
                    f"📎 Case: {links[0]}\n\n📊 RESULT:\n{comparison}")


# ============================================================
# /issues COMMAND
# ============================================================

async def issues_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await safe_send(context.bot, update.effective_chat.id, "Usage: /issues <query>")
        return

    query = " ".join(context.args)

    try:
        corrected = await correct_query(query)
    except:
        corrected = query

    issues = await extract_legal_issues(corrected)
    await safe_send(context.bot, update.effective_chat.id,
                    f"📌 *LEGAL ISSUES*\n\n{issues}", parse_mode="Markdown")


# ============================================================
# /uploadfir COMMAND (reply to PDF)
# ============================================================

async def uploadfir_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message.reply_to_message or not update.message.reply_to_message.document:
        await safe_send(context.bot, update.effective_chat.id, "Reply to a PDF with /uploadfir")
        return

    doc = update.message.reply_to_message.document
    if not doc.file_name.lower().endswith(".pdf"):
        await safe_send(context.bot, update.effective_chat.id, "Only PDF files supported.")
        return

    os.makedirs("pdfs", exist_ok=True)
    path = f"pdfs/{doc.file_id}.pdf"

    try:
        tg_file = await doc.get_file()
        await tg_file.download_to_drive(path)
    except:
        await safe_send(context.bot, update.effective_chat.id, "Failed to download PDF.")
        return

    text = extract_text_from_pdf(path)
    if not text:
        await safe_send(context.bot, update.effective_chat.id, "Could not extract text from PDF.")
        return

    query = await correct_query(text[:1000])
    await safe_send(context.bot, update.effective_chat.id, f"Auto Query:\n`{query}`", parse_mode="Markdown")

    context.args = query.split()
    await pdf_command(update, context)


# ============================================================
# DEFAULT MESSAGE HANDLER
# ============================================================

async def reply_to_user(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.message.text.strip()
    await safe_send(context.bot, update.effective_chat.id, "🔍 Searching...")

    try:
        corrected = await correct_query(query)
    except:
        corrected = query

    links = deep_search(corrected)
    if not links:
        await safe_send(context.bot, update.effective_chat.id, "No matching cases found.")
        return

    final = ""
    for link in links:
        text = fetch_case_text(link)
        summary, score = await summarize_case(text, corrected)
        rel = min(100, score + 10)

        final += f"{link}\n\n{summary}\n\nRelevance: {rel}%\n\n{'─'*30}\n\n"

    for chunk in [final[i:i+3500] for i in range(0, len(final), 3500)]:
        await safe_send(context.bot, update.effective_chat.id, chunk)


# ============================================================
# GLOBAL ERROR HANDLER
# ============================================================

async def global_error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.exception("Error occurred: %s", context.error)
    if isinstance(update, Update) and update.effective_chat:
        await safe_send(context.bot, update.effective_chat.id,
                        "⚠️ An internal error occurred. Logged for analysis.")
# ==========================================
# NYAYAAI SUPER-BOT — PART 4 OF 4
# MAIN FUNCTION • REGISTER ALL HANDLERS
# ==========================================

def main():
    if not TELEGRAM_TOKEN:
        logger.error("❌ TELEGRAM_TOKEN missing in Railway Variables!")
        return

    logger.info("🚀 NyayaAI Booting… (with Webshare proxy protection)")

    # Build the Telegram bot app
    app = ApplicationBuilder()\
        .token(TELEGRAM_TOKEN)\
        .concurrent_updates(True)\
        .build()

    # ----------------------------
    # REGISTER COMMAND HANDLERS
    # ----------------------------
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))

    # Extended help pages
    app.add_handler(CallbackQueryHandler(help_callback))

    # Case law commands
    app.add_handler(CommandHandler("pdf", pdf_command))
    app.add_handler(CommandHandler("compare", compare_command))
    app.add_handler(CommandHandler("issues", issues_command))

    # FIR → auto-search
    app.add_handler(CommandHandler("uploadfir", uploadfir_command))

    # ----------------------------
    # DEFAULT MESSAGE HANDLER
    # ----------------------------
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, reply_to_user))

    # ----------------------------
    # GLOBAL ERROR HANDLER
    # ----------------------------
    app.add_error_handler(global_error_handler)

    # ----------------------------
    # LAUNCH BOT
    # ----------------------------
    logger.info("✅ NyayaAI is LIVE! Waiting for queries…")
    app.run_polling(drop_pending_updates=True)


# ==========================================
# RUN MAIN
# ==========================================

if __name__ == "__main__":
    main()

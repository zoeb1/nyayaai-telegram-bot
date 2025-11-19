# ==========================================
# NYAYAAI — FULL SUPER-BOT (PART 1 OF 4)
# CORE IMPORTS, LOGGING, HELP MENU, SAFE I/O
# ==========================================

import os
import re
import json
import uuid
import time
import fitz   # PyMuPDF
import pdfkit
import logging
import pathlib
import threading
from datetime import datetime
from typing import Optional

import requests
from bs4 import BeautifulSoup

from telegram import (
    Update,
    InlineKeyboardMarkup,
    InlineKeyboardButton
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
    Conflict,
    NetworkError
)

from openai import OpenAI


# ============================================================
# ENVIRONMENT VARIABLES
# ============================================================

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

client = OpenAI(api_key=OPENAI_API_KEY)


# ============================================================
# LOGGING SYSTEM (Console + File)
# ============================================================

LOG_DIR = pathlib.Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "nyayaai.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, encoding="utf-8")
    ]
)
logger = logging.getLogger("NyayaAI")

logger.info("========= NyayaAI BOT STARTING =========")
logger.info("GOOGLE_API_KEY=%s", GOOGLE_API_KEY)
logger.info("GOOGLE_CSE_ID=%s", GOOGLE_CSE_ID)
logger.info("OPENAI_API_KEY Exists=%s", bool(OPENAI_API_KEY))
logger.info("=========================================")


# ============================================================
# SAFE MESSAGE HELPERS — PREVENTS CRASHES & ERRORS
# ============================================================

async def safe_send(bot, chat_id, text, parse_mode=None, reply_markup=None):
    """Safe send with retries."""
    for attempt in range(3):
        try:
            return await bot.send_message(
                chat_id=chat_id,
                text=text,
                parse_mode=parse_mode,
                reply_markup=reply_markup
            )
        except RetryAfter as e:
            time.sleep(e.retry_after)
        except (TimedOut, NetworkError):
            time.sleep(1)
        except Conflict:
            logger.error("Bot conflict: two instances running")
            return None
        except Exception:
            logger.exception("Safe send failed")
            return None
    return None


async def safe_edit(bot, chat_id, msg_id, text, parse_mode=None, reply_markup=None):
    """Safe edit message."""
    try:
        return await bot.edit_message_text(
            chat_id=chat_id,
            message_id=msg_id,
            text=text,
            parse_mode=parse_mode,
            reply_markup=reply_markup
        )
    except BadRequest as e:
        if "not modified" in str(e):
            return None
        logger.error("Bad edit: %s", e)
        return None
    except Exception:
        logger.exception("Safe edit failed")
        return None


# ============================================================
# USAGE ANALYTICS — TRACKS USERS & COMMANDS
# ============================================================

USAGE = {
    "messages": 0,
    "unique_users": set(),
    "pdfs": 0,
    "commands": {}
}
USAGE_LOCK = threading.Lock()
USAGE_FILE = LOG_DIR / "usage.json"


def record_usage(uid, command=None):
    with USAGE_LOCK:
        USAGE["messages"] += 1
        if uid:
            USAGE["unique_users"].add(uid)
        if command:
            USAGE["commands"].setdefault(command, 0)
            USAGE["commands"][command] += 1


def snapshot_usage():
    """Dump usage stats every 60 seconds."""
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
            logger.exception("Could not save usage snapshot")


threading.Thread(target=snapshot_usage, daemon=True).start()


# ============================================================
# HELP & FAQ SYSTEM
# ============================================================

HELP_TEXT = """
📘 *NyayaAI — India’s AI Legal Research Assistant*

I can:
• Search Supreme Court + all High Courts  
• Analyse Indian Kanoon  
• Auto-correct legal queries  
• Extract FIR → Auto Query → Case laws  
• Summaries, ILAC, arguments, citations  
• Full PDF reports  
• Case comparison, issue extraction  

Tap a button below:
"""

FAQ_DATA = {
    "FAQ_WHAT": "*What does NyayaAI do?*\n\nI research Indian case law...",
    "FAQ_FIR": "*How to use FIR upload?*\n\nReply to a PDF with /uploadfir.",
    "FAQ_PDF": "*How to generate PDF?*\n\nUse `/pdf <query>`.",
    "FAQ_PRICING": "*Pricing*\n\nCurrently free. Premium coming soon.",
    "FAQ_CONTACT": "*Contact & Feedback*\n\nEmail: zoebsadeqa@gmail.com",
    "FAQ_PRIVACY": "*Privacy*\n\nWe do not store FIRs.",
    "FAQ_EXAMPLES": "*Examples*\n\n`IPC 420 cheating` → top judgments.",
}

FAQ_MENU = InlineKeyboardMarkup([
    [
        InlineKeyboardButton("What is NyayaAI?", callback_data="FAQ_WHAT"),
        InlineKeyboardButton("FIR Help", callback_data="FAQ_FIR"),
    ],
    [
        InlineKeyboardButton("PDF Guide", callback_data="FAQ_PDF"),
        InlineKeyboardButton("Pricing", callback_data="FAQ_PRICING"),
    ],
    [
        InlineKeyboardButton("Feedback", callback_data="FAQ_CONTACT"),
        InlineKeyboardButton("Privacy", callback_data="FAQ_PRIVACY"),
    ],
    [
        InlineKeyboardButton("Examples", callback_data="FAQ_EXAMPLES"),
        InlineKeyboardButton("Close", callback_data="FAQ_CLOSE"),
    ],
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
    chat_id = query.message.chat_id
    msg_id = query.message.message_id

    if key == "FAQ_CLOSE":
        await safe_edit(context.bot, chat_id, msg_id, "Menu closed.")
        return

    text = FAQ_DATA.get(key, "Unknown option.")
    back_btn = InlineKeyboardMarkup(
        [[InlineKeyboardButton("Back", callback_data="HELP_MENU")]]
    )

    await safe_edit(
        context.bot,
        chat_id,
        msg_id,
        text,
        parse_mode="Markdown",
        reply_markup=back_btn
    )


# ---------------- PART 2: SEARCH STACK, SCRAPERS, CACHING, AI HELPERS ----------------

import hashlib
from typing import List, Tuple, Dict, Any
import asyncio

# ---------- Simple in-memory cache with TTL to reduce repeated API calls
CACHE: Dict[str, Dict[str, Any]] = {}
CACHE_LOCK = threading.Lock()
DEFAULT_TTL = 60 * 60  # 1 hour

def cache_get(key: str):
    with CACHE_LOCK:
        entry = CACHE.get(key)
        if not entry:
            return None
        if time.time() > entry["expires_at"]:
            del CACHE[key]
            return None
        return entry["value"]

def cache_set(key: str, value: Any, ttl: int = DEFAULT_TTL):
    with CACHE_LOCK:
        CACHE[key] = {
            "value": value,
            "expires_at": time.time() + ttl
        }

def make_key(*parts):
    raw = "|".join(str(p) for p in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

# ---------- Google Custom Search wrapper (returns list of links)
def google_legal_search(query: str, max_results: int = 5) -> List[str]:
    """
    Uses Google Custom Search JSON API. Filters results by likely legal domains.
    Returns list of URLs or empty list.
    """
    if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
        logger.info("google_legal_search: Missing Google keys.")
        return []

    cache_key = make_key("google", query, max_results)
    cached = cache_get(cache_key)
    if cached:
        logger.debug("google_legal_search: cache hit")
        return cached

    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_API_KEY,
        "cx": GOOGLE_CSE_ID,
        "q": f"{query} judgment case law",
        "num": min(max_results, 10)
    }

    try:
        r = requests.get(url, params=params, timeout=12)
        data = r.json()
        items = data.get("items", [])
        links: List[str] = []
        for it in items:
            link = it.get("link")
            if not link:
                continue
            # filter by known legal patterns to reduce noise
            if any(p in link for p in ("indiankanoon.org", "judis", "sci.gov.in", "highcourt", "court", ".nic.in", "judgment", "judgement")):
                links.append(link)
            # also include PDFs (gov sites)
            elif link.lower().endswith(".pdf"):
                links.append(link)
            if len(links) >= max_results:
                break

        cache_set(cache_key, links, ttl=60 * 30)  # cache 30 min
        logger.info("google_legal_search: found %d links for query '%s'", len(links), query)
        return links

    except Exception as e:
        logger.exception("google_legal_search error")
        return []

# ---------- IndianKanoon scraper fallback
def search_cases_kanoon(query: str, max_results: int = 2) -> List[str]:
    """
    Scrapes IndianKanoon search results page for top judgment links.
    """
    cache_key = make_key("kanoon", query, max_results)
    cached = cache_get(cache_key)
    if cached:
        logger.debug("search_cases_kanoon: cache hit")
        return cached

    try:
        q = query.replace(" ", "+")
        url = f"https://indiankanoon.org/search/?formInput={q}"
        html = safe_get(url, timeout=15) if 'safe_get' in globals() else requests.get(url, headers={"User-Agent":"Mozilla/5.0"}).text
        soup = BeautifulSoup(html, "html.parser")

        links = []
        # look for /doc/ links which are judgments
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.startswith("/doc/"):
                full = "https://indiankanoon.org" + href
                if full not in links:
                    links.append(full)
            if len(links) >= max_results:
                break

        cache_set(cache_key, links, ttl=60 * 30)
        logger.info("search_cases_kanoon: found %d links for '%s'", len(links), query)
        return links
    except Exception as e:
        logger.exception("search_cases_kanoon error")
        return []

# ---------- Fetch judgment text (supports HTML pages and PDF links)
def fetch_case_text(url: str) -> str:
    """
    Fetches judgment text. For PDFs, tries to download and extract with PyMuPDF.
    For HTML pages, extracts <p> text.
    Returns up to ~12k chars to limit cost.
    """
    cache_key = make_key("case_text", url)
    cached = cache_get(cache_key)
    if cached:
        logger.debug("fetch_case_text: cache hit")
        return cached

    try:
        logger.info("Fetching case text: %s", url)
        if url.lower().endswith(".pdf"):
            # download PDF
            r = requests.get(url, timeout=15)
            tmp = f"pdfs/{uuid.uuid4().hex}.pdf"
            with open(tmp, "wb") as fh:
                fh.write(r.content)
            text = extract_text_from_pdf(tmp)
            # optional: delete tmp
            try:
                os.remove(tmp)
            except Exception:
                pass
        else:
            html = safe_get(url)
            soup = BeautifulSoup(html, "html.parser")
            # prioritize <div class="judgment"> or similar patterns, else p tags
            text_nodes = []
            # common selectors for legal judgment sites
            candidates = soup.select(".judgment, .judgement, #content, .casebody, .doc") or []
            if candidates:
                for c in candidates:
                    text_nodes.append(c.get_text(separator="\n", strip=True))
            else:
                paras = soup.find_all("p")
                for p in paras:
                    text_nodes.append(p.get_text(strip=True))
            text = "\n".join(text_nodes)

        text = text.strip()
        if not text:
            logger.warning("fetch_case_text: no text extracted for %s", url)

        # limit
        text = text[:12000]
        cache_set(cache_key, text, ttl=60 * 60)  # cache 1 hour
        return text
    except Exception:
        logger.exception("fetch_case_text error for: %s", url)
        return ""

# ---------- OpenAI wrapper helpers (async-friendly)
async def openai_chat_completion(messages: List[dict], model: str = "gpt-4o-mini", max_retries: int = 2) -> dict:
    """
    Calls OpenAI chat completions via client with retries. Returns the raw response dict.
    """
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            logger.debug("openai_chat_completion: sending to model %s", model)
            resp = client.chat.completions.create(model=model, messages=messages)
            return resp
        except Exception as e:
            last_exc = e
            logger.warning("OpenAI call failed (attempt %s): %s", attempt, e)
            # simple backoff
            await asyncio.sleep(1 + attempt * 2)
            continue
    logger.exception("openai_chat_completion failed after retries: %s", last_exc)
    raise last_exc

# ---------- AI: Summarize judgment
async def summarize_case(text: str, query: str) -> Tuple[str, int]:
    """
    Returns (summary_text, ai_score)
    Summary format should contain AI_SCORE:nn line which we parse.
    """
    if not text:
        return ("No text to summarize.", 0)

    cache_key = make_key("summary", hashlib.sha256((query + text[:500]).encode()).hexdigest())
    cached = cache_get(cache_key)
    if cached:
        return cached

    system = (
        "You are a precise legal summarizer for Indian judgments. "
        "Produce a short output containing: TITLE: <title line> \\n SUMMARY: <3-6 lines> \\n AI_SCORE:<0-100> "
        "Be factual and terse."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Query: {query}\n\nJudgment text:\n{text}"}
    ]

    resp = await openai_chat_completion(messages)
    out = resp.choices[0].message.content
    ai_score = 0
    for line in out.splitlines():
        if "AI_SCORE" in line.upper():
            try:
                ai_score = int(re.search(r"(\d{1,3})", line).group(1))
            except:
                ai_score = 0
            break

    cache_set(cache_key, (out, ai_score), ttl=60 * 60)
    return out, ai_score

# ---------- AI: ILAC note
async def generate_ilac_note(query: str, text: str) -> str:
    cache_key = make_key("ilac", hashlib.sha256((query + text[:400]).encode()).hexdigest())
    cached = cache_get(cache_key)
    if cached:
        return cached

    messages = [
        {"role":"system","content":"You are an Indian law assistant. Write ILAC: ISSUE, LAW, APPLICATION, CONCLUSION. Keep concise."},
        {"role":"user","content":f"Query: {query}\n\nText:\n{text}"}
    ]
    resp = await openai_chat_completion(messages)
    out = resp.choices[0].message.content
    cache_set(cache_key, out, ttl=60 * 60)
    return out

# ---------- AI: Arguments
async def generate_arguments(query: str, text: str) -> str:
    cache_key = make_key("args", hashlib.sha256((query + text[:400]).encode()).hexdigest())
    cached = cache_get(cache_key)
    if cached:
        return cached

    messages = [
        {"role":"system","content":"You are an Indian litigation analyst. Produce structured: PETITIONER ARGUMENTS, RESPONDENT ARGUMENTS, COUNTER ARGUMENTS."},
        {"role":"user","content":f"Query: {query}\n\nText:\n{text}"}
    ]
    resp = await openai_chat_completion(messages)
    out = resp.choices[0].message.content
    cache_set(cache_key, out, ttl=60 * 60)
    return out

# ---------- AI: Compare case with query
async def compare_case_with_query(query: str, judgment_text: str) -> str:
    messages = [
        {"role":"system","content":"You are an Indian legal analyst. Compare query vs judgment: list MATCHES, DIFFERENCES, and a 1-line IMPACT conclusion."},
        {"role":"user","content":f"Query: {query}\n\nJudgment:\n{judgment_text}"}
    ]
    resp = await openai_chat_completion(messages)
    return resp.choices[0].message.content

# ---------- AI: Extract legal issues
async def extract_legal_issues(query: str) -> str:
    messages = [
        {"role":"system","content":"You are an Indian lawyer. Extract 3-6 legal issues as questions a court would ask."},
        {"role":"user","content":query}
    ]
    resp = await openai_chat_completion(messages)
    return resp.choices[0].message.content.strip()

# ---------- Citation extraction (local regex)
def extract_citations(text: str) -> str:
    case_pattern = r"[A-Z][A-Za-z .]+ vs\.? [A-Z][A-Za-z .]+"
    statute_pattern = r"(IPC\s*\d+|CrPC\s*\d+|Section\s*\d+|Evidence Act\s*\d+)"
    cases = list(dict.fromkeys(re.findall(case_pattern, text)))
    statutes = list(dict.fromkeys(re.findall(statute_pattern, text, flags=re.IGNORECASE)))

    out = "📌 CITED CASES:\n"
    out += "\n".join(f"- {c}" for c in cases) if cases else "None"
    out += "\n\n📌 STATUTES:\n"
    out += "\n".join(f"- {s}" for s in statutes) if statutes else "None"
    return out

# ---------- Relevance scoring (blend AI score + keyword/statute matching)
def compute_relevance_score(ai_score: int, query: str, text: str) -> int:
    q = query.lower()
    t = text.lower()

    keyword_score = sum(5 for w in q.split() if w in t)
    keyword_score = min(keyword_score, 100)

    statute_boost = 0
    for n in re.findall(r"\b\d+\b", q):
        if n in t:
            statute_boost += 15
    statute_boost = min(statute_boost, 30)

    total = int((0.6 * ai_score) + (0.3 * keyword_score) + statute_boost)
    return min(total, 100)

# ---------- Ask about case (Q&A constrained to judgment text)
async def ask_about_case(question: str, text: str) -> str:
    messages = [
        {"role":"system","content":"Answer using ONLY the judgment text provided. If info not present, say you cannot find it."},
        {"role":"user","content":f"Question: {question}\n\nJudgment text:\n{text}"}
    ]
    resp = await openai_chat_completion(messages)
    return resp.choices[0].message.content

# ---------------- END PART 2 ----------------
# ---------------- PART 3: HANDLERS, PDF BUILDER, BOT START ----------------

# ---------- PDF HTML builder with NyayaAI branding
def build_html_report(query: str, results: List[Dict[str, Any]], ilac: str, arguments: str, citations: str, qna: str) -> str:
    now = datetime.now().strftime("%d %b %Y %I:%M %p")
    header = f"""
    <div style="background:#17223b;color:#fff;padding:18px;border-radius:8px;">
        <div style="font-size:28px;font-weight:700;">NyayaAI</div>
        <div style="font-size:13px;color:#cfd8e3;margin-top:4px;">AI-assisted legal research — India</div>
    </div>
    <p><b>Query:</b> {query} &nbsp; &nbsp; <b>Generated:</b> {now}</p><hr/>
    """
    body = ""
    for idx, r in enumerate(results, start=1):
        body += f"""
        <h3>Case {idx}</h3>
        <p><b>Link:</b> <a href="{r['link']}">{r['link']}</a></p>
        <pre style="white-space:pre-wrap;font-family:monospace">{r['summary']}</pre>
        <p><b>AI Score:</b> {r['ai_score']} — <b>Relevance:</b> {r['relevance']}%</p>
        <hr/>
        """

    body += "<h3>ILAC</h3><pre>{}</pre><hr/>".format(ilac or "None")
    body += "<h3>Arguments</h3><pre>{}</pre><hr/>".format(arguments or "None")
    body += "<h3>Citations</h3><pre>{}</pre><hr/>".format(citations or "None")
    if qna:
        body += "<h3>Q&A</h3><pre>{}</pre><hr/>".format(qna)

    footer = "<p style='font-size:11px;color:gray'>Generated by NyayaAI — for research only. Not legal advice.</p>"
    html = f"<html><body style='font-family:Arial;padding:16px'>{header}{body}{footer}</body></html>"
    return html


# ---------- Handler: /pdf <query>
async def pdf_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    record_usage(user.id if user else None, command="/pdf")

    if not context.args:
        await safe_send(context.bot, update.effective_chat.id, "Usage: /pdf <query> — e.g. `/pdf IPC 420 cheating`", parse_mode="Markdown")
        return

    raw_query = " ".join(context.args)
    logger.info("[/pdf] Query from %s: %s", user.id if user else "unknown", raw_query)

    # correct query (if possible)
    try:
        corrected = await correct_query(raw_query)
        if corrected and corrected.lower() != raw_query.lower():
            await safe_send(context.bot, update.effective_chat.id, f"🔧 Did you mean: `{corrected}` ?", parse_mode="Markdown")
            q = corrected
        else:
            q = raw_query
    except Exception:
        logger.exception("Query correction failed")
        q = raw_query

    await safe_send(context.bot, update.effective_chat.id, "🔎 Searching case law (Google CSE + Kanoon fallback)...")

    # search
    links = google_legal_search(q, max_results=2)
    if not links:
        links = search_cases_kanoon(q, max_results=2)

    if not links:
        better = await suggest_better_query(q)
        return await safe_send(context.bot, update.effective_chat.id, f"No cases found. Try: `{better}`", parse_mode="Markdown")

    await safe_send(context.bot, update.effective_chat.id, "📘 Summarizing judgments...")
    results = []
    first_text = ""
    for link in links:
        text = fetch_case_text(link)
        first_text = first_text or text
        summary, ai_score = await summarize_case(text, q)
        relevance = compute_relevance_score(ai_score, q, text)
        results.append({"link": link, "summary": summary, "ai_score": ai_score, "relevance": relevance})

    await safe_send(context.bot, update.effective_chat.id, "🤖 Generating ILAC, arguments & citations...")
    ilac = await generate_ilac_note(q, first_text)
    arguments = await generate_arguments(q, first_text)
    citations = extract_citations(first_text)

    qna_text = ""
    # if user included " ? " ask Q&A (legacy support)
    if " ? " in raw_query:
        _, question = raw_query.split(" ? ", 1)
        ans = await ask_about_case(question, first_text)
        qna_text = f"Q: {question}\n\n{ans}"

    # create PDF
    await safe_send(context.bot, update.effective_chat.id, "📄 Building PDF...")
    os.makedirs("pdfs", exist_ok=True)
    filename = f"pdfs/{uuid.uuid4().hex}.pdf"
    html = build_html_report(q, results, ilac, arguments, citations, qna_text)

    try:
        pdfkit.from_string(html, filename)
        USAGE["pdfs"] += 1
        await context.bot.send_document(chat_id=update.effective_chat.id, document=open(filename, "rb"))
        logger.info("PDF generated and sent: %s", filename)
    except Exception:
        logger.exception("Failed to create/send PDF")
        await safe_send(context.bot, update.effective_chat.id, "❌ Failed to generate PDF. Try again later.")


# ---------- Handler: /compare <query>
async def compare_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    record_usage(user.id if user else None, command="/compare")

    raw = " ".join(context.args) if context.args else ""
    if not raw:
        return await safe_send(context.bot, update.effective_chat.id, "Usage: /compare <query>")

    try:
        q = await correct_query(raw)
    except Exception:
        q = raw

    await safe_send(context.bot, update.effective_chat.id, "🔍 Finding a matching judgment...")

    links = google_legal_search(q, max_results=1) or search_cases_kanoon(q, max_results=1)
    if not links:
        better = await suggest_better_query(q)
        return await safe_send(context.bot, update.effective_chat.id, f"No cases found. Try `{better}`", parse_mode="Markdown")

    text = fetch_case_text(links[0])
    comparison = await compare_case_with_query(q, text)

    await safe_send(
        context.bot,
        update.effective_chat.id,
        f"📎 *Case:* {links[0]}\n\n📊 *COMPARISON*\n\n{comparison}",
        parse_mode="Markdown"
    )


# ---------- Handler: /issues <query>
async def issues_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    record_usage(user.id if user else None, command="/issues")

    raw = " ".join(context.args) if context.args else ""
    if not raw:
        return await safe_send(context.bot, update.effective_chat.id, "Usage: /issues <query>")

    try:
        q = await correct_query(raw)
    except Exception:
        q = raw

    await safe_send(context.bot, update.effective_chat.id, "🔎 Extracting legal issues...")
    issues = await extract_legal_issues(q)
    await safe_send(context.bot, update.effective_chat.id, f"📌 *LEGAL ISSUES IDENTIFIED*\n\n{issues}", parse_mode="Markdown")


# ---------- Handler: /uploadfir (reply to PDF)
async def uploadfir_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    record_usage(user.id if user else None, command="/uploadfir")

    if not update.message.reply_to_message or not update.message.reply_to_message.document:
        return await safe_send(context.bot, update.effective_chat.id, "➤ Reply to an FIR PDF with /uploadfir")

    doc = update.message.reply_to_message.document
    if not doc.file_name.lower().endswith(".pdf"):
        return await safe_send(context.bot, update.effective_chat.id, "Only PDF files allowed for FIR.")

    os.makedirs("pdfs", exist_ok=True)
    local_path = f"pdfs/{doc.file_id}_{doc.file_name}"

    try:
        tg_file = await doc.get_file()
        await tg_file.download_to_drive(local_path)
    except Exception:
        logger.exception("Failed to download FIR PDF")
        return await safe_send(context.bot, update.effective_chat.id, "Failed to download FIR PDF.")

    fir_text = extract_text_from_pdf(local_path)
    if not fir_text:
        return await safe_send(context.bot, update.effective_chat.id, "Could not extract text from the PDF.")

    await safe_send(context.bot, update.effective_chat.id, "🤖 Creating best search query from FIR...")
    smartq = await convert_fir_to_query(fir_text)
    await safe_send(context.bot, update.effective_chat.id, f"🔍 Auto Query:\n`{smartq}`", parse_mode="Markdown")

    # forward to pdf flow
    context.args = smartq.split()
    await pdf_command(update, context)


# ---------- Handler: feedback (from Part 1's helper)
async def feedback_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    uid = user.id if user else None
    record_usage(uid, command="/feedback")
    text = " ".join(context.args) if context.args else ""
    if not text:
        return await safe_send(context.bot, update.effective_chat.id, "Usage: /feedback <your message>")

    # append to file
    try:
        fb_dir = LOG_DIR / "feedback"
        fb_dir.mkdir(exist_ok=True)
        fname = fb_dir / f"feedback_{int(time.time())}.txt"
        fname.write_text(f"USER:{uid}\nTIME:{datetime.utcnow().isoformat()}\nMSG:{text}\n")
        logger.info("Feedback saved from %s", uid)
    except Exception:
        logger.exception("Saving feedback failed")
    await safe_send(context.bot, update.effective_chat.id, "Thanks — feedback recorded.")


# ---------- Default text handler (user typed plain query)
async def reply_to_user(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    uid = user.id if user else None
    record_usage(uid)
    raw = update.message.text.strip() if update.message.text else ""
    if not raw:
        return

    logger.info("Plain query from %s: %s", uid, raw)
    try:
        q = await correct_query(raw)
    except Exception:
        q = raw

    await safe_send(context.bot, update.effective_chat.id, "🔎 Searching for relevant cases...")

    links = google_legal_search(q, max_results=2) or search_cases_kanoon(q, max_results=2)
    if not links:
        better = await suggest_better_query(q)
        return await safe_send(context.bot, update.effective_chat.id, f"No cases found. Try `{better}`", parse_mode="Markdown")

    final_msg = ""
    for link in links:
        t = fetch_case_text(link)
        summary, ai_score = await summarize_case(t, q)
        rel = compute_relevance_score(ai_score, q, t)
        final_msg += f"{link}\n{summary}\nRelevance: {rel}%\n\n"

    # Telegram messages can be long; send safely (split if needed)
    MAX_CHUNK = 3500
    if len(final_msg) <= MAX_CHUNK:
        await safe_send(context.bot, update.effective_chat.id, final_msg)
    else:
        # split by paragraphs
        parts = [final_msg[i:i+MAX_CHUNK] for i in range(0, len(final_msg), MAX_CHUNK)]
        for p in parts:
            await safe_send(context.bot, update.effective_chat.id, p)

# ---------- Error handler (catch unhandled exceptions)
async def global_error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.exception("Unhandled exception: %s", context.error)
    try:
        if isinstance(update, Update) and update.effective_chat:
            await safe_send(context.bot, update.effective_chat.id, "⚠️ Sorry — something went wrong. The error has been logged.")
    except Exception:
        logger.exception("Failed to send error message to user")

# ---------- MAIN: Register handlers and run bot
def main():
    if not TELEGRAM_TOKEN:
        logger.error("TELEGRAM_TOKEN not set. Exiting.")
        return

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    # commands
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CallbackQueryHandler(faq_handler))
    app.add_handler(CommandHandler("pdf", pdf_command))
    app.add_handler(CommandHandler("compare", compare_command))
    app.add_handler(CommandHandler("issues", issues_command))
    app.add_handler(CommandHandler("uploadfir", uploadfir_command))
    app.add_handler(CommandHandler("feedback", feedback_command))
    # default
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, reply_to_user))

    # global error handler
    app.add_error_handler(global_error_handler)

    logger.info("Bot running (main loop)...")
    app.run_polling()

if __name__ == "__main__":
    main()

# ---------------- END PART 3 ----------------
# ---------------- PART 4: FINAL POLISH, EXTRA COMMANDS, BETTER PROMPTS ----------------

# -------------------- IMPROVED AI PROMPTS --------------------

# Break long judgments into chunks (OpenAI performs better)
def chunk_text(text: str, max_len: int = 6000) -> List[str]:
    words = text.split()
    chunks = []
    curr = []
    total = 0
    for w in words:
        curr.append(w)
        total += len(w) + 1
        if total >= max_len:
            chunks.append(" ".join(curr))
            curr = []
            total = 0
    if curr:
        chunks.append(" ".join(curr))
    return chunks


async def summarize_large_judgment(text: str, query: str) -> Tuple[str, int]:
    chunks = chunk_text(text, max_len=5000)
    combined_summary = ""

    for idx, ch in enumerate(chunks, start=1):
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a senior Indian law researcher. "
                        "Summarize this chunk of a judgment clearly and concisely. "
                        "DO NOT add facts not present in the text."
                    )
                },
                {"role": "user", "content": f"QUERY: {query}\nCHUNK {idx}:\n{ch}"}
            ]
        )
        combined_summary += f"\n[CHUNK {idx}]\n" + resp.choices[0].message.content

    # Now compress combined summary
    final_resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a senior Indian legal editor. Combine all summaries into a single structured format:\n"
                    "TITLE\nSUMMARY\nKEY POINTS\nAI_SCORE (0-100)\n"
                )
            },
            {"role": "user", "content": combined_summary}
        ]
    )

    output = final_resp.choices[0].message.content
    score = 0
    for line in output.split("\n"):
        if "AI_SCORE" in line:
            try: score = int(line.split(":")[1])
            except: pass

    return output, score


# -------------------- Automatic rate limit protection --------------------

LAST_CALL = 0

def wait_if_needed(min_gap=0.7):
    """Space out OpenAI calls."""
    global LAST_CALL
    now = time.time()
    if now - LAST_CALL < min_gap:
        time.sleep(min_gap - (now - LAST_CALL))
    LAST_CALL = time.time()


# Patch all AI helpers to use delay
_original_summarize = summarize_case

async def summarize_case(text, query):
    wait_if_needed(0.8)
    return await _original_summarize(text, query)



# -------------------- Enhanced HELP menu / ABOUT / PRICING --------------------

async def about_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "ℹ️ *About NyayaAI*\n\n"
        "NyayaAI uses advanced AI search + real High Court & Supreme Court data.\n"
        "It provides:\n"
        "• Case-law search\n"
        "• Summaries\n"
        "• ILAC analysis\n"
        "• Arguments\n"
        "• Citations\n"
        "• Full PDF research notes\n\n"
        "Not legal advice—use for research only."
    )
    await safe_send(context.bot, update.effective_chat.id, msg, parse_mode="Markdown")


async def pricing_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "💰 *NyayaAI Pricing*\n\n"
        "🔹 Free Tier\n"
        "• 15 searches/day\n"
        "• Limited ILAC\n\n"
        "🔹 Pro Tier (₹299/month)\n"
        "• Unlimited searches\n"
        "• Full ILAC + Arguments\n"
        "• Priority response speed\n"
        "• Faster PDF generation\n\n"
        "Payment link coming soon."
    )
    await safe_send(context.bot, update.effective_chat.id, msg, parse_mode="Markdown")


async def contact_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "📞 *Contact & Support*\n\n"
        "Email: support@nyayaai.com\n"
        "Telegram: @nyayaai_support\n\n"
        "We respond within 24 hours."
    )
    await safe_send(context.bot, update.effective_chat.id, msg, parse_mode="Markdown")



# -------------------- Improved HELP menu button layout --------------------

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("📘 FAQ", callback_data="faq")],
        [InlineKeyboardButton("💰 Pricing", callback_data="pricing")],
        [InlineKeyboardButton("ℹ️ About NyayaAI", callback_data="about")],
        [InlineKeyboardButton("📞 Contact", callback_data="contact")],
    ]
    markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text(
        "📚 *NyayaAI Help Menu*\nSelect an option:",
        reply_markup=markup,
        parse_mode="Markdown"
    )


async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query.data

    if q == "faq":
        await update.callback_query.answer()
        await update.callback_query.edit_message_text(
            "❓ *FAQ*\n\n"
            "Q: Where does the data come from?\n"
            "A: Google CSE + Supreme Court + High Courts + IndianKanoon.\n\n"
            "Q: Is this legal advice?\n"
            "A: No — research only.\n\n"
            "Q: Can it read FIRs?\n"
            "A: Yes — reply to a PDF and type /uploadfir.",
            parse_mode="Markdown"
        )

    elif q == "pricing":
        await pricing_command(update, context)

    elif q == "about":
        await about_command(update, context)

    elif q == "contact":
        await contact_command(update, context)



# -------------------- Stronger fallback search --------------------

def deep_search(query):
    """First Google CSE → then Kanoon → then Supreme Court archives."""
    links = google_legal_search(query, max_results=2)
    if links:
        return links

    links = search_cases_kanoon(query, max_results=2)
    if links:
        return links

    # basic SCI.gov.in fallback
    if re.search(r"\d{4}", query):
        try:
            year = re.findall(r"\d{4}", query)[0]
            url = f"https://main.sci.gov.in/supremecourt/{year}/"
            html = safe_get(url)
            soup = BeautifulSoup(html, "html.parser")
            found = []
            for a in soup.find_all("a", href=True):
                if "pdf" in a["href"].lower():
                    found.append("https://main.sci.gov.in" + a["href"])
                    if len(found) >= 1:
                        break
            return found
        except:
            return []

    return []


# Patch default handler to use deeper search
_original_reply_handler = reply_to_user

async def reply_to_user(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # use fallback search
    user = update.effective_user
    raw = update.message.text
    try:
        q = await correct_query(raw)
    except:
        q = raw

    await safe_send(context.bot, update.effective_chat.id, "🔍 Searching for judgments...")

    links = deep_search(q)
    if not links:
        better = await suggest_better_query(q)
        return await safe_send(context.bot, update.effective_chat.id, f"No cases found. Try: `{better}`", parse_mode="Markdown")

    final = ""
    for link in links:
        txt = fetch_case_text(link)
        summary, score = await summarize_large_judgment(txt, q)
        rel = compute_relevance_score(score, q, txt)
        final += f"{link}\n{summary}\nRelevance: {rel}%\n\n"

    # Auto-split messages
    MAX = 3500
    for i in range(0, len(final), MAX):
        await safe_send(context.bot, update.effective_chat.id, final[i:i+MAX])


# -------------------- Attach new handlers to main() --------------------

def patch_main():
    """Extend the main() from Part 3 without modifying it."""

    # Add new commands
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("about", about_command))
    app.add_handler(CommandHandler("pricing", pricing_command))
    app.add_handler(CommandHandler("contact", contact_command))

    # Handle callback buttons
    app.add_handler(CallbackQueryHandler(callback_handler))

    logger.info("Part 4 handlers patched successfully.")

# NOTE: Do NOT call patch_main() here.
# It will be called automatically by the deployment step if desired.

# -------------------- END PART 4 --------------------

# ---------------- PART 1 ----------------
# NyayaAI — Part 1: imports, config, logging, safe telegram helpers, usage snapshot
# ----------------

import os
import re
import json
import time
import uuid
import random
import hashlib
import pathlib
import threading
import logging
import asyncio
from typing import List, Dict, Any, Tuple

# optional dotenv
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# core libs
import requests
from bs4 import BeautifulSoup
import fitz  # PyMuPDF (pdf text extraction)
import pdfkit

# Telegram
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)

# OpenAI (new official package)
from openai import OpenAI

# ========== ENV ==========
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

# Webshare proxy credentials (optional)
WEBSHARE_USERNAME = os.getenv("WEBSHARE_USERNAME")
WEBSHARE_PASSWORD = os.getenv("WEBSHARE_PASSWORD")
ENABLE_PROXY = os.getenv("ENABLE_PROXY", "false").lower() in ("1", "true", "yes")

# ========== OpenAI client ==========
if not OPENAI_API_KEY:
    print("WARNING: OPENAI_API_KEY not set — OpenAI calls will fail.")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# ========== Logging ==========
LOG_DIR = pathlib.Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "nyayaai.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
logger = logging.getLogger("NyayaAI")
logger.info("NyayaAI starting...")

# ========== Usage & Snapshot ==========
USAGE: Dict[str, Any] = {"messages": 0, "unique_users": set(), "pdfs": 0, "commands": {}}
USAGE_LOCK = threading.Lock()
USAGE_FILE = LOG_DIR / "usage.json"

def record_usage(user_id=None, command=None):
    with USAGE_LOCK:
        USAGE["messages"] += 1
        if user_id:
            USAGE["unique_users"].add(user_id)
        if command:
            USAGE["commands"].setdefault(command, 0)
            USAGE["commands"][command] += 1

def snapshot_usage_daemon():
    while True:
        time.sleep(60)
        try:
            with USAGE_LOCK:
                snapshot = {
                    "messages": USAGE["messages"],
                    "unique_users": len(USAGE["unique_users"]),
                    "pdfs": USAGE["pdfs"],
                    "commands": USAGE["commands"],
                }
            USAGE_FILE.write_text(json.dumps(snapshot, indent=2))
        except Exception:
            logger.exception("Failed to write usage snapshot")

threading.Thread(target=snapshot_usage_daemon, daemon=True).start()

# ========== Safe Telegram helpers ==========
async def safe_send(bot, chat_id, text, parse_mode=None, reply_markup=None):
    """
    Send message with basic retry and avoid Telegram parse errors by defaulting to plain text
    (We don't pass parse_mode for AI outputs; use Markdown only for small pre-escaped strings).
    """
    for attempt in range(3):
        try:
            return await bot.send_message(chat_id=chat_id, text=text, parse_mode=parse_mode, reply_markup=reply_markup)
        except Exception as e:
            # common issues: RetryAfter, TimedOut, BadRequest etc.
            logger.warning("safe_send attempt %s failed: %s", attempt + 1, e)
            await asyncio.sleep(1 + attempt * 1.5)
    logger.error("safe_send ultimately failed for chat %s", chat_id)
    return None

async def safe_edit(bot, chat_id, message_id, text, parse_mode=None, reply_markup=None):
    try:
        return await bot.edit_message_text(chat_id=chat_id, message_id=message_id, text=text, parse_mode=parse_mode, reply_markup=reply_markup)
    except Exception as e:
        logger.warning("safe_edit failed: %s", e)
        return None

# small helper to chunk long texts for Telegram
def chunk_text(s: str, chunk_size: int = 3500) -> List[str]:
    return [s[i:i+chunk_size] for i in range(0, len(s), chunk_size)]
# ---------------- PART 2 ----------------
# NyayaAI — Part 2: search stack, proxy, caching, scrapers, fetch_case_text
# ----------------

# ========== Caching ==========
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
        CACHE[key] = {"value": value, "expires_at": time.time() + ttl}

def make_key(*parts) -> str:
    raw = "|".join(str(p) for p in parts)
    return hashlib.sha256(raw.encode()).hexdigest()

# ========== Webshare rotating proxy support ==========
def get_rotating_proxy_url() -> str:
    """
    Returns a Webshare rotating-proxy URL (http) if enabled & credentials present.
    Format used: http://<username>-session-<id>:<password>@p.webshare.io:80
    """
    if not ENABLE_PROXY:
        return ""
    if not WEBSHARE_USERNAME or not WEBSHARE_PASSWORD:
        logger.warning("ENABLE_PROXY true but WEBSHARE credentials missing.")
        return ""
    session_id = random.randint(100000, 999999)
    proxy_user = f"{WEBSHARE_USERNAME}-session-{session_id}"
    return f"http://{proxy_user}:{WEBSHARE_PASSWORD}@p.webshare.io:80"

def safe_request(url: str, timeout: int = 20, allow_proxy_for: List[str] = None, **kwargs):
    """
    Wrapper that attempts proxy request first (if enabled), falls back to direct.
    allow_proxy_for: list of substrings; proxy used only when any substring in url.
    """
    headers = kwargs.pop("headers", {"User-Agent": "Mozilla/5.0 (NyayaAI)"})
    proxies = None
    try:
        proxy_url = get_rotating_proxy_url()
        allow = True
        if allow_proxy_for:
            allow = any(pat in url.lower() for pat in allow_proxy_for)
        if proxy_url and allow:
            proxies = {"http": proxy_url, "https": proxy_url}
            resp = requests.get(url, headers=headers, proxies=proxies, timeout=timeout)
            resp.raise_for_status()
            return resp
    except Exception as e:
        logger.warning("Proxy failed for %s. Retrying without proxy. (%s)", url, e)
    # fallback direct
    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        return resp
    except Exception as e:
        logger.exception("Direct request failed for %s: %s", url, e)
        raise

# ========== Google Custom Search (optional) ==========
def google_legal_search(query: str, max_results: int = 5) -> List[str]:
    if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
        logger.debug("google_legal_search: Google keys not present")
        return []
    cache_key = make_key("google", query, max_results)
    cached = cache_get(cache_key)
    if cached:
        return cached
    url = "https://www.googleapis.com/customsearch/v1"
    params = {"key": GOOGLE_API_KEY, "cx": GOOGLE_CSE_ID, "q": f"{query} judgment case law", "num": min(max_results, 10)}
    try:
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        items = data.get("items", [])
        links = []
        for it in items:
            link = it.get("link")
            if not link:
                continue
            # filter to legal sites only (simple heuristic)
            if any(dom in link for dom in ["indiankanoon.org", "sci.gov.in", "judis.nic.in", ".nic.in", "highcourt", "judgement", "judgment", ".gov.in"]):
                links.append(link)
            elif link.lower().endswith(".pdf"):
                links.append(link)
            if len(links) >= max_results:
                break
        cache_set(cache_key, links, ttl=60 * 30)
        logger.info("google_legal_search found %d links for '%s'", len(links), query)
        return links
    except Exception:
        logger.exception("google_legal_search error")
        return []

# ========== IndianKanoon scraping fallback ==========
def search_cases_kanoon(query: str, max_results: int = 3) -> List[str]:
    cache_key = make_key("kanoon", query, max_results)
    cached = cache_get(cache_key)
    if cached:
        return cached
    try:
        q = query.replace(" ", "+")
        url = f"https://indiankanoon.org/search/?formInput={q}"
        # use safe_request and limit proxy to indiankanoon if enabled
        resp = safe_request(url, timeout=20, allow_proxy_for=["indiankanoon.org"])
        html = resp.text
        soup = BeautifulSoup(html, "html.parser")
        links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.startswith("/doc/"):
                full = "https://indiankanoon.org" + href
                if full not in links:
                    links.append(full)
            if len(links) >= max_results:
                break
        cache_set(cache_key, links, ttl=60 * 30)
        logger.info("search_cases_kanoon: found %d results for '%s'", len(links), query)
        return links
    except Exception:
        logger.exception("search_cases_kanoon error")
        return []

# ========== Basic Supreme Court fallback (search index for year) ==========
def sci_fallback_search(query: str, max_results: int = 2) -> List[str]:
    # quick heuristic: if query contains year or 'supreme'
    try:
        # try simple site-limited Google as last resort if keys exist
        if GOOGLE_API_KEY and GOOGLE_CSE_ID:
            url = "https://www.googleapis.com/customsearch/v1"
            params = {"key": GOOGLE_API_KEY, "cx": GOOGLE_CSE_ID, "q": f"{query} site:main.sci.gov.in filetype:pdf", "num": max_results}
            r = requests.get(url, params=params, timeout=8).json()
            links = [it["link"] for it in r.get("items", []) if it.get("link")]
            return links
    except Exception:
        logger.debug("sci_fallback_search failed")
    return []

# ========== fetch_case_text (HTML pages + PDF) ==========
def extract_text_from_pdf(path: str) -> str:
    try:
        doc = fitz.open(path)
        text = []
        for page in doc:
            text.append(page.get_text())
        return "\n".join(text).strip()
    except Exception:
        logger.exception("extract_text_from_pdf failed for %s", path)
        return ""

def fetch_case_text(url: str) -> str:
    cache_key = make_key("case_text", url)
    cached = cache_get(cache_key)
    if cached:
        return cached
    try:
        logger.info("Fetching case text: %s", url)
        if url.lower().endswith(".pdf"):
            resp = safe_request(url, timeout=30, allow_proxy_for=["indiankanoon.org", "sci.gov.in"])
            tmp = f"pdfs/{uuid.uuid4().hex}.pdf"
            os.makedirs("pdfs", exist_ok=True)
            with open(tmp, "wb") as fh:
                fh.write(resp.content)
            text = extract_text_from_pdf(tmp)
            try:
                os.remove(tmp)
            except Exception:
                pass
        else:
            resp = safe_request(url, timeout=20, allow_proxy_for=["indiankanoon.org"])
            html = resp.text
            soup = BeautifulSoup(html, "html.parser")
            # try specific selectors else fallback to paragraphs
            selectors = soup.select(".judgment, .judgement, #content, .casebody, .doc, .judgement-body")
            nodes = []
            if selectors:
                for s in selectors:
                    nodes.append(s.get_text(separator="\n", strip=True))
            else:
                paras = soup.find_all("p")
                for p in paras:
                    nodes.append(p.get_text(strip=True))
            text = "\n".join(nodes)
        text = text.strip()[:15000]  # limit length
        cache_set(cache_key, text, ttl=60 * 60)
        return text
    except Exception:
        logger.exception("fetch_case_text error for %s", url)
        return ""
# ---------------- PART 3 ----------------
# NyayaAI — Part 3: OpenAI wrappers, summarizers, ILAC, arguments, compare, scoring, PDF
# ----------------

# ========== OpenAI wrapper with retries ==========
async def openai_chat(messages: List[dict], model: str = "gpt-4o-mini", max_retries: int = 2):
    if not client:
        raise RuntimeError("OpenAI client not configured (OPENAI_API_KEY missing).")
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            # synchronous call (OpenAI SDK returns) — keep small rate spacing
            resp = client.chat.completions.create(model=model, messages=messages)
            return resp
        except Exception as e:
            last_exc = e
            logger.warning("OpenAI call failed (attempt %s): %s", attempt + 1, e)
            await asyncio.sleep(1 + attempt * 1.5)
    logger.exception("OpenAI failed after retries")
    raise last_exc

# spacing control
_last_openai = 0.0
def wait_openai(min_gap=0.7):
    global _last_openai
    now = time.time()
    if now - _last_openai < min_gap:
        time.sleep(min_gap - (now - _last_openai))
    _last_openai = time.time()

# ========== Summarizer ==========
async def summarize_case(text: str, query: str) -> Tuple[str, int]:
    if not text:
        return ("No text available to summarize.", 0)
    cache_key = make_key("summary", query, hashlib.sha256(text[:500].encode()).hexdigest())
    cached = cache_get(cache_key)
    if cached:
        return cached
    wait_openai(0.8)
    system = ("You are a precise Indian legal summarizer. Produce output with lines: "
              "TITLE: <one-line title>\nSUMMARY: <3-6 lines>\nKEY_FACTS:\n- ...\nAI_SCORE: <0-100>")
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Query: {query}\n\nJudgment text:\n{text[:10000]}"}
    ]
    resp = await openai_chat(messages)
    out = resp.choices[0].message.content
    score = 0
    for line in out.splitlines():
        if "AI_SCORE" in line.upper():
            m = re.search(r"(\d{1,3})", line)
            if m:
                try:
                    score = int(m.group(1))
                except:
                    score = 0
            break
    cache_set(cache_key, (out, score), ttl=60 * 60)
    return out, score

# ========== ILAC ==========
async def generate_ilac(query: str, text: str) -> str:
    if not text:
        return "No text to generate ILAC."
    cache_key = make_key("ilac", query, hashlib.sha256(text[:400].encode()).hexdigest())
    cached = cache_get(cache_key)
    if cached:
        return cached
    wait_openai(0.7)
    messages = [
        {"role":"system","content":"You are an Indian law assistant. Write ILAC: ISSUE, LAW, APPLICATION, CONCLUSION. Keep concise."},
        {"role":"user","content":f"Query: {query}\n\nJudgment text:\n{text[:8000]}"}
    ]
    resp = await openai_chat(messages)
    out = resp.choices[0].message.content
    cache_set(cache_key, out, ttl=60 * 60)
    return out

# ========== Arguments ==========
async def generate_arguments(query: str, text: str) -> str:
    if not text:
        return "No text to generate arguments."
    cache_key = make_key("args", query, hashlib.sha256(text[:400].encode()).hexdigest())
    cached = cache_get(cache_key)
    if cached:
        return cached
    wait_openai(0.7)
    messages = [
        {"role":"system","content":"You are an Indian litigation analyst. Produce PETITIONER ARGUMENTS, RESPONDENT ARGUMENTS, COUNTER ARGUMENTS (concise bullet points)."},
        {"role":"user","content":f"Query: {query}\n\nJudgment text:\n{text[:8000]}"}
    ]
    resp = await openai_chat(messages)
    out = resp.choices[0].message.content
    cache_set(cache_key, out, ttl=60 * 60)
    return out

# ========== Compare and issues ==========
async def compare_case_with_query(query: str, judgment_text: str) -> str:
    if not judgment_text:
        return "No judgment text to compare."
    wait_openai(0.6)
    messages = [
        {"role":"system","content":"You are an Indian legal analyst. Compare QUERY vs JUDGMENT. Output structured: MATCHES:, DIFFERENCES:, IMPACT (1-line)."},
        {"role":"user","content":f"Query: {query}\n\nJudgment:\n{judgment_text[:9000]}"}
    ]
    resp = await openai_chat(messages)
    return resp.choices[0].message.content

async def extract_legal_issues(query: str) -> str:
    wait_openai(0.6)
    messages = [
        {"role":"system","content":"You are an Indian lawyer. Extract 3-6 legal issues from the query as court-style questions."},
        {"role":"user","content":query}
    ]
    resp = await openai_chat(messages)
    return resp.choices[0].message.content

# ========== Q&A constrained to judgment ==========
async def ask_about_case(question: str, text: str) -> str:
    if not text:
        return "No judgment text provided."
    wait_openai(0.6)
    messages = [
        {"role":"system","content":"Answer using ONLY the judgment text. If not present, say you cannot find it."},
        {"role":"user","content":f"Question: {question}\n\nJudgment:\n{text[:10000]}"}
    ]
    resp = await openai_chat(messages)
    return resp.choices[0].message.content

# ========== Citation extraction ==========
def extract_citations(text: str) -> str:
    case_pattern = r"[A-Z][A-Za-z .]+ vs\.? [A-Z][A-Za-z .]+"
    statute_pattern = r"(IPC\s*\d+|CrPC\s*\d+|Section\s*\d+|Evidence Act\s*\d+)"
    cases = list(dict.fromkeys(re.findall(case_pattern, text)))
    statutes = list(dict.fromkeys(re.findall(statute_pattern, text, flags=re.IGNORECASE)))
    out = "📌 CITED CASES:\n" + ("\n".join(f"- {c}" for c in cases) if cases else "None")
    out += "\n\n📌 STATUTES:\n" + ("\n".join(f"- {s}" for s in statutes) if statutes else "None")
    return out

# ========== Relevance scoring ==========
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

# ========== PDF builder ==========
def build_html_report(query: str, results: List[Dict[str, Any]], ilac: str, arguments: str, citations: str, qna: str = "") -> str:
    now = time.strftime("%d %b %Y %I:%M %p")
    header = f'<div style="background:#0f3a63;color:#fff;padding:15px;border-radius:8px;"><h1>NyayaAI</h1><div>AI Legal Research • {now}</div></div>'
    header += f"<h3>Query: {query}</h3><hr>"
    body = ""
    for i, r in enumerate(results, 1):
        body += f"<h4>Result {i} — Relevance: {r.get('relevance','N/A')}%</h4>"
        body += f"<p><a href='{r['link']}'>{r['link']}</a></p>"
        body += f"<pre style='background:#f8f8f8;padding:12px;border-left:4px solid #0f3a63'>{r['summary']}</pre><hr/>"
    body += "<h3>ILAC</h3><pre>" + (ilac or "Not generated") + "</pre>"
    body += "<h3>Arguments</h3><pre>" + (arguments or "Not generated") + "</pre>"
    body += "<h3>Citations</h3><pre>" + (citations or "None") + "</pre>"
    if qna:
        body += "<h3>Q&A</h3><pre>" + qna + "</pre>"
    footer = "<p style='font-size:11px;color:gray'>Generated by NyayaAI — For research only. Not legal advice.</p>"
    return f"<html><body style='font-family:Arial;margin:18px'>{header}{body}{footer}</body></html>"
# ---------------- PART 4 ----------------
# NyayaAI — Part 4: Handlers, Help/FAQ, main(), error handler
# ----------------

# ========== HELP / FAQ menu (safe small markdown strings) ==========
HELP_TEXT = (
    "👋 *Welcome to NyayaAI*\n\n"
    "NyayaAI helps with Indian case-law research:\n"
    "• Search Supreme & High Court case law\n"
    "• Auto-correct legal queries (IPC/CrPC)\n"
    "• Summaries + AI relevance score\n"
    "• ILAC, Arguments, Citations\n"
    "• Compare your query with judgments (/compare)\n"
    "• Extract legal issues (/issues)\n"
    "• Create PDF research reports (/pdf)\n"
    "• Convert FIR PDF → query → report (/uploadfir)\n\n"
    "Tap a button below for FAQ, Pricing or Contact."
)

FAQ_KB = InlineKeyboardMarkup([
    [InlineKeyboardButton("📘 FAQ", callback_data="faq")],
    [InlineKeyboardButton("💰 Pricing", callback_data="pricing"),
     InlineKeyboardButton("✉️ Feedback", callback_data="feedback")],
    [InlineKeyboardButton("ℹ️ About", callback_data="about"),
     InlineKeyboardButton("📞 Contact", callback_data="contact")],
])

FAQ_TEXTS = {
    "faq": ("❓ *FAQ*\n\n"
            "Q: Where does data come from?\nA: Google CSE + Supreme Court + High Courts + IndianKanoon.\n\n"
            "Q: Is this legal advice?\nA: No — research only.\n\n"
            "Q: How to use FIR?\nA: Reply to a FIR PDF with /uploadfir."),
    "pricing": ("💰 *Pricing*\n\nFree tier available. Pro tier will unlock more features and higher rate limits."),
    "about": ("ℹ️ *About NyayaAI*\n\nNyayaAI is an AI-assisted legal research tool for India. Generates summaries, ILAC, citations and PDF reports."),
    "contact": ("📞 *Contact*\n\nEmail: support@nyayaai.com\nTelegram: @nyayaai_support"),
    "feedback": ("✉️ *Feedback*\n\nUse /feedback <your message> to record feedback.")
}

# ========== Commands & Handlers ==========
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    record_usage(update.effective_user.id if update.effective_user else None, "/start")
    await safe_send(context.bot, update.effective_chat.id, HELP_TEXT, parse_mode="Markdown", reply_markup=FAQ_KB)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    record_usage(update.effective_user.id if update.effective_user else None, "/help")
    await safe_send(context.bot, update.effective_chat.id, HELP_TEXT, parse_mode="Markdown", reply_markup=FAQ_KB)

async def faq_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    key = query.data
    if key == "faq":
        text = FAQ_TEXTS.get("faq")
    elif key == "pricing":
        text = FAQ_TEXTS.get("pricing")
    elif key == "about":
        text = FAQ_TEXTS.get("about")
    elif key == "contact":
        text = FAQ_TEXTS.get("contact")
    elif key == "feedback":
        text = FAQ_TEXTS.get("feedback")
    else:
        text = "Unknown option."
    # edit message; use plain text (no heavy markdown fields from AI)
    try:
        await query.edit_message_text(text, parse_mode="Markdown")
    except Exception as e:
        logger.warning("FAQ edit failed: %s", e)
        try:
            await safe_send(context.bot, query.message.chat_id, text, parse_mode="Markdown")
        except:
            pass

# ========== /pdf handler ==========
async def pdf_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id if update.effective_user else None
    record_usage(user_id, "/pdf")
    if not context.args:
        return await safe_send(context.bot, update.effective_chat.id, "Usage: /pdf <query>\nExample: /pdf IPC 420 cheating")
    raw = " ".join(context.args)
    # correct query step (light)
    try:
        wait_openai(0.6)
        corrected_msg = None
        if client:
            resp = client.chat.completions.create(model="gpt-4o-mini", messages=[
                {"role":"system","content":"Correct and normalize this Indian legal search query (5-8 words). Output only the corrected query."},
                {"role":"user","content": raw}
            ])
            corrected = resp.choices[0].message.content.strip()
            if corrected and corrected.lower() != raw.lower():
                corrected_msg = corrected
    except Exception:
        corrected = raw
    q = corrected_msg or raw

    msg = await safe_send(context.bot, update.effective_chat.id, "🔎 Searching case law (GoogleCSE + Kanoon fallback)...")
    # search
    links = google_legal_search(q, max_results=3) or search_cases_kanoon(q, max_results=3) or sci_fallback_search(q, max_results=2)
    if not links:
        better = None
        try:
            if client:
                wait_openai(0.6)
                resp = client.chat.completions.create(model="gpt-4o-mini", messages=[
                    {"role":"system","content":"Rewrite this weak search query into a stronger Indian Kanoon search query (5-8 words). Output only the improved query."},
                    {"role":"user","content": q}
                ])
                better = resp.choices[0].message.content.strip()
        except Exception:
            better = None
        if better:
            return await safe_edit(context.bot, update.effective_chat.id, msg.message_id, f"No cases found. Try: `{better}`")
        return await safe_edit(context.bot, update.effective_chat.id, "No cases found. Try a broader query.")

    await safe_edit(context.bot, update.effective_chat.id, msg.message_id, f"Found {len(links)} results. Summarizing...")
    # summarize & build report
    results = []
    first_text = ""
    for link in links:
        text = fetch_case_text(link)
        first_text = first_text or text
        summary, ai_score = await summarize_case(text, q)
        rel = compute_relevance_score(ai_score, q, text)
        results.append({"link": link, "summary": summary, "ai_score": ai_score, "relevance": rel})

    ilac = await generate_ilac(q, first_text)
    arguments = await generate_arguments(q, first_text)
    citations = extract_citations(first_text)
    qna = ""

    await safe_edit(context.bot, update.effective_chat.id, msg.message_id, "Building PDF report...")
    os.makedirs("pdfs", exist_ok=True)
    filename = f"pdfs/nyaya_{uuid.uuid4().hex}.pdf"
    html = build_html_report(q, results, ilac, arguments, citations, qna)
    try:
        pdfkit.from_string(html, filename)
        USAGE["pdfs"] += 1
        await context.bot.send_document(chat_id=update.effective_chat.id, document=open(filename, "rb"))
        try:
            os.remove(filename)
        except:
            pass
    except Exception:
        logger.exception("PDF generation failed")
        await safe_send(context.bot, update.effective_chat.id, "Failed to generate PDF. Make sure wkhtmltopdf is installed.")

# ========== /compare handler ==========
async def compare_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    record_usage(update.effective_user.id if update.effective_user else None, "/compare")
    raw = " ".join(context.args) if context.args else ""
    if not raw:
        return await safe_send(context.bot, update.effective_chat.id, "Usage: /compare <query>")
    q = raw
    links = google_legal_search(q, max_results=1) or search_cases_kanoon(q, max_results=1)
    if not links:
        return await safe_send(context.bot, update.effective_chat.id, "No matching judgment found.")
    text = fetch_case_text(links[0])
    comparison = await compare_case_with_query(q, text)
    # send in safe chunks
    for chunk in chunk_text(f"Case: {links[0]}\n\nComparison:\n{comparison}"):
        await safe_send(context.bot, update.effective_chat.id, chunk)

# ========== /issues handler ==========
async def issues_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    record_usage(update.effective_user.id if update.effective_user else None, "/issues")
    raw = " ".join(context.args) if context.args else ""
    if not raw:
        return await safe_send(context.bot, update.effective_chat.id, "Usage: /issues <query>")
    issues = await extract_legal_issues(raw)
    for chunk in chunk_text(f"Legal issues:\n{issues}"):
        await safe_send(context.bot, update.effective_chat.id, chunk)

# ========== /uploadfir handler ==========
async def uploadfir_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    record_usage(update.effective_user.id if update.effective_user else None, "/uploadfir")
    if not update.message.reply_to_message or not update.message.reply_to_message.document:
        return await safe_send(context.bot, update.effective_chat.id, "Reply to a FIR PDF with /uploadfir")
    doc = update.message.reply_to_message.document
    if not doc.file_name.lower().endswith(".pdf"):
        return await safe_send(context.bot, update.effective_chat.id, "Only PDF is supported.")
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
        return await safe_send(context.bot, update.effective_chat.id, "Could not extract text from FIR.")
    # convert FIR to query
    try:
        wait_openai(0.6)
        resp = client.chat.completions.create(model="gpt-4o-mini", messages=[
            {"role":"system","content":"Read FIR text and return a short (5-8 words) Indian Kanoon search query. Do NOT include names or locations; include IPC sections if present. Output only the query."},
            {"role":"user","content": fir_text[:8000]}
        ])
        smart_query = resp.choices[0].message.content.strip()
    except Exception:
        smart_query = " ".join(fir_text.split()[:7])
    await safe_send(context.bot, update.effective_chat.id, f"Auto query generated:\n`{smart_query}`")
    context.args = smart_query.split()
    await pdf_command(update, context)

# ========== Feedback command ==========
async def feedback_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    record_usage(user.id if user else None, "/feedback")
    text = " ".join(context.args) if context.args else ""
    if not text:
        return await safe_send(context.bot, update.effective_chat.id, "Usage: /feedback <your message>")
    fb_dir = LOG_DIR / "feedback"
    fb_dir.mkdir(exist_ok=True)
    fname = fb_dir / f"fb_{int(time.time())}.txt"
    try:
        fname.write_text(json.dumps({"user": user.id if user else None, "time": time.time(), "msg": text}))
        await safe_send(context.bot, update.effective_chat.id, "Thanks — feedback recorded.")
    except Exception:
        logger.exception("Feedback save failed")
        await safe_send(context.bot, update.effective_chat.id, "Failed to record feedback.")

# ========== Default message handler ==========
async def reply_to_user(update: Update, context: ContextTypes.DEFAULT_TYPE):
    record_usage(update.effective_user.id if update.effective_user else None)
    raw = update.message.text or ""
    raw = raw.strip()
    if not raw:
        return
    await safe_send(context.bot, update.effective_chat.id, "🔎 Searching for judgments...")
    q = raw
    links = google_legal_search(q, max_results=2) or search_cases_kanoon(q, max_results=2) or sci_fallback_search(q, max_results=2)
    if not links:
        return await safe_send(context.bot, update.effective_chat.id, "No cases found. Try simpler keywords.")
    final = ""
    for link in links:
        txt = fetch_case_text(link)
        summary, score = await summarize_case(txt, q)
        rel = compute_relevance_score(score, q, txt)
        final += f"{link}\n\n{summary}\n\nRelevance: {rel}%\n\n{'-'*30}\n\n"
    for part in chunk_text(final):
        await safe_send(context.bot, update.effective_chat.id, part)

# ========== Global error handler ==========
async def global_error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.exception("Unhandled error: %s", context.error)
    try:
        if isinstance(update, Update) and update.effective_chat:
            await safe_send(context.bot, update.effective_chat.id, "⚠️ Something went wrong — the error was logged.")
    except Exception:
        logger.exception("Failed to notify user about error.")

# ========== MAIN ==========
def main():
    if not TELEGRAM_TOKEN:
        logger.error("TELEGRAM_TOKEN missing — cannot start bot.")
        return
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    # commands
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CallbackQueryHandler(faq_callback))
    app.add_handler(CommandHandler("pdf", pdf_command))
    app.add_handler(CommandHandler("compare", compare_command))
    app.add_handler(CommandHandler("issues", issues_command))
    app.add_handler(CommandHandler("uploadfir", uploadfir_command))
    app.add_handler(CommandHandler("feedback", feedback_command))
    # default text handler
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, reply_to_user))
    app.add_error_handler(global_error_handler)

    logger.info("NyayaAI bot is running.")
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()

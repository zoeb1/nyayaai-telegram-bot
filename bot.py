# version 1.1
# PART 1/4: Imports, config, utilities
import os
import re
import uuid
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

# ---------------------------
# Load API keys
# ---------------------------

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)


# ---------------------------
# UTILITY: PDF text extractor
# ---------------------------
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

# ---------------------------
# UTILITY: Simple safe HTTP GET
# ---------------------------
def safe_get(url, headers=None, timeout=15):
    try:
        return requests.get(url, headers=headers or {"User-Agent": "Mozilla/5.0"}, timeout=timeout).text
    except Exception as e:
        print("HTTP error:", e)
        return ""

# PART 2/4: AI helper functions

# ---------------------------
# Query auto-corrector
# ---------------------------
async def correct_query(user_query):
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Correct the user's Indian legal query:\n"
                    "- Fix IPC/CrPC numbers\n"
                    "- Fix typos\n"
                    "- Add essential offence keyword if missing\n"
                    "- Keep output short (5-8 words)\n"
                    "- Output ONLY corrected query"
                )
            },
            {"role": "user", "content": user_query}
        ]
    )
    return resp.choices[0].message.content.strip()

# ---------------------------
# FIR -> query
# ---------------------------
async def convert_fir_to_query(fir_text):
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Read the FIR text and produce a short 5-8 word Indian Kanoon search query.\n"
                    "- MUST include IPC/CrPC if present.\n"
                    "- Include core offence word (murder, cheating, robbery etc.)\n"
                    "- No names, no locations, no long sentences.\n"
                    "- Output ONLY the query."
                )
            },
            {"role": "user", "content": fir_text}
        ]
    )
    return resp.choices[0].message.content.strip()

# ---------------------------
# Suggest better query when no results
# ---------------------------
async def suggest_better_query(bad_query):
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Rewrite the user's weak query into a strong Indian Kanoon search query.\n"
                    "- Add IPC/CrPC sections if inferable.\n"
                    "- Add offence keyword.\n"
                    "- Keep 5-8 words.\n"
                    "- Output only the improved query."
                )
            },
            {"role": "user", "content": bad_query}
        ]
    )
    return resp.choices[0].message.content.strip()

# ---------------------------
# Search Indian Kanoon (scrape)
# ---------------------------
def search_cases_kanoon(query, max_results=2):
    query = query.replace(" ", "+")
    url = f"https://indiankanoon.org/search/?formInput={query}"
    try:
        resp_text = safe_get(url)
        soup = BeautifulSoup(resp_text, "html.parser")
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
        print("Kanoon search error:", e)
        return []

# ---------------------------
# Fetch judgment text
# ---------------------------
def fetch_case_text(url):
    try:
        resp_text = safe_get(url)
        soup = BeautifulSoup(resp_text, "html.parser")
        paragraphs = soup.find_all("p")
        text = "\n".join(p.get_text(strip=True) for p in paragraphs)
        return text[:10000]
    except Exception as e:
        print("fetch_case_text error:", e)
        return ""

# ---------------------------
# Summarize
# ---------------------------
async def summarize_case(text, query):
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Summarize the judgment. Output: TITLE, SUMMARY, AI_SCORE:0-100"},
            {"role": "user", "content": f"{query}\n\nJudgment text:\n{text}"}
        ]
    )
    out = resp.choices[0].message.content
    ai_score = 0
    for line in out.splitlines():
        if "AI_SCORE" in line:
            try:
                ai_score = int(line.split(":")[1])
            except:
                pass
    return out, ai_score

# ---------------------------
# ILAC
# ---------------------------
async def generate_ilac_note(query, text):
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Write ILAC: ISSUE, LAW, APPLICATION, CONCLUSION."},
            {"role": "user", "content": f"{query}\n\n{text}"}
        ]
    )
    return resp.choices[0].message.content

# ---------------------------
# Arguments
# ---------------------------
async def generate_arguments(query, text):
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Produce PETITIONER ARGUMENTS, RESPONDENT ARGUMENTS, COUNTER ARGUMENTS."},
            {"role": "user", "content": f"{query}\n\n{text}"}
        ]
    )
    return resp.choices[0].message.content

# ---------------------------
# Citations
# ---------------------------
def extract_citations(text):
    case_pattern = r"[A-Z][A-Za-z .]+ vs\.? [A-Z][A-Za-z .]+"
    cases = list(dict.fromkeys(re.findall(case_pattern, text)))

    statute_pattern = r"(IPC\s*\d+|CrPC\s*\d+|Section\s*\d+|Evidence Act\s*\d+)"
    statutes = list(dict.fromkeys(re.findall(statute_pattern, text, flags=re.IGNORECASE)))

    out = "📌 CITED CASES:\n"
    out += "\n".join(f"- {c}" for c in cases) if cases else "No cited cases."
    out += "\n\n📌 STATUTES:\n"
    out += "\n".join(f"- {s}" for s in statutes) if statutes else "No statutes."
    return out

# ---------------------------
# Compare case
# ---------------------------
async def compare_case_with_query(user_query, judgment_text):
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an Indian legal analyst. Compare the user's query with the judgment text.\n"
                    "Provide structured output: MATCHES, DIFFERENCES, IMPACT (1-line conclusion). Keep concise."
                )
            },
            {"role": "user", "content": f"User Query:\n{user_query}\n\nJudgment:\n{judgment_text}"}
        ]
    )
    return resp.choices[0].message.content

# ---------------------------
# Extract legal issues
# ---------------------------
async def extract_legal_issues(user_query):
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": (
                "You are an Indian lawyer. Extract 3-6 legal issues from the user's query. "
                "Write them as questions courts would ask. Short and precise."
            )},
            {"role": "user", "content": user_query}
        ]
    )
    return resp.choices[0].message.content.strip()
# PART 3/4: PDF builder, branding, start/help/faq, pdf/uploadfir commands

# ---------------------------
# Build HTML for PDF with NyayaAI branding
# ---------------------------
def build_html(query, results, ilac, arguments, citations, qna):
    now = datetime.now().strftime("%d %b %Y %I:%M %p")
    header = f"""
    <!-- NYAYAAI HEADER -->
    <div style='background:#1F2A44;color:white;padding:20px;border-radius:8px;margin-bottom:25px;'>
        <div style='font-size:36px;font-weight:800;letter-spacing:1px;'>NYAYAAI</div>
        <div style='font-size:16px;margin-top:4px;color:#D0D4D9;'>India’s AI-Powered Legal Research Assistant</div>
    </div>
    """
    body = f"<p><b>Query:</b> {query}</p><p><b>Generated:</b> {now}</p><hr/>"
    for i, r in enumerate(results, 1):
        body += f"<h2>Case {i}</h2><p><b>Link:</b> {r['link']}</p><pre>{r['summary']}</pre><p>AI Score: {r['ai_score']} — Relevance: {r['relevance']}%</p><hr/>"
    body += f"<h2>ILAC Note</h2><pre>{ilac}</pre><hr/>"
    body += f"<h2>Arguments</h2><pre>{arguments}</pre><hr/>"
    body += f"<h2>Citations</h2><pre>{citations}</pre><hr/>"
    if qna:
        body += f"<h2>Q&A</h2><pre>{qna}</pre><hr/>"
    footer = "<div style='margin-top:40px;text-align:center;font-size:12px;color:gray;'>Report generated by NyayaAI – Automated Legal Research<br>© 2025 NyayaAI. All rights reserved.</div>"
    return f"<html><body style='font-family:Arial;padding:20px'>{header}{body}{footer}</body></html>"

# ---------------------------
# /start - updated
# ---------------------------
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "👋 *Welcome to NyayaAI — India’s AI-Powered Legal Research Assistant*\n\n"
        "NyayaAI helps you instantly:\n"
        "• Search Supreme & High Court case laws\n"
        "• Auto-correct legal queries (IPC/CrPC)\n"
        "• Summarize judgments + AI relevance score\n"
        "• Generate ILAC notes & legal arguments\n"
        "• Extract citations (cases + statutes)\n"
        "• Compare your query with real judgments (/compare)\n"
        "• Identify legal issues clearly (/issues)\n"
        "• Create full PDF legal research reports (/pdf)\n"
        "• Convert FIR PDFs → Search queries → Case laws (/uploadfir)\n\n"
        "📌 *Popular Commands*\n"
        "🔍 Search Cases: `IPC 420 cheating`\n"
        "📄 Generate PDF: `/pdf IPC 302 murder`\n"
        "📂 FIR to Case Law: `/uploadfir` (reply to FIR PDF)\n"
        "📊 Compare Case: `/compare IPC 420 false promise`\n"
        "⚖️ Extract Issues: `/issues 498A cruelty`\n\n"
        "Type `help` or press the Help button for more options."
    )
    # Add inline help button for easy access
    keyboard = [[InlineKeyboardButton("Help", callback_data="HELP_MENU")]]
    await update.message.reply_text(msg, parse_mode="Markdown", reply_markup=InlineKeyboardMarkup(keyboard))

# ---------------------------
# /help command and callback handler (FAQ with buttons)
# ---------------------------
FAQ_BUTTONS = [
    ("What can NyayaAI do?", "FAQ_CAPABILITIES"),
    ("How to convert FIR to query?", "FAQ_FIR"),
    ("How to generate PDF?", "FAQ_PDF"),
    ("How accurate are summaries?", "FAQ_ACCURACY"),
    ("Pricing & limits", "FAQ_PRICING"),
    ("Contact / Feedback", "FAQ_CONTACT"),
    ("Close", "FAQ_CLOSE"),
]

def build_faq_keyboard():
    # two-column layout
    buttons = []
    row = []
    for i, (label, key) in enumerate(FAQ_BUTTONS):
        row.append(InlineKeyboardButton(label, callback_data=key))
        if len(row) == 2:
            buttons.append(row)
            row = []
    if row:
        buttons.append(row)
    return InlineKeyboardMarkup(buttons)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = "📘 *NyayaAI Help Menu*\n\nChoose an item to learn more (press buttons)."
    await update.message.reply_text(msg, parse_mode="Markdown", reply_markup=build_faq_keyboard())

# Callback handler for FAQ buttons
async def faq_button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    key = query.data

    if key == "HELP_MENU":
        await query.edit_message_text(
            "📘 *NyayaAI Help Menu*\n\nChoose an item to learn more (press buttons).",
            parse_mode="Markdown",
            reply_markup=build_faq_keyboard()
        )
        return

    if key == "FAQ_CLOSE":
        await query.edit_message_text("Closed help menu.")
        return

    if key == "FAQ_CAPABILITIES":
        text = (
            "*What NyayaAI can do:*\n"
            "- Search Supreme/High Court judgments\n"
            "- Summaries + relevance scores\n"
            "- ILAC notes & arguments\n"
            "- Citations\n"
            "- Compare cases (/compare)\n"
            "- Extract legal issues (/issues)\n"
            "- Generate PDF reports (/pdf)\n"
            "- Process FIR PDFs (/uploadfir)\n"
        )

    elif key == "FAQ_FIR":
        text = (
            "*FIR → Query:* Reply to a FIR PDF with `/uploadfir`.\n"
            "NyayaAI will generate the best legal search query and full research report."
        )

    elif key == "FAQ_PDF":
        text = (
            "*PDF Reports:*\n"
            "Use `/pdf <query>`.\n"
            "Example: `/pdf IPC 302 murder`\n"
            "NyayaAI will summarize top cases, ILAC, arguments & citations."
        )

    elif key == "FAQ_ACCURACY":
        text = (
            "*About Accuracy:*\n"
            "Summaries & relevance scores are AI-generated and intended to support legal research."
        )

    elif key == "FAQ_PRICING":
        text = (
            "*Pricing:* Premium plans can include daily limits, report credits, or monthly subscriptions."
        )

    elif key == "FAQ_CONTACT":
        text = (
            "*Contact / Feedback:*\n"
            "Email: zoebsadeqa@gmail.com"
        )

    else:
        text = "Unknown item."

    back_kb = InlineKeyboardMarkup([
        [InlineKeyboardButton("Back", callback_data="HELP_MENU"),
         InlineKeyboardButton("Close", callback_data="FAQ_CLOSE")]
    ])

    await query.edit_message_text(text, parse_mode="Markdown", reply_markup=back_kb)

# ---------------------------
# /pdf command
# ---------------------------
async def pdf_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        return await update.message.reply_text("Usage: /pdf <query>")

    raw = " ".join(context.args)
    fixed = await correct_query(raw)
    if fixed.lower() != raw.lower():
        await update.message.reply_text(f"🔧 Did you mean:\n👉 `{fixed}`", parse_mode="Markdown")
    query = fixed

    await update.message.reply_text("🔎 Scanning case laws...")
    links = search_cases_kanoon(query)
    if not links:
        better = await suggest_better_query(query)
        return await update.message.reply_text(f"No cases found. Try: 👉 `{better}`", parse_mode="Markdown")

    await update.message.reply_text("📘 Summarizing case laws...")
    results = []
    first_text = ""
    for link in links:
        t = fetch_case_text(link)
        first_text = first_text or t
        s, ai = await summarize_case(t, query)
        rel = compute_relevance_score(ai, query, t)
        results.append({"link": link, "summary": s, "ai_score": ai, "relevance": rel})

    await update.message.reply_text("🤖 Generating ILAC, arguments, citations...")
    ilac = await generate_ilac_note(query, first_text)
    arguments = await generate_arguments(query, first_text)
    citations = extract_citations(first_text)

    qna = ""
    if " ? " in raw:
        _, question = raw.split(" ? ", 1)
        ans = await ask_about_case(question, first_text)
        qna = f"Q: {question}\n\n{ans}"

    await update.message.reply_text("📄 Creating PDF...")
    os.makedirs("pdfs", exist_ok=True)
    filename = f"pdfs/{uuid.uuid4().hex}.pdf"
    html = build_html(query, results, ilac, arguments, citations, qna)
    pdfkit.from_string(html, filename)
    await update.message.reply_document(open(filename, "rb"))

# ---------------------------
# /uploadfir command
# ---------------------------
async def uploadfir_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("⚙️ Extracting FIR...")
    if not update.message.reply_to_message or not update.message.reply_to_message.document:
        return await update.message.reply_text("❗ Reply to a FIR PDF and type /uploadfir")

    file = update.message.reply_to_message.document
    if not file.file_name.lower().endswith(".pdf"):
        return await update.message.reply_text("❗ Only PDF files allowed.")

    os.makedirs("pdfs", exist_ok=True)
    file_path = f"pdfs/{file.file_name}"
    tg_file = await file.get_file()
    await tg_file.download_to_drive(file_path)

    fir_text = extract_text_from_pdf(file_path)
    if not fir_text:
        return await update.message.reply_text("❗ Could not read FIR text.")

    await update.message.reply_text("🤖 Creating optimum legal query...")
    smart_query = await convert_fir_to_query(fir_text)
    await update.message.reply_text(f"🔍 Auto Query:\n`{smart_query}`", parse_mode="Markdown")

    context.args = smart_query.split()
    await pdf_command(update, context)
# PART 4/4: compare, issues, default handler, main()

# ---------------------------
# /compare command
# ---------------------------
async def compare_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    raw = " ".join(context.args)
    if not raw:
        return await update.message.reply_text("Usage: /compare <query>")

    fixed = await correct_query(raw)
    if fixed.lower() != raw.lower():
        await update.message.reply_text(f"🔧 Did you mean:\n👉 `{fixed}`", parse_mode="Markdown")
    query = fixed

    await update.message.reply_text("🔍 Finding closest matching judgment...")
    links = search_cases_kanoon(query)
    if not links:
        better = await suggest_better_query(query)
        return await update.message.reply_text(f"No cases found. Try: 👉 `{better}`", parse_mode="Markdown")

    text = fetch_case_text(links[0])
    await update.message.reply_text("📘 Comparing facts...")
    comparison = await compare_case_with_query(query, text)
    await update.message.reply_text(f"📎 *Case:* {links[0]}\n\n📊 *CASE COMPARISON*\n\n{comparison}", parse_mode="Markdown")

# ---------------------------
# /issues command
# ---------------------------
async def issues_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    raw = " ".join(context.args)
    if not raw:
        return await update.message.reply_text("Usage: /issues <query>")

    fixed = await correct_query(raw)
    if fixed.lower() != raw.lower():
        await update.message.reply_text(f"🔧 Did you mean:\n👉 `{fixed}`", parse_mode="Markdown")
    query = fixed

    await update.message.reply_text("📘 Identifying legal issues...")
    issues = await extract_legal_issues(query)
    await update.message.reply_text(f"📌 *LEGAL ISSUES IDENTIFIED*\n\n{issues}", parse_mode="Markdown")

# ---------------------------
# Help command wrapper
# ---------------------------
async def help_command_wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await help_command(update, context)

# ---------------------------
# Default text handler with help keyword detection
# ---------------------------
HELP_KEYWORDS = [
    "what does this bot do", "help", "how to use", "features",
    "bot kya karta hai", "what can you do", "guide", "instructions"
]

async def reply_to_user(update: Update, context: ContextTypes.DEFAULT_TYPE):
    raw_original = update.message.text or ""
    raw = raw_original.lower().strip()

    # check for help-like queries
    if any(k in raw for k in HELP_KEYWORDS):
        # reuse help menu
        keyboard = [[InlineKeyboardButton("Help", callback_data="HELP_MENU")]]
        await update.message.reply_text("Press Help for options or type /help", reply_markup=InlineKeyboardMarkup(keyboard))
        return

    # otherwise process as query
    fixed = await correct_query(raw_original)
    if fixed.lower() != raw_original.lower():
        await update.message.reply_text(f"🔧 Did you mean:\n👉 `{fixed}`", parse_mode="Markdown")
    q = fixed

    links = search_cases_kanoon(q)
    if not links:
        better = await suggest_better_query(q)
        return await update.message.reply_text(f"No cases found. Try: 👉 `{better}`", parse_mode="Markdown")

    final = ""
    for link in links:
        t = fetch_case_text(link)
        s, ai = await summarize_case(t, q)
        rel = compute_relevance_score(ai, q, t)
        final += f"{link}\n{s}\nRelevance: {rel}%\n\n"

    await update.message.reply_text(final)

# ---------------------------
# START BOT (register handlers)
# ---------------------------
def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    # commands
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command_wrapper))
    # callback for FAQ buttons
    app.add_handler(CallbackQueryHandler(faq_button_handler))
    # core features
    app.add_handler(CommandHandler("pdf", pdf_command))
    app.add_handler(CommandHandler("uploadfir", uploadfir_command))
    app.add_handler(CommandHandler("compare", compare_command))
    app.add_handler(CommandHandler("issues", issues_command))
    # default text handler
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, reply_to_user))

    print("BOT RUNNING...")
    app.run_polling()

if __name__ == "__main__":
    main()

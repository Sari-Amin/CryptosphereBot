#!/usr/bin/env python3
"""
Crypto news + BTC price bot for Telegram.

- Watches RSS feeds of major outlets (BBC, Guardian, CNBC, Al Jazeera + crypto media)
- Uses an LLM to decide if a story could affect crypto markets
- Posts valuable stories to your Telegram channel
- Posts BTC price alerts every time price crosses a $1,000 level (up or down)

Setup:
    pip install feedparser requests google-genai
    export TELEGRAM_BOT_TOKEN="123456:ABC..."
    export TELEGRAM_CHAT_ID="@yourchannel"      # or -100xxxxxxxxxx
    export GEMINI_API_KEY="AIza..."          # from Google AI Studio
    python crypto_news_bot.py
"""
import html
import json
import math
import os
import re
import sqlite3
import sys
import time
from datetime import datetime, timezone

import feedparser
import requests
from google import genai
from google.genai import types

sys.stdout.reconfigure(line_buffering=True)
START = time.time()

# ---------------- CONFIG ----------------
TELEGRAM_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
# Fast, cheap model, minimal thinking by default (good for scoring headlines).
# gemini-2.5-flash is being shut down by Google on Dec 1, 2026, so don't use it.
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")

PRICE_CHECK_SECONDS = 60      # how often to check BTC price
NEWS_CHECK_SECONDS = 600      # how often to check news (10 min)
PRICE_STEP = 1000             # alert every $1000
PRICE_BUFFER = 150            # must pass the level by this much (prevents spam when price hovers)

SCORE_THRESHOLD = 7           # post news with score >= this (1-10)
MAX_POSTS_PER_CYCLE = 5       # flood protection
MAX_SCORED_PER_RUN = 25       # cap LLM calls per run (free-tier rate limits)
LLM_PAUSE_SECONDS = 4         # pause between LLM calls (~15 requests/min)
TIME_BUDGET_SECONDS = 120     # stop scoring after this long (job timeout is 4 min)
MAX_LLM_FAILURES = 3          # give up on the LLM for this run after this many errors
llm_failures = 0
MAX_AGE_HOURS = 12            # ignore older stories

# General outlets: pre-filtered by keywords before calling the LLM (saves cost)
GENERAL_FEEDS = {
    "BBC Business": "https://feeds.bbci.co.uk/news/business/rss.xml",
    "BBC World": "https://feeds.bbci.co.uk/news/world/rss.xml",
    "BBC Tech": "https://feeds.bbci.co.uk/news/technology/rss.xml",
    "Guardian Business": "https://www.theguardian.com/business/rss",
    "CNBC Finance": "https://www.cnbc.com/id/10000664/device/rss/rss.html",
    "CNBC Top": "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "Al Jazeera": "https://www.aljazeera.com/xml/rss/all.xml",
}
# Crypto outlets: sent straight to the LLM
CRYPTO_FEEDS = {
    "CoinDesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "Cointelegraph": "https://cointelegraph.com/rss",
    "Decrypt": "https://decrypt.co/feed",
}

KEYWORDS = re.compile(
    r"bitcoin|btc|crypto|ethereum|stablecoin|binance|coinbase|blockchain|\bsec\b|etf|"
    r"federal reserve|\bfed\b|interest rate|rate cut|rate hike|inflation|cpi|treasury|"
    r"tariff|sanction|bank collapse|bank failure|recession|dollar|debt ceiling|"
    r"war|ceasefire|missile|oil price|trump|powell|regulat|ban|hack|exploit",
    re.I,
)

SYSTEM_PROMPT = """You filter news for a crypto Telegram channel.
Score 1-10 how likely the story is to move crypto markets or matter to crypto traders.
High (7-10): crypto regulation/laws, ETF decisions, major hacks/exploits, exchange or
stablecoin trouble, central bank rate decisions, major macro shocks (inflation data,
bank failures, sanctions, big geopolitical escalation), large institutional/government
bitcoin moves, major listings.
Low (1-6): opinion, price commentary, minor altcoin news, ads, sponsored content,
general news with no plausible market link.
Return ONLY JSON: {"score": int, "summary": "1-2 sentence neutral summary in your own words",
"impact": "one short line on why it matters for crypto"}"""

# ---------------- STORAGE ----------------
db = sqlite3.connect("bot.db")
db.execute("CREATE TABLE IF NOT EXISTS seen (id TEXT PRIMARY KEY, ts INTEGER)")
db.execute("CREATE TABLE IF NOT EXISTS state (k TEXT PRIMARY KEY, v TEXT)")
db.commit()


def get_state(k, default=None):
    row = db.execute("SELECT v FROM state WHERE k=?", (k,)).fetchone()
    return row[0] if row else default


def set_state(k, v):
    db.execute("INSERT OR REPLACE INTO state VALUES (?,?)", (k, str(v)))
    db.commit()


def is_seen(i):
    return db.execute("SELECT 1 FROM seen WHERE id=?", (i,)).fetchone() is not None


def mark_seen(i):
    db.execute("INSERT OR IGNORE INTO seen VALUES (?,?)", (i, int(time.time())))
    db.commit()


# ---------------- TELEGRAM ----------------
def send_telegram(text):
    r = requests.post(
        f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
        json={
            "chat_id": CHAT_ID,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": False,
        },
        timeout=20,
    )
    if not r.ok:
        print("Telegram error:", r.status_code, r.text)
    return r.ok


# ---------------- BTC PRICE ----------------
def get_btc_price():
    sources = [
        ("https://api.coinbase.com/v2/prices/BTC-USD/spot",
         lambda j: float(j["data"]["amount"])),
        ("https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd",
         lambda j: float(j["bitcoin"]["usd"])),
        ("https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT",
         lambda j: float(j["price"])),
    ]
    for url, parse in sources:
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            return parse(r.json())
        except Exception as e:
            print(f"Price source failed ({url.split('/')[2]}): {e}")
    return None


def check_price():
    price = get_btc_price()
    if price is None:
        return
    level = math.floor(price / PRICE_STEP)
    last = get_state("last_level")

    if last is None:  # first run: just record, don't alert
        set_state("last_level", level)
        print(f"Price tracking started at ${price:,.0f}")
        return

    last = int(last)
    if level == last:
        return

    if level > last:
        passed = price >= level * PRICE_STEP + PRICE_BUFFER
        emoji, word = "🟢📈", "above"
        boundary = level * PRICE_STEP
    else:
        passed = price <= (level + 1) * PRICE_STEP - PRICE_BUFFER
        emoji, word = "🔴📉", "below"
        boundary = (level + 1) * PRICE_STEP

    if not passed:
        return  # hovering around the line, wait for a clear move

    msg = (f"{emoji} <b>BTC moved {word} ${boundary:,.0f}</b>\n"
           f"Current price: <b>${price:,.0f}</b>")
    if send_telegram(msg):
        set_state("last_level", level)


# ---------------- NEWS ----------------
client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"],
    http_options=types.HttpOptions(timeout=60000),  # 60s per request
)

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer"},
        "summary": {"type": "string"},
        "impact": {"type": "string"},
    },
    "required": ["score", "summary", "impact"],
}


def parse_json(text):
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    return json.loads(text)


def score_item(title, summary, source):
    global llm_failures
    prompt = f"Source: {source}\nTitle: {title}\nText: {summary[:800]}"
    for attempt in range(2):
        if llm_failures >= MAX_LLM_FAILURES:
            return None
        try:
            r = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    response_schema=RESPONSE_SCHEMA,
                    temperature=0,
                ),
            )
            time.sleep(LLM_PAUSE_SECONDS)
            return parse_json(r.text)
        except Exception as e:
            llm_failures += 1
            print(f"LLM error ({llm_failures}/{MAX_LLM_FAILURES}): {e}")
            if llm_failures < MAX_LLM_FAILURES:
                time.sleep(10)
    return None


def entry_age_hours(e):
    t = e.get("published_parsed") or e.get("updated_parsed")
    if not t:
        return 0
    ts = datetime(*t[:6], tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - ts).total_seconds() / 3600


def fetch_feed(url):
    r = requests.get(url, timeout=10,
                     headers={"User-Agent": "Mozilla/5.0 (crypto-news-bot)"})
    r.raise_for_status()
    return feedparser.parse(r.content)


def check_news(first_run=False):
    posted = 0
    scored = 0
    all_feeds = [(n, u, False) for n, u in GENERAL_FEEDS.items()] + \
                [(n, u, True) for n, u in CRYPTO_FEEDS.items()]

    for name, url, is_crypto_outlet in all_feeds:
        try:
            feed = fetch_feed(url)
        except Exception as e:
            print(f"Feed failed {name}: {e}")
            continue

        for e in feed.entries[:25]:
            uid = e.get("id") or e.get("link")
            if not uid or is_seen(uid):
                continue

            # first run: record everything, post nothing (avoids flooding the channel)
            if first_run or entry_age_hours(e) > MAX_AGE_HOURS:
                mark_seen(uid)
                continue

            title = e.get("title", "")
            summary = re.sub(r"<[^>]+>", "", e.get("summary", ""))
            if not is_crypto_outlet and not KEYWORDS.search(f"{title} {summary}"):
                mark_seen(uid)
                continue

            # over the per-run limits: leave unseen so the next run handles it
            if posted >= MAX_POSTS_PER_CYCLE or scored >= MAX_SCORED_PER_RUN:
                continue

            if time.time() - START > TIME_BUDGET_SECONDS or llm_failures >= MAX_LLM_FAILURES:
                continue  # leave unseen, next run handles it

            scored += 1
            res = score_item(title, summary, name)
            if res is None:
                continue  # LLM failed: leave unseen and retry next run
            mark_seen(uid)

            print(f"[{res.get('score')}] {name}: {title}")
            if res.get("score", 0) >= SCORE_THRESHOLD:
                msg = (f"📰 <b>{html.escape(title)}</b>\n\n"
                       f"{html.escape(res.get('summary', ''))}\n\n"
                       f"💡 <i>{html.escape(res.get('impact', ''))}</i>\n\n"
                       f"Source: {html.escape(name)} | "
                       f"<a href=\"{html.escape(e.get('link', ''))}\">Read more</a>")
                if send_telegram(msg):
                    posted += 1
    print(f"News cycle done: scored {scored}, posted {posted}")


# ---------------- RUN MODES ----------------
def run_once():
    """Single pass, for schedulers like GitHub Actions."""
    first_run = get_state("initialized") is None
    print(f"Run start (first_run={first_run})")
    check_price()
    print("Price check done")
    check_news(first_run=first_run)
    set_state("initialized", "1")
    # keep the DB small: forget stories older than 14 days
    db.execute("DELETE FROM seen WHERE ts < ?", (int(time.time()) - 14 * 86400,))
    db.commit()


def main():
    if "--once" in sys.argv:
        run_once()
        return
    print("Bot started")
    first_run = get_state("initialized") is None
    check_news(first_run=first_run)
    set_state("initialized", "1")
    check_price()

    last_news = time.time()
    while True:
        time.sleep(PRICE_CHECK_SECONDS)
        try:
            check_price()
            if time.time() - last_news >= NEWS_CHECK_SECONDS:
                check_news()
                last_news = time.time()
        except Exception as e:
            print("Loop error:", e)


if __name__ == "__main__":
    main()
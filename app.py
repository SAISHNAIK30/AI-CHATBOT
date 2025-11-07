# kb_chat.py — Confluence KB + OpenAI hybrid bot
# - Full-site crawl (everything the bot user can view)
# - Error Catalog row-first retrieval (exact/near error match ⇒ single Remedy path)
# - Creator Qs → "Saish Naik" (configurable; robust matcher)
# - Confluence links → force KB path; no LLM fallback if no KB match
# - Screenshot OCR + chat memory + login isolation
# - Fixed ranked.sort bug; short, specific, single-path answers

import os, re, io, math, json, uuid, threading
from datetime import datetime
from collections import Counter
from functools import wraps

import pytesseract
from PIL import Image
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from flask import Flask, request, render_template_string, session, redirect, url_for, abort
from rapidfuzz import fuzz

# ======================
# ENV & CONFIG (set in .env)
# Required:
#   CONFLUENCE_BASE_URL=https://<site>.atlassian.net
#   CONFLUENCE_USER_EMAIL=<bot email>
#   CONFLUENCE_API_TOKEN=<token>
# Recommended:
#   FULL_SITE_CRAWL=1
#   USE_LLM=1, OPENAI_API_KEY=<key>, LLM_MODEL=gpt-4o-mini
# Precision & behavior:
#   PINPOINT=1
#   ROW_SCORE_MIN=0.90
#   MAX_SECTIONS=1
#   MAX_SOURCES_PER_ANSWER=1
#   PINPOINT_MAX_STEPS=8
#   MAX_ANSWER_CHARS=-1
# Error Catalog preference:
#   ERROR_TITLE_HINTS=error,errors,solution,solutions,catalog
#   ERROR_PAGE_BIAS=0.18
#   ERROR_ROW_HARDMATCH=1
# Branding:
#   CREATOR_NAME=Saish Naik
# ======================
load_dotenv()

CONFLUENCE_BASE_URL   = (os.getenv("CONFLUENCE_BASE_URL", "").rstrip("/"))
CONFLUENCE_USER_EMAIL = os.getenv("CONFLUENCE_USER_EMAIL") or os.getenv("CONFLUENCE_EMAIL")
CONFLUENCE_API_TOKEN  = os.getenv("CONFLUENCE_API_TOKEN")

SPACE_KEYS                 = [s.strip() for s in os.getenv("SPACE_KEYS","").split(",") if s.strip()]
ERROR_CATALOG_PARENT_URL   = os.getenv("ERROR_CATALOG_PARENT_URL", "")
MAX_PAGES_PER_SPACE        = int(os.getenv("MAX_PAGES_PER_SPACE","2000"))

FULL_SITE_CRAWL            = os.getenv("FULL_SITE_CRAWL","0") == "1"
MAX_PAGES_TOTAL            = int(os.getenv("MAX_PAGES_TOTAL","10000"))

USE_LLM        = os.getenv("USE_LLM","1") == "1"
LLM_MODEL      = os.getenv("LLM_MODEL","gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY","")

CREATOR_NAME   = os.getenv("CREATOR_NAME","Saish Naik")

FLASK_SECRET_KEY = os.getenv("APP_SECRET_KEY") or os.getenv("FLASK_SECRET_KEY","dev-secret-change-me")

USERS_JSON   = os.getenv("USERS_JSON","").strip()
APP_USERNAME = os.getenv("APP_USERNAME","").strip()
APP_PASSWORD = os.getenv("APP_PASSWORD","").strip()

TESSERACT_CMD = os.getenv("TESSERACT_CMD")
if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
else:
    default_win_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(default_win_path):
        pytesseract.pytesseract.tesseract_cmd = default_win_path

MAX_SECTIONS            = int(os.getenv("MAX_SECTIONS","1"))
MAX_ANSWER_CHARS        = int(os.getenv("MAX_ANSWER_CHARS","-1"))  # -1 disables trimming
MAX_STEP_LINES          = 12
MAX_HISTORY_MSGS        = 240
FOLLOWUP_CONTEXT_TURNS  = 6

PINPOINT           = os.getenv("PINPOINT","1") == "1"
ROW_SCORE_MIN      = float(os.getenv("ROW_SCORE_MIN","0.90"))
PINPOINT_MAX_STEPS = int(os.getenv("PINPOINT_MAX_STEPS","8"))
MAX_SOURCES_PER_ANSWER = int(os.getenv("MAX_SOURCES_PER_ANSWER","1"))

ERROR_TITLE_HINTS = [s.strip().lower() for s in os.getenv("ERROR_TITLE_HINTS","error,errors,solution,solutions,catalog").split(",") if s.strip()]
ERROR_PAGE_BIAS   = float(os.getenv("ERROR_PAGE_BIAS","0.18"))
ERROR_ROW_HARDMATCH = os.getenv("ERROR_ROW_HARDMATCH","1") == "1"

STORE_PATH = os.getenv("CHAT_STORE_PATH","chats_store.json")

if not (CONFLUENCE_BASE_URL and CONFLUENCE_USER_EMAIL and CONFLUENCE_API_TOKEN):
    raise RuntimeError("Missing required env: CONFLUENCE_BASE_URL, CONFLUENCE_USER_EMAIL, CONFLUENCE_API_TOKEN")

if CONFLUENCE_BASE_URL.endswith("/wiki"):
    CONFLUENCE_BASE_URL = CONFLUENCE_BASE_URL[:-5]
CONFLUENCE_BASE_URL = CONFLUENCE_BASE_URL.rstrip("/")

# ======================
# OpenAI helper
# ======================
_openai_client = None
def _get_openai():
    global _openai_client
    if _openai_client or not (USE_LLM and OPENAI_API_KEY):
        return _openai_client
    try:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
        return _openai_client
    except Exception as e:
        print("[LLM] OpenAI client unavailable:", e)
        return None

def llm_chat(prompt: str, temperature: float = 0.25) -> str:
    client = _get_openai()
    if not client: return ""
    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role":"user","content":prompt}],
            temperature=temperature,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print("[LLM] error:", e)
        return ""

def llm_chat_messages(messages: list, temperature: float = 0.25) -> str:
    client = _get_openai()
    if not client: return ""
    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=temperature,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print("[LLM] error:", e)
        return ""

# ======================
# Flask
# ======================
app = Flask(__name__)
app.secret_key = FLASK_SECRET_KEY

def _load_users():
    users = []
    if USERS_JSON:
        try:
            users = json.loads(USERS_JSON)
        except Exception:
            pass
    if not users and APP_USERNAME and APP_PASSWORD:
        users = [{"id": APP_USERNAME, "password": APP_PASSWORD}]
    if not users:
        users = [{"id":"demo", "password":"demo"}]
    return users

def _check_login(uid, pw):
    for u in _load_users():
        if u.get("id")==uid and u.get("password")==pw:
            return True
    return False

def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not session.get("uid"):
            return redirect(url_for("login", next=request.path))
        return f(*args, **kwargs)
    return wrapper

LOGIN_HTML = """
<!doctype html>
<html>
<head>
  <title>Sign in - KB Chat</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body{margin:0; background:#f3f6fb; font-family:Inter,system-ui,Roboto,Arial,sans-serif;}
    .card{max-width:420px; margin:12vh auto; background:#fff; padding:24px; border-radius:16px; box-shadow:0 10px 25px rgba(0,0,0,.08)}
    h1{margin:0 0 12px 0; font-size:20px}
    label{display:block; font-size:14px; color:#374151; margin:12px 0 6px}
    input{width:100%; padding:10px 12px; border:1px solid #e5e7eb; border-radius:10px; font-size:15px}
    .row{display:flex; gap:8px; align-items:center; justify-content:space-between; margin-top:18px}
    button{background:#3b82f6; color:#fff; border:none; padding:10px 14px; border-radius:10px; cursor:pointer}
    .muted{color:#6b7280; font-size:13px}
    .err{color:#ef4444; margin-top:10px}
  </style>
</head>
<body>
  <div class="card">
    <h1>Sign in</h1>
    {% if err %}<div class="err">{{ err }}</div>{% endif %}
    <form method="POST" autocomplete="off">
      <label>User ID</label>
      <input name="uid" autofocus>
      <label>Password</label>
      <input name="pw" type="password">
      <div class="row">
        <span class="muted">Only your chats will be visible after sign in.</span>
        <button>Sign in</button>
      </div>
    </form>
  </div>
</body>
</html>
"""

@app.route("/login", methods=["GET","POST"])
def login():
    err = ""
    if request.method=="POST":
        uid = (request.form.get("uid") or "").strip()
        pw  = (request.form.get("pw") or "").strip()
        if _check_login(uid, pw):
            session["uid"] = uid
            nxt = request.args.get("next") or url_for("root")
            return redirect(nxt)
        err = "Invalid credentials."
    return render_template_string(LOGIN_HTML, err=err)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# ------------- Main HTML Template -------------
HTML = """
<!doctype html>
<html>
<head>
  <title>KB Chat</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root{
      --bg:#f3f6fb; --card:#ffffff; --accent:#3b82f6; --accent-2:#10b981; --muted:#6b7280;
      --user:#dcfce7; --bot:#eaf2ff; --shadow:0 10px 25px rgba(0,0,0,.06);
    }
    *{box-sizing:border-box}
    body{margin:0; background:var(--bg); font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;}
    .app{display:flex; min-height:100vh; overflow:hidden;}
    .side{width: 280px; background:#0f172a; color:#cbd5e1; padding:20px 14px; display:flex; flex-direction:column; gap:16px;}
    .side h1{font-size:18px; color:#e2e8f0; margin:0 6px 8px 6px;}
    .new-btn{background:var(--accent-2); color:#052e1f; border:none; padding:10px 12px; border-radius:10px; cursor:pointer; font-weight:600; box-shadow: var(--shadow)}
    .chats{flex:1; overflow:auto; padding-right:4px;}
    .chat-link{display:block; color:#cbd5e1; text-decoration:none; padding:10px 12px; border-radius:10px; margin:6px 4px; background:rgba(255,255,255,0.04)}
    .chat-link.active{background:#1f2937; color:#fff; outline:1px solid rgba(255,255,255,0.12)}
    .small{font-size:12px; color:#94a3b8; margin:8px 6px}
    .main{flex:1; display:flex; flex-direction:column; }
    .head{padding:16px 22px; background:var(--card); box-shadow:var(--shadow); display:flex; align-items:center; gap:10px; position:sticky; top:0; z-index:5}
    .head h2{margin:0; font-size:18px;}
    .wrap{flex:1; overflow:auto; padding:26px 18px 120px}
    .msgs{max-width:900px; margin:0 auto;}
    .msg{display:flex; gap:10px; margin:14px 0;}
    .bubble{background:var(--card); box-shadow:var(--shadow); padding:18px 18px; border-radius:16px; max-width:75%; white-space: pre-wrap; line-height:1.45}
    .u .bubble{background:var(--user)}
    .b .bubble{background:var(--bot)}
    .avatar{width:28px; height:28px; border-radius:50%; display:flex; align-items:center; justify-content:center; background:#1e293b; color:#fff; font-weight:700}
    .u .avatar{background:#3b82f6;}
    .b .avatar{background:#111827;}
    .sources{ margin-top:6px; font-size:13px; color:#6b7280; }
    .sources a{color:#111827; background:#f3f4f6; padding:2px 8px; border-radius:6px; text-decoration:none; margin-right:8px;}
    .dock{position:sticky; bottom:0; background:linear-gradient(to top, #f3f6fb, rgba(243,246,251,.7)); padding:10px 0 18px; }
    .inwrap{max-width:900px; margin:0 auto; background:var(--card); box-shadow:var(--shadow); border-radius:16px; padding:8px;}
    textarea{width:100%; min-height:80px; border:none; outline:none; font-size:16px; padding:12px; resize:vertical; border-radius:12px;}
    .row{display:flex; align-items:center; gap:10px; flex-wrap:wrap; padding:8px 8px 2px}
    .send{background:var(--accent); color:#fff; border:none; padding:10px 16px; border-radius:10px; cursor:pointer; font-weight:700}
    .new{background:#111827; color:#fff; border:none; padding:10px 14px; border-radius:10px; cursor:pointer}
    .file{font-size:14px;}
    .hint{color:#6b7280; font-size:12px; padding:0 12px 10px}
    .gap{height:24px}
    .loader{display:inline-block; min-width:60px}
    .dot{display:inline-block; width:6px; height:6px; margin-right:3px; background:#9ca3af; border-radius:50%; animation:blink 1.2s infinite}
    .dot:nth-child(2){animation-delay:.2s}
    .dot:nth-child(3){animation-delay:.4s}
    @keyframes blink{0%{opacity:.2} 20%{opacity:1} 100%{opacity:.2}}
    .mic{background:#f3f4f6; border:none; padding:10px 12px; border-radius:10px; cursor:pointer; color:#111827; font-weight:600}
    .mic.recording{outline:2px solid #ef4444; background:#fee2e2}
    .mic[disabled]{opacity:.5; cursor:not-allowed}
    .hdr-right{margin-left:auto; display:flex; align-items:center; gap:10px}
    .logout{background:#ef4444; color:#fff; border:none; padding:8px 10px; border-radius:10px; cursor:pointer; font-weight:700}
    @media(max-width: 960px){ .side{display:none} .bubble{max-width: 90%} }
  </style>
</head>
<body>
  <div class="app">
    <aside class="side">
      <h1>KB Chats</h1>
      <form action="{{ url_for('new_chat') }}" method="post" style="margin:0">
        <button class="new-btn" type="submit">+ New chat</button>
      </form>
      <div class="small">Sessions (right-click to delete)</div>
      <div class="chats" id="chats">
        {% for c in chats %}
          <a class="chat-link {% if c.id == chat_id %}active{% endif %}" href="{{ url_for('chat', chat_id=c.id) }}" data-id="{{ c.id }}">
            {{ c.title }}
            <div class="small">{{ c.when }}</div>
          </a>
        {% endfor %}
      </div>
      <div class="small">KB preference → OpenAI fallback</div>
    </aside>

    <main class="main">
      <div class="head">
        <h2>{{ chat_title }}</h2>
        <div class="hdr-right">
          <div class="small">Signed in as <b>{{ user_id }}</b></div>
          <a href="{{ url_for('logout') }}"><button class="logout" type="button">Logout</button></a>
        </div>
      </div>

      <div class="wrap" id="wrap">
        <div class="msgs" id="msgs">
          {% for m in messages %}
            <div class="msg {% if m.role == 'user' %}u{% else %}b{% endif %}">
              <div class="avatar">{% if m.role == 'user' %}U{% else %}L{% endif %}</div>
              <div class="bubble">
                {{ m.text }}
                {% if m.sources %}
                  <div class="sources">
                    {% for s in m.sources %}
                      <a href="{{ s.url }}" target="_blank">{{ s.title }}</a>
                    {% endfor %}
                  </div>
                {% endif %}
              </div>
            </div>
          {% endfor %}
          {% if loading %}
            <div class="msg b">
              <div class="avatar">L</div>
              <div class="bubble"><span class="loader"><span class="dot"></span><span class="dot"></span><span class="dot"></span></span> thinking…</div>
            </div>
          {% endif %}
          <div class="gap"></div>
        </div>
      </div>

      <div class="dock">
        <div class="inwrap">
          <form id="f" method="POST" enctype="multipart/form-data" autocomplete="off">
            <textarea id="ta" name="text" placeholder="Type your error or chat question… (Enter=Send, Shift+Enter=new line)"></textarea>
            <input type="hidden" id="shadowText" name="text" value="">
            <div class="row">
              <input class="file" type="file" name="image" accept="image/*">
              <button type="button" class="mic" id="micBtn">🎤 Start</button>
              <div style="flex:1"></div>
              <button class="send" type="submit">Send</button>
              <a href="{{ url_for('new_chat') }}"><button class="new" type="button">New chat</button></a>
            </div>
            <div class="hint">Screenshots show as “📷 Screenshot attached” but are OCR’d for searching.</div>
          </form>
        </div>
      </div>
    </main>
  </div>

  <script>
    const ta = document.getElementById('ta');
    const form = document.getElementById('f');
    const wrap = document.getElementById('wrap');
    const shadow = document.getElementById('shadowText');

    function preparePayloadAndClear() {
      shadow.value = (ta.value || '');
      ta.removeAttribute('name');
      ta.value = '';
      ta.disabled = true;
    }
    ta.addEventListener('keydown', (e)=>{
      if (e.key === 'Enter' && !e.shiftKey){
        e.preventDefault();
        preparePayloadAndClear();
        form.requestSubmit();
      }
    });
    form.addEventListener('submit', ()=>{
      if (ta.hasAttribute('name')) { preparePayloadAndClear(); }
      sessionStorage.setItem('autoscroll', '1');
    });
    window.addEventListener('load', ()=>{
      const should = sessionStorage.getItem('autoscroll');
      wrap.scrollTop = wrap.scrollHeight + 1000;
      if (should) sessionStorage.removeItem('autoscroll');
      ta.disabled = false;
      if (!ta.hasAttribute('name')) ta.setAttribute('name','text');
      ta.focus();
    });

    const chatsEl = document.getElementById('chats');
    chatsEl.addEventListener('contextmenu', (e) => {
      const a = e.target.closest('a.chat-link');
      if (!a) return;
      e.preventDefault();
      const chatId = a.dataset.id || (new URL(a.href, window.location.href)).pathname.split('/').pop();
      if (confirm('Delete this chat?')) {
        const tmp = document.createElement('form');
        tmp.method = 'POST';
        tmp.action = `{{ url_for('delete_chat', chat_id='__ID__') }}`.replace('__ID__', chatId);
        document.body.appendChild(tmp);
        tmp.submit();
      }
    });

    (function(){
      const btn = document.getElementById('micBtn');
      if (!btn) return;
      function supported(){ return ('webkitSpeechRecognition' in window) || ('SpeechRecognition' in window); }
      if (!supported()){ btn.disabled = true; btn.textContent = '🎤 Unavailable'; return; }
      const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
      const rec = new SpeechRec(); rec.continuous = true; rec.interimResults = true; rec.lang = navigator.language || 'en-US';
      let listening = false;
      rec.onstart = ()=>{ listening = true; btn.classList.add('recording'); btn.textContent = '⏺ Listening…'; };
      rec.onend   = ()=>{ listening = false; btn.classList.remove('recording'); btn.textContent = '🎤 Start'; };
      rec.onerror = ()=>{ listening = false; btn.classList.remove('recording'); btn.textContent = '🎤 Start'; };
      rec.onresult = (e)=>{
        let finalText = '';
        for (let i = e.resultIndex; i < e.results.length; i++){
          const r = e.results[i];
          if (r.isFinal) finalText += r[0].transcript + ' ';
        }
        if (finalText){ ta.value = (ta.value ? ta.value + ' ' : '') + finalText.trim(); }
      };
      btn.addEventListener('click', ()=>{ if (!listening) rec.start(); else rec.stop(); });
    })();
  </script>
</body>
</html>
"""

# ======================
# Confluence helpers
# ======================
def api_auth():
    return (CONFLUENCE_USER_EMAIL, CONFLUENCE_API_TOKEN)

def build_web_link_from_links(links: dict) -> str | None:
    if not links: return None
    webui = links.get("webui")
    if webui:
        if webui.startswith("/spaces/"): return f"{CONFLUENCE_BASE_URL}/wiki{webui}"
        return f"{CONFLUENCE_BASE_URL}{webui}"
    return None

def extract_page_id_from_url(url: str) -> str | None:
    m = re.search(r"/pages/(\d+)", url)
    return m.group(1) if m else None

def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html or "", "html.parser")
    for br in soup.find_all(["br"]): br.replace_with("\n")
    for li in soup.find_all("li"): li.insert_before("\n- ")
    for h in soup.find_all(["h1","h2","h3","h4","h5","h6"]):
        heading = h.get_text(strip=True)
        if heading:
            h.insert_before("\n\n" + heading + "\n")
            h.decompose()
    for p in soup.find_all(["p"]): p.insert_before("\n")
    text = soup.get_text()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def get_page_content(page_id: str) -> dict | None:
    url = f"{CONFLUENCE_BASE_URL}/wiki/rest/api/content/{page_id}"
    params = {"expand": "body.storage,body.view,_links,title"}
    try:
        r = requests.get(url, params=params, auth=api_auth(),
                         headers={"Accept":"application/json"}, timeout=25)
        if r.status_code != 200:
            return {"error": f"HTTP {r.status_code}"}
        data = r.json()
        html_storage = ((data.get("body", {}) or {}).get("storage", {}) or {}).get("value") or ""
        html_view = ((data.get("body", {}) or {}).get("view", {}) or {}).get("value") or ""
        html = html_storage or html_view or ""
        title = data.get("title", "Untitled")
        url_web = build_web_link_from_links(data.get("_links", {}))
        text = html_to_text(html)
        return {"text": text, "html": html, "title": title, "url": url_web}
    except Exception as e:
        print("[Confluence] Exception:", e)
        return {"error": str(e)}

def list_pages_in_space(space_key: str, max_pages: int) -> list[dict]:
    results, start = [], 0
    url = f"{CONFLUENCE_BASE_URL}/wiki/rest/api/content/search"
    cql = f"space = {space_key} and type = page and status = current"
    try:
        while True:
            params = {"cql": cql, "limit": "100", "start": str(start)}
            r = requests.get(url, params=params, auth=api_auth(),
                             headers={"Accept":"application/json"}, timeout=25)
            if r.status_code != 200: break
            data = r.json()
            if not data.get("results"): break
            for row in data["results"]:
                pid = row.get("id")
                title = row.get("title","Untitled")
                links = row.get("_links") or {}
                web = build_web_link_from_links(links)
                results.append({"page_id": pid, "title": title, "url": web})
                if len(results) >= max_pages: return results
            start += len(data["results"])
    except Exception as e:
        print("[CQL] Exception:", e)
    return results

def list_descendant_pages(parent_page_id: str, max_pages: int) -> list[dict]:
    results, start = [], 0
    url = f"{CONFLUENCE_BASE_URL}/wiki/rest/api/search"
    cql = f"ancestor = {parent_page_id} AND type = page AND status = current"
    try:
        while True:
            params = {"cql": cql, "limit": "50", "start": str(start)}
            r = requests.get(url, params=params, auth=api_auth(),
                             headers={"Accept":"application/json"}, timeout=25)
            if r.status_code != 200: break
            data = r.json()
            batch = 0
            for row in data.get("results", []):
                content = row.get("content") or {}
                pid = content.get("id") or row.get("id")
                title = row.get("title") or content.get("title") or "Untitled"
                links = row.get("_links") or content.get("_links") or {}
                web = build_web_link_from_links(links)
                if pid:
                    results.append({"page_id": pid, "title": title, "url": web})
                    batch += 1
            if not data.get("_links", {}).get("next") or batch == 0: break
            start += 50
            if len(results) >= max_pages: break
    except Exception as e:
        print("[CQL] Exception:", e)
    return results

def list_all_pages(max_pages: int) -> list[dict]:
    results, start = [], 0
    url = f"{CONFLUENCE_BASE_URL}/wiki/rest/api/search"
    cql = "type = page AND status = current"
    try:
        while True:
            params = {"cql": cql, "limit": "50", "start": str(start)}
            r = requests.get(url, params=params, auth=api_auth(),
                             headers={"Accept":"application/json"}, timeout=25)
            if r.status_code != 200: break
            data = r.json()
            batch = 0
            for row in data.get("results", []):
                content = row.get("content") or {}
                pid = content.get("id") or row.get("id")
                title = row.get("title") or content.get("title") or "Untitled"
                links = row.get("_links") or content.get("_links") or {}
                web = build_web_link_from_links(links)
                if pid:
                    results.append({"page_id": pid, "title": title, "url": web})
                    batch += 1
                if len(results) >= max_pages: return results
            if not data.get("_links", {}).get("next") or batch == 0: break
            start += 50
    except Exception as e:
        print("[CQL] Exception (all pages):", e)
    return results

# ======================
# OCR
# ======================
def ocr_image(file_storage) -> str:
    try:
        raw = file_storage.read()
        if not raw:
            print("OCR error: empty upload")
            return ""
        img = Image.open(io.BytesIO(raw))
        if img.mode not in ("L", "RGB"):
            img = img.convert("RGB")
        gray = img.convert("L")
        bw = gray.point(lambda x: 255 if x > 180 else 0, mode="1")
        config = "--oem 3 --psm 6"
        text = pytesseract.image_to_string(bw, lang="eng", config=config).strip()
        if len(text) < 3:
            text = pytesseract.image_to_string(gray, lang="eng", config=config).strip()
        if text:
            print("OCR ok, first 120 chars:", text[:120].replace("\n", " "))
        else:
            print("OCR warning: empty result")
        return text
    except Exception as e:
        print("OCR error:", e)
        return ""

# ======================
# Tokenization / BM25
# ======================
STOP = {
    "the","and","for","that","with","this","from","into","your","you","are","was","were","have","has","had","but","not",
    "any","can","could","should","would","will","shall","about","also","then","than","when","what","which","how","why",
    "use","used","using","via","into","out","over","under","on","in","of","to","a","an","is","it","be","as","at","by",
    "or","if","we","our","their","they","them","he","she","his","her","its"
}
def tokenize(s: str) -> list:
    s = s.lower()
    s = re.sub(r"[^\w\s-]", " ", s)
    return [w for w in s.split() if len(w) > 2 and w not in STOP]

class BM25:
    def __init__(self, docs_tokens: list, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs_tokens = docs_tokens
        self.N = len(docs_tokens)
        self.df = Counter()
        self.doc_len = [len(d) for d in docs_tokens]
        for d in docs_tokens: self.df.update(set(d))
        self.avgdl = (sum(self.doc_len) / self.N) if self.N else 0
    def idf(self, term: str) -> float:
        n_qi = self.df.get(term, 0)
        return math.log(1 + (self.N - n_qi + 0.5) / (n_qi + 0.5)) if self.N else 0.0
    def score(self, q_tokens: list, doc_idx: int) -> float:
        score, dl = 0.0, (self.doc_len[doc_idx] or 1)
        tf = Counter(self.docs_tokens[doc_idx])
        for t in q_tokens:
            if t not in tf: continue
            idf = self.idf(t); f = tf[t]
            denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
            score += idf * ((f * (self.k1 + 1)) / denom)
        return score

# ======================
# Table parsing (Error Catalog)
# ======================
HEADER_ALIASES = {
    "error": {"error","errors","message","error message","problem"},
    "cause": {"cause","causes","root cause"},
    "remedy": {"remedy","solution","fix","resolution"},
    "suggestions": {"suggestion","suggestions","note","notes","hint","hints","tip","tips"},
}
def _clean_cell_text(el) -> str:
    return " ".join(el.get_text(separator=" ").split()).strip()
def _normalize_header(h: str) -> str:
    h = (h or "").strip().lower()
    for key, aliases in HEADER_ALIASES.items():
        if h in aliases: return key
    if "error" in h or "message" in h or "problem" in h: return "error"
    if "cause" in h: return "cause"
    if "remedy" in h or "solution" in h or "fix" in h or "resolution" in h: return "remedy"
    if "suggest" in h or "note" in h or "tip" in h: return "suggestions"
    return h or "col"
def extract_table_rows_from_html(html: str) -> list[dict]:
    out, soup = [], BeautifulSoup(html or "", "html.parser")
    for table in soup.find_all("table"):
        headers = []
        thead = table.find("thead")
        if thead:
            ths = thead.find_all("th")
            if ths: headers = [_normalize_header(_clean_cell_text(th)) for th in ths]
        if not headers:
            first_tr = table.find("tr")
            if first_tr:
                headers = [_normalize_header(_clean_cell_text(th)) for th in first_tr.find_all(["th","td"])]
        for tr in table.find_all("tr"):
            tds = tr.find_all("td")
            if not tds: continue
            cells = {headers[i] if i < len(headers) else f"col{i}": _clean_cell_text(td) for i, td in enumerate(tds)}
            norm = {
                "error": cells.get("error","").strip(),
                "cause": cells.get("cause","").strip(),
                "remedy": cells.get("remedy","").strip(),
                "suggestions": cells.get("suggestions","").strip(),
                "raw": cells
            }
            if len(norm["error"]) >= 3:
                norm["text"] = " ".join(v for k,v in norm.items() if k in ("error","cause","remedy","suggestions") and v)
                out.append(norm)
    return out

# ======================
# Answer formatting
# ======================
def post_process_answer(raw: str) -> str:
    raw = re.split(r'\bSource:', raw)[0]
    raw = re.sub(r'[*_`]', '', raw)
    raw = re.sub(r' +', ' ', raw)
    raw = re.sub(r'\n\s*\n', '\n', raw)
    out = "\n".join([ln.strip() for ln in raw.splitlines() if ln.strip()])
    if MAX_ANSWER_CHARS is not None and MAX_ANSWER_CHARS >= 0:
        return (out[:MAX_ANSWER_CHARS-1] + "…") if len(out) > MAX_ANSWER_CHARS else out
    return out

def format_row_better(row: dict) -> str:
    steps = []
    if row.get("remedy"):
        steps = [x.strip(" -•") for x in re.split(r"[.;]\s+|\n+", row["remedy"]) if x.strip()]
    notes = []
    if row.get("suggestions"):
        notes = [x.strip(" -•") for x in re.split(r"[.;]\s+|\n+", row["suggestions"]) if x.strip()]
    parts = []
    if row.get("error"): parts.append(f"Resolution for: {row['error']}")
    if row.get("cause"):
        c = row["cause"]
        if len(c) <= 160:
            parts.append(f"Why: {c}")
    if steps:
        parts.append("\nDo this:")
        for i, st in enumerate(steps[:min(PINPOINT_MAX_STEPS, MAX_STEP_LINES)] , 1):
            parts.append(f"{i}. {st}")
    if notes:
        parts.append("\nNotes:")
        for t in notes[:6]:
            parts.append(f"- {t}")
    return "\n".join(parts).strip()

def synthesize_from_row(row: dict, query: str) -> str:
    if not (USE_LLM and OPENAI_API_KEY): return ""
    kb_text = []
    if row.get("error"):       kb_text.append(f"Error: {row['error']}")
    if row.get("cause"):       kb_text.append(f"Cause: {row['cause']}")
    if row.get("remedy"):      kb_text.append(f"Remedy: {row['remedy']}")
    if row.get("suggestions"): kb_text.append(f"Suggestions: {row['suggestions']}")
    kb = "\n".join(kb_text)[:3200]
    prompt = f"""Use ONLY the KB excerpt to answer.

<KB>
{kb}
</KB>

User: {query}

Return the most specific single path:
- 1-line summary naming the exact thing (UI labels, error text).
- If a cause is present, one short "Why:" line.
- Numbered steps to fix (max {PINPOINT_MAX_STEPS}) with exact menu paths/fields/values if present.
- "Notes:" only if the KB provides constraints/caveats.
- No alternatives, no outside facts."""
    return llm_chat(prompt, temperature=0.2)

def synthesize_from_chunks(query: str, chunks: list[str]) -> str:
    if not (USE_LLM and OPENAI_API_KEY): return ""
    joined = "\n\n---\n\n".join(chunks)[:6000]
    prompt = f"""Using ONLY <KB>, give one highly-specific path.

<KB>
{joined}
</KB>
Question: {query}

Answer:
- one-line summary using exact KB labels/paths
- numbered steps (max {PINPOINT_MAX_STEPS})
- optional "Notes:" if KB includes caveats
No alternatives."""
    return llm_chat(prompt, temperature=0.25)

# ======================
# KB storage/build
# ======================
KB_PAGES = []            # [{title,url,text,html}]
KB_SECTIONS = []         # [{title,url,text,tokens}]
KB_SECTION_TOKENS = []
SECTION_BM25 = None

KB_ROWS = []             # [{title,url,row(dict)}]
ROW_TOKENS = []
ROW_BM25 = None

def sectionize(text: str) -> list[tuple[str, int, int]]:
    if not text: return []
    paras = [p for p in text.split("\n\n") if p.strip()]
    sections, offset, chunk, start = [], 0, [], 0
    for p in paras:
        is_heading = bool(re.match(r"^[A-Z][A-Z0-9 \-_/()]+$|.*:\s*$", p.strip()))
        if is_heading and chunk:
            sec = "\n\n".join(chunk).strip()
            sections.append((sec, start, start + len(sec)))
            chunk = [p]; start = offset
        else:
            if not chunk: start = offset
            chunk.append(p)
        offset += len(p) + 2
    if chunk:
        sec = "\n\n".join(chunk).strip()
        sections.append((sec, start, start + len(sec)))
    return sections

def preload_knowledge():
    global KB_PAGES, KB_SECTIONS, KB_SECTION_TOKENS, SECTION_BM25
    global KB_ROWS, ROW_TOKENS, ROW_BM25

    KB_PAGES, KB_SECTIONS, KB_SECTION_TOKENS = [], [], []
    KB_ROWS, ROW_TOKENS = [], []

    refs = []
    if SPACE_KEYS:
        for sk in SPACE_KEYS:
            refs.extend(list_pages_in_space(sk, max_pages=MAX_PAGES_PER_SPACE))
    elif ERROR_CATALOG_PARENT_URL:
        pid = extract_page_id_from_url(ERROR_CATALOG_PARENT_URL)
        if pid: refs.extend(list_descendant_pages(pid, max_pages=MAX_PAGES_PER_SPACE))
    elif FULL_SITE_CRAWL:
        refs.extend(list_all_pages(max_pages=MAX_PAGES_TOTAL))
    else:
        print("[KB] No crawl mode configured (SPACE_KEYS, ERROR_CATALOG_PARENT_URL or FULL_SITE_CRAWL).")

    seen = set()
    for ref in refs:
        pid = ref["page_id"]
        if pid in seen: continue
        seen.add(pid)
        page = get_page_content(pid)
        if not page or page.get("error"): continue
        text = (page.get("text") or "").strip()
        html = (page.get("html") or "").strip()
        if not text and not html: continue
        KB_PAGES.append({
            "title": page.get("title","Untitled"),
            "url": page.get("url") or ref.get("url",""),
            "text": text, "html": html
        })

    for p in KB_PAGES:
        secs = sectionize(p["text"]) or [(p["text"][:1200], 0, min(1200, len(p["text"])))]
        for (stext, _, _) in secs:
            toks = tokenize(stext)
            KB_SECTIONS.append({"title": p["title"], "url": p["url"], "text": stext, "tokens": toks})
            KB_SECTION_TOKENS.append(toks)
    SECTION_BM25 = BM25(KB_SECTION_TOKENS) if KB_SECTIONS else None

    for p in KB_PAGES:
        rows = extract_table_rows_from_html(p["html"])
        for r in rows:
            KB_ROWS.append({"title": p["title"], "url": p["url"], "row": r})
            ROW_TOKENS.append(tokenize(" ".join([r.get("error",""), r.get("cause",""), r.get("remedy",""), r.get("suggestions","")])))

    ROW_BM25 = BM25(ROW_TOKENS) if KB_ROWS else None
    print(f"[KB] pages={len(KB_PAGES)}, sections={len(KB_SECTIONS)}, rows={len(KB_ROWS)}")

def _build_bg():
    try: preload_knowledge()
    except Exception as e: print("[KB] preload error:", e)
threading.Thread(target=_build_bg, daemon=True).start()

# ======================
# Retrieval (row-first & Error Catalog biased)
# ======================
UI_VISIBILITY_RE = re.compile(r'\b(option|options|menu|button|tab|checkbox|not\s+(present|visible|found)|missing|disabled|hidden)\b', re.I)
GREETING_PAT = re.compile(r"^\s*(hi|hello|hey|good (morning|afternoon|evening)|how are you)\b", re.I)

CREATOR_PAT  = re.compile(
    r"""(?ix)
    \bwho\s+(?:is\s+)?(?:your|ur|the|this)?\s*
        (?:creator|developer|owner|author|maker|maintainer|founder)\b
    |
    \bwho\s+(?:made|build|built|create|created|develop|developed)\s+(?:you|u|this|the\s*bot)\b
    |
    \b(?:you|u|this\s*bot)\s*(?:was|were)?\s*(?:made|built|created|developed)\s*by\b
    """,
    re.I
)
def is_creator_query(text: str) -> bool:
    if not text: return False
    t = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    if "who" not in t: return False
    pronouns = (" you ", " u ", " this ", " bot ")
    nouns    = (" creator ", " developer ", " owner ", " author ", " maker ", " maintainer ", " founder ")
    verbs    = (" create ", " created ", " build ", " built ", " make ", " made ", " develop ", " developed ")
    has_pronoun = any(p in f" {t} " for p in pronouns)
    has_noun    = any(n in f" {t} " for n in nouns)
    has_verb    = any(v in f" {t} " for v in verbs)
    return (has_pronoun and (has_noun or has_verb))

ERROR_CODE_PAT = re.compile(
    r"(?:\bORA-\d{4,5}\b)|(?:\bSQLSTATE\s*[0-9A-Z]{5}\b)|(?:\bHTTP\s*[1-5]\d{2}\b)|(?:\b[A-Z]{2,5}-\d{3,5}\b)",
    re.I
)

def score_row_against_query(row: dict, query: str, page_title: str = "") -> float:
    q = (query or "").strip()
    ql = q.lower()
    err = (row.get("error") or "")
    txt = (row.get("text") or "")
    remedy = (row.get("remedy") or "")

    hard = 0.0
    if ERROR_ROW_HARDMATCH:
        if err and ql and (ql in err.lower() or err.lower() in ql):
            hard = 1.0
        else:
            m = ERROR_CODE_PAT.search(q)
            if m and (m.group(0).lower() in err.lower() or m.group(0).lower() in txt.lower()):
                hard = 0.95

    p_err = fuzz.partial_ratio(ql, err.lower())/100.0
    t_err = fuzz.token_set_ratio(ql, err.lower())/100.0
    p_txt = fuzz.partial_ratio(ql, txt.lower())/100.0
    t_txt = fuzz.token_set_ratio(ql, txt.lower())/100.0

    base = max(hard, 0.52*p_err + 0.20*t_err + 0.18*p_txt + 0.10*t_txt)

    if remedy:
        base += 0.08
        if len(remedy) > 120:
            base += 0.03

    if page_title:
        tl = page_title.lower()
        if any(h in tl for h in ERROR_TITLE_HINTS):
            base += ERROR_PAGE_BIAS

    if UI_VISIBILITY_RE.search(ql) and (UI_VISIBILITY_RE.search(txt.lower()) or UI_VISIBILITY_RE.search(err.lower())):
        base += 0.08

    return max(0.0, min(1.0, base))

def get_row_candidates(query: str, top_k: int = 12):
    q = (query or "")
    cands = []
    for i, item in enumerate(KB_ROWS):
        page_title = item.get("title") or ""
        sc = score_row_against_query(item["row"], q, page_title)
        if ROW_BM25 and i < len(ROW_TOKENS):
            sc = 0.90*sc + 0.10*(ROW_BM25.score(tokenize(q), i) or 0.0)
        if page_title:
            sc += 0.05 * (fuzz.partial_ratio(q.lower(), page_title.lower())/100.0)
        cands.append((sc, i, item))
    cands.sort(key=lambda x: x[0], reverse=True)
    return cands[:top_k]

def answer_sections(query: str):
    q_tokens = tokenize(query)
    ql = query.lower()
    scored = []
    for idx, sec in enumerate(KB_SECTIONS):
        bm = SECTION_BM25.score(q_tokens, idx) if SECTION_BM25 else 0.0
        title_sim = fuzz.partial_ratio(ql, sec["title"].lower()) / 100.0
        sc = 0.85*bm + 0.15*title_sim
        tl = (sec["title"] or "").lower()
        if any(h in tl for h in ERROR_TITLE_HINTS):
            sc += ERROR_PAGE_BIAS * 0.7
        scored.append((sc, idx))
    scored.sort(key=lambda x: x[0], reverse=True)

    if not scored and KB_PAGES:
        p0 = KB_PAGES[0]
        chunks = [p0["text"][:800]]
        sources = [{"title": p0["title"], "url": p0["url"]}]
    else:
        top = [idx for _, idx in scored[:MAX_SECTIONS]]
        chunks = [KB_SECTIONS[i]["text"][:700].strip() for i in top]
        seen, sources = set(), []
        for i in top:
            t, u = KB_SECTIONS[i]["title"], KB_SECTIONS[i]["url"]
            if t not in seen and len(sources) < MAX_SOURCES_PER_ANSWER:
                sources.append({"title": t, "url": u}); seen.add(t)

    llm_txt = synthesize_from_chunks(query, chunks)
    if llm_txt:
        return post_process_answer(llm_txt), sources

    sentences = []
    for ch in chunks:
        sentences.extend([s.strip() for s in re.split(r"(?<=[.!?])\s+", ch.replace("\n"," ")) if s.strip()])
    qs = set(q_tokens)
    ranked = []
    for s in sentences:
        stoks = tokenize(s)
        overlap = len(qs & set(stoks))
        density = overlap / (len(stoks) + 1e-6)
        ranked.append((overlap + 0.75*density, s))
    ranked.sort(key=lambda x: x[0], reverse=True)  # fixed bug
    top_sents = [s for _, s in ranked[:1]]
    out = []
    if top_sents: out.append(top_sents[0])
    return post_process_answer("\n".join(out).strip()), sources

def answer_from_kb(query: str):
    q = re.sub(r"[^\w\s:/.-]", " ", (query or "")).strip()
    candidates = get_row_candidates(q, top_k=12)
    if candidates and candidates[0][0] >= ROW_SCORE_MIN:
        _, _, item0 = candidates[0]
        txt = synthesize_from_row(item0["row"], q) or format_row_better(item0["row"])
        answer = post_process_answer(txt)
        sources = [{"title": item0["title"], "url": item0["url"]}]
        return True, answer, sources
    if KB_SECTIONS:
        ans, sources = answer_sections(q)
        if ans.strip():
            return True, ans, sources
    return False, "", []

def route_intent(text: str) -> str:
    t = (text or "").strip().lower()
    if GREETING_PAT.search(t): return "chitchat"
    if CREATOR_PAT.search(t) or is_creator_query(t): return "creator"
    # Confluence mention ⇒ KB route
    if "confluence" in t or "atlassian.net" in t or re.search(r"https?://[^ \t\r\n]*atlassian\.net", t):
        return "kb"
    if re.search(r"\b(err(or)?|exception|failed|permission|denied|http\s*[45]\d{2}|ora-\d{4,5}|sqlstate|remedy|fix|solution|root cause|collate|scenario|capacity|generate)\b", t):
        return "kb"
    if UI_VISIBILITY_RE.search(t): return "kb"
    return "general"

# ======================
# Store (per-user isolation)
# ======================
_store_lock = threading.Lock()
def _load_store():
    if not os.path.exists(STORE_PATH):
        return {"users": {}}
    try:
        with open(STORE_PATH,"r",encoding="utf-8") as f:
            data = json.load(f)
    except:
        return {"users": {}}
    if "chats" in data and "users" not in data:
        data = {"users": {"public": {"chats": data["chats"]}}}
    if "users" not in data:
        data["users"] = {}
    return data

def _save_store(data):
    with _store_lock:
        tmp = STORE_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
        os.replace(tmp, STORE_PATH)

def _get_user_bucket(store, uid):
    users = store.setdefault("users", {})
    return users.setdefault(uid, {"chats": []})

def _make_title_from_text(t: str) -> str:
    t = (t or "").strip()
    if not t: return "New chat"
    t = re.sub(r"\s+"," ", t)
    return (t[:48] + "…") if len(t) > 48 else t

def _find_chat(store, uid, chat_id):
    ub = _get_user_bucket(store, uid)
    for c in ub["chats"]:
        if c["id"] == chat_id:
            return c
    return None

def _create_chat(store, uid):
    cid = uuid.uuid4().hex[:10]
    chat = {
        "id": cid,
        "title":"New chat",
        "created": datetime.utcnow().isoformat()+"Z",
        "messages":[],
        "memory": { "facts": {}, "notes": [] }
    }
    _get_user_bucket(store, uid)["chats"].insert(0, chat)
    _save_store(store)
    return chat

def _append_msg(store, uid, chat_id, role, text, sources=None):
    chat = _find_chat(store, uid, chat_id)
    if not chat: return
    chat["messages"].append({"role":role, "text":text, "sources":(sources or [])})
    if len(chat["messages"]) > MAX_HISTORY_MSGS:
        chat["messages"] = chat["messages"][-MAX_HISTORY_MSGS:]
    _save_store(store)

def _set_title_if_new(store, uid, chat_id, first_text):
    chat = _find_chat(store, uid, chat_id)
    if not chat: return
    if chat["title"] == "New chat" and first_text.strip():
        chat["title"] = _make_title_from_text(first_text)
        _save_store(store)

# ======================
# Memory & small helpers
# ======================
MEM_KEY_ALIASES = {
    "name": {"name","full name"},
    "email": {"email","mail","e-mail"},
    "phone": {"phone","mobile","cell","phone number"},
    "company": {"company","org","organization"},
    "project": {"project","proj"},
    "timezone": {"timezone","tz","time zone"},
}
NAME_PAT   = re.compile(r"\b(my\s+name\s+is|i\s*am|i'm)\s+([A-Z][\w'.-]+(?:\s+[A-Z][\w'.-]+)*)", re.I)
EMAIL_PAT  = re.compile(r"\b(my\s+email\s+is|email\s*[:=])\s*([A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,})", re.I)
PHONE_PAT  = re.compile(r"\b(my\s+(?:phone|mobile|cell)\s*(?:number)?\s*(?:is)?|phone\s*[:=])\s*([+()\d][\d\s()+-]{6,})", re.I)
CALLME_PAT = re.compile(r"\b(call\s+me)\s+([A-Z][\w'.-]+(?:\s+[A-Z][\w'.-]+)*)", re.I)
KV_PAT     = re.compile(r"\b(remember|save)\s+(?:that\s+)?(.+)", re.I)
FORGET_PAT = re.compile(r"\b(forget|delete|remove)\s+(my\s+)?(name|email|phone|company|project|timezone)\b", re.I)
COLON_PAT  = re.compile(r"\b([A-Za-z][A-Za-z _-]{1,24})\s*[:=]\s*(.+)")

def _mem_set(store, chat, key, value):
    key = key.strip().lower()
    chat.setdefault("memory", {}).setdefault("facts", {})[key] = value.strip()
def _mem_get(chat, key): return chat.get("memory", {}).get("facts", {}).get(key)
def _mem_all(chat): return chat.get("memory", {}).get("facts", {})
def _mem_forget(chat, key):
    mem = chat.get("memory", {}).get("facts", {})
    if key in mem: del mem[key]
def _mem_add_note(chat, text):
    chat.setdefault("memory", {}).setdefault("notes", []).append({"text": text, "when": datetime.utcnow().isoformat()+"Z"})

def extract_and_update_memory(chat, raw_text):
    if not raw_text: return []
    updates = []
    m = NAME_PAT.search(raw_text) or CALLME_PAT.search(raw_text)
    if m: _mem_set(None, chat, "name", m.group(2).strip()); updates.append(f"name → {m.group(2).strip()}")
    m = EMAIL_PAT.search(raw_text)
    if m: _mem_set(None, chat, "email", m.group(2).strip()); updates.append(f"email → {m.group(2).strip()}")
    m = PHONE_PAT.search(raw_text)
    if m: _mem_set(None, chat, "phone", m.group(2).strip()); updates.append(f"phone → {m.group(2).strip()}")
    mf = FORGET_PAT.search(raw_text)
    if mf: _mem_forget(chat, mf.group(3).lower()); updates.append(f"forgot {mf.group(3).lower()}")
    mkv = KV_PAT.search(raw_text)
    if mkv: _mem_add_note(chat, mkv.group(2).strip()); updates.append(f"noted: {mkv.group(2).strip()}")
    for (k,v) in COLON_PAT.findall(raw_text):
        k_norm = k.strip().lower()
        matched = False
        for canon, aliases in MEM_KEY_ALIASES.items():
            if k_norm == canon or k_norm in aliases:
                _mem_set(None, chat, canon, v.strip()); updates.append(f"{canon} → {v.strip()}"); matched = True; break
        if not matched:
            _mem_set(None, chat, k_norm, v.strip()); updates.append(f"{k_norm} → {v.strip()}")
    return updates

def memory_answer(chat, user_text):
    t = (user_text or "").lower()
    if re.search(r"\b(what\s+do\s+you\s+remember|what\s+have\s+you\s+saved|my\s+details)\b", t):
        facts = _mem_all(chat)
        if not facts: return "I don’t have any saved facts yet in this chat."
        return "Here’s what I’ve saved for this chat:\n" + "\n".join(f"- {k}: {v}" for k,v in facts.items())
    checks = {
        "name":   r"\b(what('| i)s\s+my\s+name|who\s+am\s+i)\b",
        "email":  r"\b(what('| i)s\s+my\s+email)\b",
        "phone":  r"\b(what('| i)s\s+my\s+(?:phone|mobile|cell)(?:\s+number)?)\b",
        "company":r"\b(what('| i)s\s+my\s+company)\b",
        "project":r"\b(what('| i)s\s+my\s+project)\b",
        "timezone":r"\b(what('| i)s\s+my\s+time\s*zone|timezone)\b",
    }
    for key, pat in checks.items():
        if re.search(pat, t):
            val = _mem_get(chat, key)
            return f"Your {key} is {val}." if val else f"I haven’t saved your {key} in this chat yet."
    return None

def handle_history_question(chat: dict, text: str) -> str | None:
    t = (text or "").lower().strip()
    user_msgs = [m for m in chat.get("messages", []) if m.get("role") == "user" and (m.get("text") or "").strip()]
    if not user_msgs: return None
    if re.search(r"\bfirst (question|msg|message)\b", t):
        return f'Your first message was: "{user_msgs[0]["text"]}"'
    if re.search(r"\b(last|previous) (question|msg|message)\b", t):
        if len(user_msgs) >= 2:
            return f'Your previous message was: "{user_msgs[-2]["text"]}"'
        else:
            return "There isn’t a previous message yet."
    if re.search(r"\bwhat did i ask\b", t):
        return f'Your last message was: "{user_msgs[-1]["text"]}"'
    return None

# ======================
# Follow-up resolution
# ======================
FOLLOWUP_HINTS = re.compile(r"\b(this|that|it|they|those|above|previous|earlier|same|approach|method|solution|steps|error|issue|problem|fix)\b", re.I)
def build_context_augmented_query(chat: dict, user_text: str, ocr_text: str) -> str:
    combined = " ".join([x for x in [user_text, ocr_text] if x]).strip()
    if not combined: return combined
    if FOLLOWUP_HINTS.search(combined) and chat.get("messages"):
        history = chat["messages"][-(FOLLOWUP_CONTEXT_TURNS*2):]
        last_bot = next((m["text"] for m in reversed(history) if m["role"]=="bot" and m.get("text")), "")
        last_user = next((m["text"] for m in reversed(history) if m["role"]=="user" and m.get("text")), "")
        context = "\n\n--- Context ---\n"
        if last_user: context += f"Last user question: {last_user}\n"
        if last_bot:  context += f"Last assistant answer: {last_bot}\n"
        return f"{combined}\n{context}"
    return combined

def general_chat_response(user_text: str) -> str:
    t = (user_text or "").lower().strip()
    if re.match(r"^\s*(hi|hello|hey)\b", t): return "Hello! How can I assist you today?"
    if "how are you" in t: return "I’m doing well — thanks for asking! What can I help you solve?"
    return "Happy to help! Share your error or question — I’ll search the KB first and fill gaps with general help."

def llm_with_history(chat: dict, user_text: str) -> str:
    if not (USE_LLM and OPENAI_API_KEY):
        return general_chat_response(user_text)
    msgs = []
    history = chat["messages"][-(FOLLOWUP_CONTEXT_TURNS*2):]
    for m in history:
        role = "assistant" if m["role"]=="bot" else "user"
        content = m["text"]
        if content: msgs.append({"role": role, "content": content})
    msgs.append({"role":"user","content": user_text})
    out = llm_chat_messages(msgs, temperature=0.3)
    return out or general_chat_response(user_text)

# ======================
# Routes
# ======================
@app.route("/")
@login_required
def root():
    uid = session["uid"]
    store = _load_store()
    ub = _get_user_bucket(store, uid)
    if not ub["chats"]:
        chat = _create_chat(store, uid)
    else:
        chat = ub["chats"][0]
    return redirect(url_for("chat", chat_id=chat["id"]))

@app.route("/new", methods=["POST","GET"])
@login_required
def new_chat():
    uid = session["uid"]
    store = _load_store()
    chat = _create_chat(store, uid)
    return redirect(url_for("chat", chat_id=chat["id"]))

@app.route("/delete_chat/<chat_id>", methods=["POST"])
@login_required
def delete_chat(chat_id):
    uid = session["uid"]
    store = _load_store()
    ub = _get_user_bucket(store, uid)
    idx = next((i for i, c in enumerate(ub["chats"]) if c.get("id") == chat_id), None)
    if idx is None:
        if ub["chats"]:
            return redirect(url_for("chat", chat_id=ub["chats"][0]["id"]))
        new_chat = _create_chat(store, uid)
        return redirect(url_for("chat", chat_id=new_chat["id"]))
    ub["chats"].pop(idx)
    _save_store(store)
    if ub["chats"]:
        next_id = ub["chats"][0]["id"]
    else:
        new_chat = _create_chat(store, uid)
        next_id = new_chat["id"]
    return redirect(url_for("chat", chat_id=next_id))

@app.route("/chat/<chat_id>", methods=["GET","POST"])
@login_required
def chat(chat_id):
    uid = session["uid"]
    store = _load_store()
    chat = _find_chat(store, uid, chat_id)
    if not chat: abort(404)
    loading = False

    if request.method == "POST":
        user_text_raw = (request.form.get("text") or "")
        user_text = user_text_raw.strip()
        img = request.files.get("image")
        ocr_txt = ""
        if img and img.filename:
            ocr_txt = ocr_image(img)

        attach_note = "📷 Screenshot attached" if (img and img.filename) else ""
        display_user = " ".join([x for x in [user_text, attach_note] if x]).strip() or attach_note or "…"

        _append_msg(store, uid, chat_id, "user", display_user, [])
        _set_title_if_new(store, uid, chat_id, user_text or attach_note)
        loading = True

        q_for_reasoning = build_context_augmented_query(chat, user_text, ocr_txt)

        # EARLY: deterministic creator response (catches variants & 'openai' claims)
        lowered = (user_text or "").lower()
        if (CREATOR_PAT.search(lowered) or is_creator_query(user_text)
            or ("openai" in lowered and ("develop" in lowered or "made" in lowered or "created" in lowered))):
            _append_msg(store, uid, chat_id, "bot", f"I was created by {CREATOR_NAME}.", [])
            return redirect(url_for("chat", chat_id=chat_id))

        updates = extract_and_update_memory(chat, " ".join([user_text, ocr_txt]))
        _save_store(store)

        mem_ans = memory_answer(chat, user_text)
        if mem_ans:
            if updates:
                mem_ans += f"\n(saved: {', '.join(updates)})"
            _append_msg(store, uid, chat_id, "bot", mem_ans, [])
            return redirect(url_for("chat", chat_id=chat_id))

        hist_ans = handle_history_question(chat, user_text)
        if hist_ans:
            if updates:
                hist_ans += f"\n(saved: {', '.join(updates)})"
            _append_msg(store, uid, chat_id, "bot", hist_ans, [])
            return redirect(url_for("chat", chat_id=chat_id))

        intent = route_intent(q_for_reasoning or user_text)
        answered = False
        answer, sources = "", []

        if intent in ("kb",):
            ok, answer, sources = answer_from_kb(q_for_reasoning or user_text)
            answered = ok

            # If user referenced Confluence but no KB answer, show diagnostics (no LLM fallback)
            if (not answered) and (
                "confluence" in lowered or "atlassian.net" in lowered or re.search(r"https?://[^ \t\r\n]*atlassian\.net", lowered)
            ):
                _append_msg(
                    store, uid, chat_id, "bot",
                    "I didn’t find a matching page in the indexed Confluence KB. "
                    "If this page should be included, verify:\n"
                    "• FULL_SITE_CRAWL=1 and the bot user can view those spaces\n"
                    "• CONFLUENCE_BASE_URL points to your site (e.g., https://<org>.atlassian.net)\n"
                    "• Reload at /reload_kb and check logs for [KB] counts",
                    []
                )
                return redirect(url_for("chat", chat_id=chat_id))

        if not answered:
            answer = llm_with_history(chat, q_for_reasoning or user_text or ocr_txt or "(no text)")
            sources = []

        if updates:
            answer = (answer + "\n\n" + f"(saved: {', '.join(updates)})").strip()

        _append_msg(store, uid, chat_id, "bot", answer, sources)
        return redirect(url_for("chat", chat_id=chat_id))

    chats = []
    ub = _get_user_bucket(store, uid)
    for c in ub["chats"]:
        chats.append({"id": c["id"], "title": c["title"], "when": c["created"].split("T")[0]})

    return render_template_string(HTML,
        chats=chats,
        chat_id=chat_id,
        chat_title=chat["title"],
        messages=chat["messages"],
        loading=loading,
        user_id=uid
    )

@app.route("/reload_kb")
@login_required
def reload_kb():
    threading.Thread(target=preload_knowledge, daemon=True).start()
    return "KB reload started."

# ======================
# RUN
# ======================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5105, debug=True)

“””
AI Analyst Chat — FastAPI + Ollama streaming app
Run: uvicorn main:app –reload –port 8000
“””

import json
import uuid
import os
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator

import ollama
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_NAME   = “gpt-oss:20b”
TEMPERATURE  = 0
MAX_STEPS    = 8
HISTORY_FILE = Path(“conversation_history.json”)

SYSTEM_PROMPT = “”“You are an expert data analyst for a banking analytics platform.
You have access to tools to read skill documentation, query Redshift, and interact with Jira.
Always think step by step before calling tools.
Use ONLY these exact tool names: read_skill, query_redshift, query_redshift_and_comment, get_jira_details, skill_index.
Do NOT invent tool names.”””

app = FastAPI()

# ── Conversation history (local JSON file) ────────────────────────────────────

def load_history() -> list:
if HISTORY_FILE.exists():
try:
return json.loads(HISTORY_FILE.read_text())
except Exception:
return []
return []

def save_history(history: list):
HISTORY_FILE.write_text(json.dumps(history, indent=2, default=str))

def append_to_history(conv_id: str, role: str, content: str, metadata: dict = None):
history = load_history()
history.append({
“id”: str(uuid.uuid4()),
“conv_id”: conv_id,
“role”: role,
“content”: content,
“metadata”: metadata or {},
“timestamp”: datetime.utcnow().isoformat()
})
save_history(history)

def get_conversation(conv_id: str) -> list:
history = load_history()
return [h for h in history if h[“conv_id”] == conv_id]

# ── Stub tools (replace with real implementations) ───────────────────────────

def read_skill(skill_path: str) -> str:
“”“Read a markdown skill file.”””
try:
p = Path(“skills”) / skill_path
if p.exists():
return p.read_text()
return f”Skill file not found: {skill_path}”
except Exception as e:
return f”Error reading skill: {e}”

def query_redshift(sql: str) -> str:
“”“Execute a Redshift SQL query and return results.”””
# Replace with actual Redshift connection
return f”[STUB] Query executed: {sql[:100]}… | Returned 8 rows”

def query_redshift_and_comment(sql: str, comment: str) -> str:
“”“Execute query and post result as Jira comment.”””
result = query_redshift(sql)
return f”{result}\n[STUB] Posted as Jira comment: {comment[:80]}”

def get_jira_details() -> str:
“”“Get current Jira ticket details.”””
return “[STUB] Jira ticket: DSG-1234 — Analyst query for MTU/RPU metrics”

def skill_index() -> str:
“”“List available skill files.”””
p = Path(“skills”)
if p.exists():
files = list(p.rglob(”*.md”))
return “\n”.join(str(f.relative_to(p)) for f in files) or “No skills found”
return “Skills directory not found”

TOOLS_CALLABLE = {
“read_skill”:                 read_skill,
“query_redshift”:             query_redshift,
“query_redshift_and_comment”: query_redshift_and_comment,
“get_jira_details”:           get_jira_details,
“skill_index”:                skill_index,
}

TOOLS_SCHEMA = [
{
“type”: “function”,
“function”: {
“name”: “read_skill”,
“description”: “Read a skill/documentation markdown file”,
“parameters”: {
“type”: “object”,
“properties”: {“skill_path”: {“type”: “string”, “description”: “Relative path to skill file”}},
“required”: [“skill_path”]
}
}
},
{
“type”: “function”,
“function”: {
“name”: “query_redshift”,
“description”: “Execute a Redshift SQL query and return results”,
“parameters”: {
“type”: “object”,
“properties”: {“sql”: {“type”: “string”, “description”: “SQL query to execute”}},
“required”: [“sql”]
}
}
},
{
“type”: “function”,
“function”: {
“name”: “query_redshift_and_comment”,
“description”: “Execute query and post result as a Jira comment”,
“parameters”: {
“type”: “object”,
“properties”: {
“sql”:     {“type”: “string”},
“comment”: {“type”: “string”, “description”: “Comment to post on Jira”}
},
“required”: [“sql”, “comment”]
}
}
},
{
“type”: “function”,
“function”: {
“name”: “get_jira_details”,
“description”: “Get the current Jira ticket details. No input needed.”,
“parameters”: {“type”: “object”, “properties”: {}}
}
},
{
“type”: “function”,
“function”: {
“name”: “skill_index”,
“description”: “List all available skill files in the knowledge base”,
“parameters”: {“type”: “object”, “properties”: {}}
}
},
]

# ── SSE helper ────────────────────────────────────────────────────────────────

def sse(event: str, data: dict) -> str:
return f”event: {event}\ndata: {json.dumps(data)}\n\n”

# ── Core streaming generator ──────────────────────────────────────────────────

async def run_agent_stream(conv_id: str, user_message: str) -> AsyncGenerator[str, None]:
# Build messages from history
messages = [{“role”: “system”, “content”: SYSTEM_PROMPT}]
for h in get_conversation(conv_id):
if h[“role”] in (“user”, “assistant”, “tool”):
messages.append({“role”: h[“role”], “content”: h[“content”]})

```
# Add new user message
messages.append({"role": "user", "content": user_message})
append_to_history(conv_id, "user", user_message)

final_response = ""
all_tool_results = []

for step in range(1, MAX_STEPS + 1):
    yield sse("step_start", {"step": step})

    # ── Stream from Ollama ────────────────────────────────────────────────
    full_thinking = ""
    full_content  = ""
    tool_calls    = []
    in_thinking   = False
    in_content    = False

    stream = ollama.chat(
        model=MODEL_NAME,
        messages=messages,
        tools=TOOLS_SCHEMA,
        think=True,
        stream=True,
        options={"temperature": TEMPERATURE}
    )

    for chunk in stream:
        msg = chunk.get("message", {})

        # Thinking delta
        td = msg.get("thinking", "")
        if td:
            if not in_thinking:
                yield sse("thinking_start", {"step": step})
                in_thinking = True
            full_thinking += td
            yield sse("thinking_delta", {"text": td})

        # Content delta
        cd = msg.get("content", "")
        if cd:
            if in_thinking and not in_content:
                yield sse("thinking_end", {})
                in_thinking = False
            if not in_content:
                yield sse("content_start", {"step": step})
                in_content = True
            full_content += cd
            yield sse("content_delta", {"text": cd})

        # Tool calls
        if msg.get("tool_calls"):
            tool_calls.extend(msg["tool_calls"])

    if in_thinking:
        yield sse("thinking_end", {})
    if in_content:
        yield sse("content_end", {})

    # ── No tool calls → done ─────────────────────────────────────────────
    if not tool_calls:
        final_response = full_content
        break

    # ── Execute tool calls ────────────────────────────────────────────────
    messages.append({
        "role": "assistant",
        "content": full_content,
        "thinking": full_thinking,
        "tool_calls": tool_calls
    })

    for tc in tool_calls:
        fn_name = tc["function"]["name"]
        fn_args = tc["function"]["arguments"]

        yield sse("tool_start", {"step": step, "name": fn_name, "args": fn_args})

        try:
            fn = TOOLS_CALLABLE.get(fn_name)
            if fn and callable(fn):
                result = fn(**fn_args)
            else:
                result = (
                    f"Tool '{fn_name}' not found. "
                    f"Available: {list(TOOLS_CALLABLE.keys())}"
                )
            status = "ok"
        except Exception as e:
            result = f"Error running {fn_name}: {e}"
            status = "error"

        all_tool_results.append({
            "step": step, "tool": fn_name,
            "args": fn_args, "result": result, "status": status
        })

        yield sse("tool_result", {
            "step": step, "name": fn_name,
            "result": str(result)[:500], "status": status
        })

        messages.append({
            "role": "tool",
            "content": json.dumps(result) if not isinstance(result, str) else result
        })

# ── Save final response to history ────────────────────────────────────────
append_to_history(conv_id, "assistant", final_response, {
    "steps": step,
    "tool_calls": all_tool_results
})

# Optionally save results as a downloadable file
result_file = None
if all_tool_results:
    fname = f"results_{conv_id[:8]}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    fpath = Path("outputs") / fname
    fpath.parent.mkdir(exist_ok=True)
    fpath.write_text(json.dumps(all_tool_results, indent=2, default=str))
    result_file = fname

yield sse("final", {
    "response": final_response,
    "steps":    step,
    "file":     result_file
})
yield sse("done", {})
```

# ── API Routes ────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
conv_id:     str
user_message: str

@app.post(”/chat/stream”)
async def chat_stream(req: ChatRequest):
return StreamingResponse(
run_agent_stream(req.conv_id, req.user_message),
media_type=“text/event-stream”,
headers={
“Cache-Control”:               “no-cache”,
“X-Accel-Buffering”:           “no”,
“Access-Control-Allow-Origin”: “*”,
}
)

@app.get(”/history/{conv_id}”)
async def get_history(conv_id: str):
return {“history”: get_conversation(conv_id)}

@app.get(”/history”)
async def list_conversations():
history = load_history()
convs = {}
for h in history:
cid = h[“conv_id”]
if cid not in convs:
convs[cid] = {“conv_id”: cid, “messages”: 0, “last”: h[“timestamp”]}
convs[cid][“messages”] += 1
convs[cid][“last”] = h[“timestamp”]
return {“conversations”: list(convs.values())}

@app.delete(”/history/{conv_id}”)
async def delete_conversation(conv_id: str):
history = load_history()
filtered = [h for h in history if h[“conv_id”] != conv_id]
save_history(filtered)
return {“deleted”: len(history) - len(filtered)}

@app.get(”/outputs/{filename}”)
async def download_file(filename: str):
from fastapi.responses import FileResponse
fpath = Path(“outputs”) / filename
if fpath.exists():
return FileResponse(fpath, filename=filename)
return {“error”: “File not found”}

@app.get(”/”, response_class=HTMLResponse)
async def index():
return HTMLResponse(HTML_PAGE)

# ── Frontend HTML (self-contained) ───────────────────────────────────────────

HTML_PAGE = r”””<!DOCTYPE html>

<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI Analyst</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<style>
  :root {
    --bg:#0d0f12; --surface:#151820; --surface2:#1c2030; --border:#252a3a;
    --accent:#4f9eff; --accent2:#7c6af7; --think:#f0a500; --tool:#3ecf8e;
    --text:#e2e8f0; --muted:#64748b; --user-bg:#1e3a5f; --user-border:#2d5a8e;
  }
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:var(--bg);color:var(--text);font-family:'IBM Plex Sans',sans-serif;height:100vh;display:flex;flex-direction:column;overflow:hidden}

/* Header */
.hdr{display:flex;align-items:center;gap:12px;padding:14px 24px;border-bottom:1px solid var(–border);background:var(–surface);flex-shrink:0}
.hdr-dot{width:8px;height:8px;border-radius:50%;background:var(–tool);box-shadow:0 0 8px var(–tool);animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
.hdr-title{font-family:‘IBM Plex Mono’,monospace;font-size:13px;font-weight:500;letter-spacing:.05em}
.hdr-model{font-size:11px;color:var(–muted);margin-left:auto;font-family:‘IBM Plex Mono’,monospace}
.hdr-new{background:var(–surface2);border:1px solid var(–border);border-radius:5px;padding:4px 10px;font-size:11px;font-family:‘IBM Plex Mono’,monospace;color:var(–muted);cursor:pointer;transition:all .15s}
.hdr-new:hover{color:var(–text);border-color:var(–accent)}

/* Layout */
.body{display:flex;flex:1;overflow:hidden}

/* Sidebar */
.sidebar{width:220px;border-right:1px solid var(–border);background:var(–surface);display:flex;flex-direction:column;overflow:hidden;flex-shrink:0}
.sb-title{font-size:10px;font-family:‘IBM Plex Mono’,monospace;color:var(–muted);padding:12px 14px 8px;text-transform:uppercase;letter-spacing:.08em;border-bottom:1px solid var(–border)}
.sb-list{flex:1;overflow-y:auto;padding:6px}
.sb-item{padding:8px 10px;border-radius:5px;cursor:pointer;font-size:12px;color:var(–muted);transition:all .15s;display:flex;flex-direction:column;gap:2px;margin-bottom:2px}
.sb-item:hover{background:var(–surface2);color:var(–text)}
.sb-item.active{background:rgba(79,158,255,.1);border:1px solid rgba(79,158,255,.2);color:var(–accent)}
.sb-item-id{font-family:‘IBM Plex Mono’,monospace;font-size:10px}
.sb-item-time{font-size:10px;color:var(–muted)}

/* Messages */
.msgs{flex:1;overflow-y:auto;padding:24px 0;display:flex;flex-direction:column;gap:12px;scrollbar-width:thin;scrollbar-color:var(–border) transparent}
.msgs::-webkit-scrollbar{width:4px}
.msgs::-webkit-scrollbar-thumb{background:var(–border);border-radius:2px}

/* User bubble */
.msg-user{display:flex;justify-content:flex-end;padding:0 24px}
.bubble{background:var(–user-bg);border:1px solid var(–user-border);border-radius:16px 16px 4px 16px;padding:10px 16px;max-width:65%;font-size:14px;line-height:1.6;color:#c8deff}

/* Agent message wrapper */
.msg-agent{padding:0 12px}
.agent-card{background:var(–surface);border:1px solid var(–border);border-radius:8px;overflow:hidden;margin-bottom:4px}

/* Thinking */
.think-hdr{display:flex;align-items:center;gap:8px;padding:8px 14px;cursor:pointer;background:rgba(240,165,0,.05);font-size:11px;font-family:‘IBM Plex Mono’,monospace;color:var(–think);user-select:none;transition:background .15s}
.think-hdr:hover{background:rgba(240,165,0,.1)}
.think-tog{margin-left:auto;font-size:10px;opacity:.6}
.think-body{padding:12px 14px;font-family:‘IBM Plex Mono’,monospace;font-size:12px;line-height:1.7;color:rgba(240,165,0,.75);background:rgba(240,165,0,.03);max-height:160px;overflow-y:auto;white-space:pre-wrap}
.think-block.collapsed .think-body{display:none}
.think-block{border-bottom:1px solid var(–border)}

/* Tool */
.tool-block{border-bottom:1px solid var(–border);background:rgba(62,207,142,.03)}
.tool-hdr{display:flex;align-items:center;gap:8px;padding:8px 14px;font-size:11px;font-family:‘IBM Plex Mono’,monospace;color:var(–tool)}
.tool-badge{background:rgba(62,207,142,.12);border:1px solid rgba(62,207,142,.25);border-radius:4px;padding:1px 7px;font-size:11px}
.tool-body{padding:0 14px 10px;display:flex;gap:12px}
.tool-col{flex:1}
.tool-lbl{font-size:10px;color:var(–muted);font-family:‘IBM Plex Mono’,monospace;margin-bottom:4px;text-transform:uppercase;letter-spacing:.08em}
.tool-code{background:var(–bg);border:1px solid var(–border);border-radius:4px;padding:8px 10px;font-family:‘IBM Plex Mono’,monospace;font-size:11px;color:#a8c4f0;white-space:pre-wrap;overflow-x:auto;word-break:break-all}
.ok{color:var(–tool)} .err{color:#ff6b6b}

/* Content */
.content-block{padding:14px;font-size:14px;line-height:1.75}

/* Final */
.final-wrap{margin:0 12px;padding:16px 20px;background:linear-gradient(135deg,rgba(79,158,255,.06),rgba(124,106,247,.06));border:1px solid rgba(79,158,255,.2);border-radius:8px}
.final-lbl{font-size:10px;font-family:‘IBM Plex Mono’,monospace;color:var(–accent);text-transform:uppercase;letter-spacing:.1em;margin-bottom:10px;display:flex;align-items:center;gap:6px}
.final-text{font-size:14px;line-height:1.8;margin-bottom:14px}
.attach{display:inline-flex;align-items:center;gap:8px;background:var(–surface2);border:1px solid var(–border);border-radius:6px;padding:8px 14px;font-family:‘IBM Plex Mono’,monospace;font-size:12px;color:var(–accent);text-decoration:none;transition:all .15s}
.attach:hover{background:rgba(79,158,255,.1);border-color:var(–accent)}

/* Step badge */
.step-badge{display:flex;align-items:center;gap:8px;padding:2px 12px;font-size:10px;font-family:‘IBM Plex Mono’,monospace;color:var(–muted)}
.step-badge::before,.step-badge::after{content:’’;flex:1;height:1px;background:var(–border)}

/* Cursor */
.cur{display:inline-block;width:2px;height:13px;background:var(–accent);margin-left:2px;vertical-align:middle;animation:blink 1s infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:0}}

/* Input */
.input-area{padding:16px 24px;border-top:1px solid var(–border);background:var(–surface);flex-shrink:0}
.input-row{display:flex;gap:10px;align-items:flex-end;background:var(–surface2);border:1px solid var(–border);border-radius:10px;padding:10px 14px;transition:border-color .2s}
.input-row:focus-within{border-color:rgba(79,158,255,.4)}
textarea{flex:1;background:transparent;border:none;outline:none;color:var(–text);font-family:‘IBM Plex Sans’,sans-serif;font-size:14px;resize:none;min-height:24px;max-height:120px;line-height:1.5}
textarea::placeholder{color:var(–muted)}
.send-btn{background:var(–accent);border:none;border-radius:6px;width:32px;height:32px;cursor:pointer;display:flex;align-items:center;justify-content:center;flex-shrink:0;transition:all .15s;color:#fff;font-size:16px}
.send-btn:hover{background:#6aadff;transform:scale(1.05)}
.send-btn:disabled{background:var(–border);cursor:not-allowed;transform:none}
.hint{font-size:11px;color:var(–muted);margin-top:8px;text-align:center;font-family:‘IBM Plex Mono’,monospace}

/* Empty state */
.empty{display:flex;flex-direction:column;align-items:center;justify-content:center;flex:1;gap:10px;color:var(–muted);font-family:‘IBM Plex Mono’,monospace;font-size:13px}
.empty-icon{font-size:32px;opacity:.3}
</style>

</head>
<body>

<div class="hdr">
  <div class="hdr-dot"></div>
  <span class="hdr-title">AI_ANALYST</span>
  <span class="hdr-model" id="modelLabel">gpt-oss:20b</span>
  <button class="hdr-new" onclick="newConversation()">+ New</button>
</div>

<div class="body">
  <div class="sidebar">
    <div class="sb-title">Conversations</div>
    <div class="sb-list" id="sbList"></div>
  </div>

  <div style="flex:1;display:flex;flex-direction:column;overflow:hidden">
    <div class="msgs" id="msgs">
      <div class="empty" id="emptyState">
        <div class="empty-icon">⬡</div>
        <div>Start a conversation</div>
      </div>
    </div>

```
<div class="input-area">
  <div class="input-row">
    <textarea id="inp" placeholder="Ask the analyst…" rows="1"
      onkeydown="handleKey(event)" oninput="autoResize(this)"></textarea>
    <button class="send-btn" id="sendBtn" onclick="send()">↑</button>
  </div>
  <div class="hint">enter · send &nbsp;|&nbsp; shift+enter · newline</div>
</div>
```

  </div>
</div>

<script>
// ── State ──────────────────────────────────────────────────────────────────
let convId = null;
let streaming = false;

// ── Init ───────────────────────────────────────────────────────────────────
window.onload = () => {
  newConversation();
  loadSidebar();
};

function newConversation() {
  convId = crypto.randomUUID();
  document.getElementById('msgs').innerHTML =
    `<div class="empty" id="emptyState"><div class="empty-icon">⬡</div><div>Start a conversation</div></div>`;
  loadSidebar();
  markActive(convId);
}

async function loadSidebar() {
  try {
    const r = await fetch('/history');
    const {conversations} = await r.json();
    const list = document.getElementById('sbList');
    list.innerHTML = '';
    conversations.sort((a,b) => b.last.localeCompare(a.last)).forEach(c => {
      const el = document.createElement('div');
      el.className = 'sb-item' + (c.conv_id === convId ? ' active' : '');
      el.dataset.id = c.conv_id;
      el.innerHTML = `<span class="sb-item-id">${c.conv_id.slice(0,8)}…</span>
                      <span class="sb-item-time">${c.messages} msgs · ${c.last.slice(0,10)}</span>`;
      el.onclick = () => loadConversation(c.conv_id);
      list.appendChild(el);
    });
  } catch(e) {}
}

async function loadConversation(id) {
  convId = id;
  markActive(id);
  try {
    const r = await fetch(`/history/${id}`);
    const {history} = await r.json();
    const msgs = document.getElementById('msgs');
    msgs.innerHTML = '';
    history.forEach(h => {
      if (h.role === 'user') appendUserBubble(h.content);
    });
  } catch(e) {}
}

function markActive(id) {
  document.querySelectorAll('.sb-item').forEach(el => {
    el.classList.toggle('active', el.dataset.id === id);
  });
}

// ── Send ───────────────────────────────────────────────────────────────────
async function send() {
  if (streaming) return;
  const inp = document.getElementById('inp');
  const text = inp.value.trim();
  if (!text) return;

  inp.value = '';
  inp.style.height = 'auto';
  document.getElementById('sendBtn').disabled = true;
  document.getElementById('emptyState')?.remove();
  streaming = true;

  appendUserBubble(text);

  // State for current response
  let currentStep = 0;
  let thinkBlock = null;
  let thinkBody = null;
  let contentBlock = null;
  let agentCard = null;
  let stepWrapper = null;

  function ensureCard(step) {
    if (step !== currentStep) {
      currentStep = step;
      const badge = document.createElement('div');
      badge.className = 'step-badge';
      badge.textContent = `Step ${step}`;
      document.getElementById('msgs').appendChild(badge);

      stepWrapper = document.createElement('div');
      stepWrapper.className = 'msg-agent';
      agentCard = document.createElement('div');
      agentCard.className = 'agent-card';
      stepWrapper.appendChild(agentCard);
      document.getElementById('msgs').appendChild(stepWrapper);

      thinkBlock = null;
      thinkBody = null;
      contentBlock = null;
    }
  }

  try {
    const resp = await fetch('/chat/stream', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({conv_id: convId, user_message: text})
    });

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';

    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      buf += decoder.decode(value, {stream: true});

      const parts = buf.split('\n\n');
      buf = parts.pop();

      for (const part of parts) {
        const lines = part.trim().split('\n');
        const evtLine = lines.find(l => l.startsWith('event:'));
        const dataLine = lines.find(l => l.startsWith('data:'));
        if (!evtLine || !dataLine) continue;

        const evt = evtLine.slice(7).trim();
        const data = JSON.parse(dataLine.slice(5).trim());

        switch(evt) {

          case 'step_start':
            ensureCard(data.step);
            break;

          case 'thinking_start':
            ensureCard(data.step);
            thinkBlock = document.createElement('div');
            thinkBlock.className = 'think-block';
            const thinkHdr = document.createElement('div');
            thinkHdr.className = 'think-hdr';
            thinkHdr.innerHTML = `<span>⟳</span><span>Thinking</span><span class="think-tog">▾</span>`;
            thinkBody = document.createElement('div');
            thinkBody.className = 'think-body';
            thinkBody.innerHTML = '<span class="cur"></span>';
            thinkBlock.appendChild(thinkHdr);
            thinkBlock.appendChild(thinkBody);
            thinkHdr.onclick = () => {
              thinkBlock.classList.toggle('collapsed');
              thinkHdr.querySelector('.think-tog').textContent =
                thinkBlock.classList.contains('collapsed') ? '▸' : '▾';
            };
            agentCard.appendChild(thinkBlock);
            break;

          case 'thinking_delta':
            if (thinkBody) {
              const cur = thinkBody.querySelector('.cur');
              if (cur) cur.insertAdjacentText('beforebegin', data.text);
              else thinkBody.appendChild(document.createTextNode(data.text));
              scrollBottom();
            }
            break;

          case 'thinking_end':
            if (thinkBody) {
              thinkBody.querySelector('.cur')?.remove();
              // Auto-collapse after done
              if (thinkBlock) {
                thinkBlock.classList.add('collapsed');
                thinkBlock.querySelector('.think-tog').textContent = '▸';
              }
            }
            break;

          case 'tool_start': {
            ensureCard(data.step);
            const tb = document.createElement('div');
            tb.className = 'tool-block';
            tb.id = `tool_${data.step}_${data.name}`;
            tb.innerHTML = `
              <div class="tool-hdr">
                <span>⚙</span>
                <span class="tool-badge">${data.name}</span>
                <span style="color:var(--muted);font-size:10px;margin-left:auto">step ${data.step}</span>
              </div>
              <div class="tool-body">
                <div class="tool-col">
                  <div class="tool-lbl">params</div>
                  <div class="tool-code">${JSON.stringify(data.args, null, 2)}</div>
                </div>
                <div class="tool-col">
                  <div class="tool-lbl">result</div>
                  <div class="tool-code" id="tr_${data.step}_${data.name}"><span class="cur"></span></div>
                </div>
              </div>`;
            agentCard.appendChild(tb);
            contentBlock = null;
            scrollBottom();
            break;
          }

          case 'tool_result': {
            const tr = document.getElementById(`tr_${data.step}_${data.name}`);
            if (tr) {
              tr.className = 'tool-code ' + (data.status === 'ok' ? 'ok' : 'err');
              tr.textContent = (data.status === 'ok' ? '✓ ' : '✗ ') + data.result;
            }
            break;
          }

          case 'content_start':
            ensureCard(data.step);
            contentBlock = document.createElement('div');
            contentBlock.className = 'content-block';
            contentBlock.innerHTML = '<span class="cur"></span>';
            agentCard.appendChild(contentBlock);
            break;

          case 'content_delta':
            if (contentBlock) {
              const cur = contentBlock.querySelector('.cur');
              if (cur) cur.insertAdjacentText('beforebegin', data.text);
              else contentBlock.appendChild(document.createTextNode(data.text));
              scrollBottom();
            }
            break;

          case 'content_end':
            if (contentBlock) contentBlock.querySelector('.cur')?.remove();
            break;

          case 'final': {
            // Remove intermediate content blocks — final goes in its own div
            const fw = document.createElement('div');
            fw.className = 'final-wrap';
            fw.innerHTML = `
              <div class="final-lbl"><span>◈</span> Final Response</div>
              <div class="final-text">${data.response.replace(/\n/g,'<br>')}</div>
              ${data.file ? `<a class="attach" href="/outputs/${data.file}" download>
                <span>⬇</span> ${data.file}
              </a>` : ''}`;
            document.getElementById('msgs').appendChild(fw);
            scrollBottom();
            loadSidebar();
            break;
          }

          case 'done':
            break;
        }
      }
    }
  } catch(e) {
    const err = document.createElement('div');
    err.style = 'padding:12px 24px;color:#ff6b6b;font-size:13px;font-family:monospace';
    err.textContent = 'Stream error: ' + e.message;
    document.getElementById('msgs').appendChild(err);
  } finally {
    streaming = false;
    document.getElementById('sendBtn').disabled = false;
  }
}

function appendUserBubble(text) {
  const d = document.createElement('div');
  d.className = 'msg-user';
  d.innerHTML = `<div class="bubble">${escHtml(text)}</div>`;
  document.getElementById('msgs').appendChild(d);
  scrollBottom();
}

function escHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function scrollBottom() {
  const m = document.getElementById('msgs');
  m.scrollTop = m.scrollHeight;
}

function handleKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
}

function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 120) + 'px';
}
</script>

</body>
</html>
"""

(() => {
  "use strict";

  const API_BASE = window.BHUMIBONDHU_API || "";       // same origin by default
  const API_KEY = window.BHUMIBONDHU_KEY || "";        // only needed when PUBLIC_API_KEYS is set
  const STORE = "bhumibondhu.conversation";

  const $ = (id) => document.getElementById(id);
  const chat = $("chat"), thread = $("thread"), welcome = $("welcome");
  const form = $("form"), input = $("input"), sendBtn = $("send"), statusEl = $("status");

  const bn = (n) => String(n).replace(/\d/g, (d) => "০১২৩৪৫৬৭৮৯"[d]);
  const esc = (s) => s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const safeUrl = (u) => { try { const x = new URL(u); return /^https?:$/.test(x.protocol) ? x.href : null; } catch { return null; } };

  let conversationId = null;
  try { conversationId = localStorage.getItem(STORE); } catch {}
  let busy = false;
  let uid = 0;

  const BOT_AVATAR = `<span class="avatar"><img src="assets/bot.png" alt="" width="30" height="30"></span>`;

  /* ---------- rendering ---------- */
  function inline(text, msgId) {
    let h = esc(text)
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(/(^|[\s(])\*([^*\s][^*]*?)\*(?=[\s).,;:!?]|$)/g, "$1<em>$2</em>");
    return h.replace(/\[(\d{1,2})\]/g, (_, n) => `<a class="cite" href="#${msgId}-src-${n}" data-src="${msgId}-src-${n}">${bn(n)}</a>`);
  }

  const cells = (row) => row.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((c) => c.trim());
  const isSep = (row) => /^\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?$/.test(row.trim());

  function renderTable(rows, msgId) {
    const hasHead = rows.length > 1 && isSep(rows[1]);
    const body = rows.filter((r, i) => !isSep(r) && !(hasHead && i === 0)).map(cells);
    const head = hasHead ? cells(rows[0]) : null;
    const cols = Math.max(head ? head.length : 0, ...body.map((r) => r.length));
    // A narrow "number | text" table reads better as a numbered card list on small screens; keep as table.
    const td = (t, tag) => `<${tag}>${inline(t, msgId)}</${tag}>`;
    const pad = (r) => Array.from({ length: cols }, (_, i) => r[i] || "");
    return `<div class="tbl"><table>${head ? `<thead><tr>${pad(head).map((c) => td(c, "th")).join("")}</tr></thead>` : ""}` +
      `<tbody>${body.map((r) => `<tr>${pad(r).map((c) => td(c, "td")).join("")}</tr>`).join("")}</tbody></table></div>`;
  }

  function renderAnswer(text, msgId) {
    const lines = text.replace(/\r/g, "").split("\n");
    const out = [];
    let list = null, para = [];
    const flushPara = () => { if (para.length) { out.push(`<p>${para.map((l) => inline(l, msgId)).join("<br>")}</p>`); para = []; } };
    const flushList = () => { if (list) { out.push(`</${list}>`); list = null; } };
    const flush = () => { flushPara(); flushList(); };

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i].trim();
      if (!line) { flush(); continue; }

      if (line.startsWith("|")) {                       // table: consecutive pipe rows
        flush();
        const rows = [];
        while (i < lines.length && lines[i].trim().startsWith("|")) rows.push(lines[i++].trim());
        i--;
        out.push(renderTable(rows, msgId));
        continue;
      }
      if (/^(-{3,}|\*{3,}|_{3,})$/.test(line)) { flush(); out.push("<hr>"); continue; }
      const h = line.match(/^(#{1,6})\s+(.*)/);
      if (h) { flush(); out.push(`<h${Math.min(h[1].length + 2, 5)}>${inline(h[2], msgId)}</h${Math.min(h[1].length + 2, 5)}>`); continue; }
      const q = line.match(/^>\s?(.*)/);
      if (q) { flush(); out.push(`<blockquote>${inline(q[1], msgId)}</blockquote>`); continue; }

      const ul = line.match(/^[-*•]\s+(.*)/);
      const ol = line.match(/^(?:\d+|[০-৯]+)[.)]\s+(.*)/);
      if (ul || ol) {
        flushPara();
        const tag = ul ? "ul" : "ol";
        if (list !== tag) { flushList(); out.push(`<${tag}>`); list = tag; }
        out.push(`<li>${inline((ul || ol)[1], msgId)}</li>`);
        continue;
      }
      flushList();
      para.push(line);
    }
    flush();
    return out.join("");
  }

  // Q&A records are never listed under তথ্যসূত্র (their [n] markers are removed from the answer too).
  const isQna = (s) => /^qna_/.test(s.element_type || s.source_type || "");
  function stripHiddenCites(text, sources) {
    const hidden = new Set((sources || []).filter(isQna).map((s) => String(s.index)));
    if (!hidden.size) return text;
    return text.replace(/[ \t]*\[(\d{1,2})\]/g, (m, n) => (hidden.has(n) ? "" : m));
  }

  function renderSources(sources, msgId) {
    const cited = (sources || []).filter((s) => s.cited !== false && !isQna(s));
    if (!cited.length) return "";
    const items = cited.map((s) => {
      const url = s.url && safeUrl(s.url);
      const title = esc(s.title || "শিরোনামহীন সূত্র");
      const head = url ? `<a href="${esc(url)}" target="_blank" rel="noopener noreferrer">${title} ↗</a>` : title;
      const tags = [s.doc_type || (s.source_type && !/[_\d]/.test(s.source_type) ? s.source_type : null), s.year && bn(s.year), s.section].filter(Boolean)
        .map((t) => `<span class="tag">${esc(String(t))}</span>`).join("");
      const auth = s.authority ? `<div class="m">${esc(s.authority)}</div>` : "";
      return `<div class="src" id="${msgId}-src-${s.index}"><span class="n">${bn(s.index)}</span><div><div class="t">${head}</div><div class="m">${tags}</div>${auth}</div></div>`;
    }).join("");
    return `<div class="sources"><h4>তথ্যসূত্র</h4>${items}</div>`;
  }

  function addMessage(kind, html, extra = "") {
    const el = document.createElement("div");
    el.className = `msg ${kind === "error" ? "bot error" : kind}`;
    el.innerHTML = (kind === "user" ? "" : BOT_AVATAR) + `<div class="bubble">${kind === "user" ? "" : `<div class="who">${kind === "error" ? "সতর্কতা" : "ভূমিবন্ধু"}</div>`}${html}${extra}</div>`;
    thread.appendChild(el);
    scrollDown();
    return el;
  }

  const scrollDown = () => requestAnimationFrame(() => { chat.scrollTop = chat.scrollHeight; });

  /* ---------- network ---------- */
  function errorText(status, detail) {
    if (status === 429) return "অনেক বেশি অনুরোধ পাঠানো হয়েছে। অনুগ্রহ করে একটু পরে আবার চেষ্টা করুন।";
    if (status === 401) return "অনুমোদন ব্যর্থ হয়েছে। অনুগ্রহ করে প্রশাসকের সাথে যোগাযোগ করুন।";
    if (status === 422 && typeof detail === "string") return detail;
    if (status === 422) return "প্রশ্নটি গ্রহণযোগ্য নয়। অনুগ্রহ করে অন্যভাবে লিখে আবার চেষ্টা করুন।";
    if (status === 503 || status === 504) return "সেবাটি সাময়িকভাবে অনুপলব্ধ। কিছুক্ষণ পরে আবার চেষ্টা করুন।";
    return "দুঃখিত, একটি সমস্যা হয়েছে। অনুগ্রহ করে আবার চেষ্টা করুন।";
  }

  async function ask(message) {
    const headers = { "Content-Type": "application/json" };
    if (API_KEY) headers["X-API-Key"] = API_KEY;
    const body = { message };
    if (conversationId) body.conversation_id = conversationId;
    const res = await fetch(`${API_BASE}/api/chat`, { method: "POST", headers, body: JSON.stringify(body) });
    let data = null;
    try { data = await res.json(); } catch {}
    if (!res.ok) {
      const err = new Error("http"); err.status = res.status; err.detail = data && data.detail; throw err;
    }
    return data;
  }

  async function send(text) {
    text = text.trim();
    if (!text || busy) return;
    busy = true; sendBtn.disabled = true;
    welcome.hidden = true;
    addMessage("user", esc(text));
    input.value = ""; autosize();

    const typing = addMessage("bot", `<span class="typing" aria-label="উত্তর তৈরি হচ্ছে"><i></i><i></i><i></i></span>`);
    try {
      const data = await ask(text);
      if (data.conversation_id) {
        conversationId = data.conversation_id;
        try { localStorage.setItem(STORE, conversationId); } catch {}
      }
      const id = `m${++uid}`;
      typing.remove();
      const answer = stripHiddenCites(data.answer || "", data.sources);
      const el = addMessage("bot", renderAnswer(answer, id), renderSources(data.sources, id) +
        `<div class="tools"><button class="tool" type="button" data-copy>কপি করুন</button></div>`);
      el.dataset.text = answer;
      setStatus(true);
    } catch (e) {
      typing.remove();
      addMessage("error", esc(e.status ? errorText(e.status, e.detail) : "সংযোগ বিচ্ছিন্ন। ইন্টারনেট সংযোগ পরীক্ষা করে আবার চেষ্টা করুন।"));
      setStatus(!!e.status);
    } finally {
      busy = false; sendBtn.disabled = !input.value.trim(); input.focus();
    }
  }

  /* ---------- theme ---------- */
  const root = document.documentElement, themeBtn = $("theme");
  function applyTheme(t) {
    root.dataset.theme = t;
    themeBtn.setAttribute("aria-label", t === "dark" ? "লাইট মোড চালু করুন" : "ডার্ক মোড চালু করুন");
    const m = document.querySelector('meta[name="theme-color"]');
    if (m) m.content = t === "dark" ? "#0a4a3d" : "#0f6e5a";
  }
  applyTheme(root.dataset.theme === "dark" ? "dark" : "light");
  themeBtn.addEventListener("click", () => {
    const t = root.dataset.theme === "dark" ? "light" : "dark";
    applyTheme(t);
    try { localStorage.setItem("bhumibondhu.theme", t); } catch {}
  });

  /* ---------- UI wiring ---------- */
  function autosize() {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 160) + "px";
    sendBtn.disabled = busy || !input.value.trim();
  }
  function setStatus(ok) {
    statusEl.classList.toggle("off", !ok);
    statusEl.lastElementChild.textContent = ok ? "অনলাইন" : "অফলাইন";
  }

  /* ---------- type-ahead suggestions ---------- */
  const ta = $("typeahead");
  let taItems = [], taIndex = -1, taTimer = null, taSeq = 0;

  function taClose() {
    taItems = []; taIndex = -1; ta.hidden = true; ta.innerHTML = "";
    input.setAttribute("aria-expanded", "false"); input.removeAttribute("aria-activedescendant");
  }
  function taHighlight(i) {
    taIndex = i;
    [...ta.children].forEach((li, n) => li.setAttribute("aria-selected", String(n === i)));
    if (i >= 0) input.setAttribute("aria-activedescendant", `ta-${i}`); else input.removeAttribute("aria-activedescendant");
  }
  function taRender(items) {
    taItems = items; taIndex = -1;
    if (!items.length) return taClose();
    ta.innerHTML = items.map((q, i) => `<li role="option" id="ta-${i}" aria-selected="false" data-i="${i}">${esc(q)}</li>`).join("");
    ta.hidden = false; input.setAttribute("aria-expanded", "true");
  }
  async function taFetch(q) {
    const seq = ++taSeq;
    try {
      const headers = API_KEY ? { "X-API-Key": API_KEY } : {};
      const res = await fetch(`${API_BASE}/api/suggest?q=${encodeURIComponent(q)}`, { headers });
      if (!res.ok) return;
      const { suggestions } = await res.json();
      if (seq === taSeq && input.value.trim() === q) taRender(suggestions);
    } catch {}
  }
  function taQueue() {
    clearTimeout(taTimer);
    const q = input.value.trim();
    taSeq++;
    if (q.length < 2 || busy) return taClose();
    taTimer = setTimeout(() => taFetch(q), 200);
  }
  function taPick(i) {
    const q = taItems[i]; taClose();
    if (!q) return;
    input.value = q; autosize(); input.focus();   // user reviews/edits, then sends
    input.setSelectionRange(q.length, q.length);
  }
  ta.addEventListener("mousedown", (e) => e.preventDefault());   // keep focus in the textarea
  ta.addEventListener("click", (e) => { const li = e.target.closest("li[data-i]"); if (li) taPick(+li.dataset.i); });
  input.addEventListener("blur", () => setTimeout(taClose, 100));

  form.addEventListener("submit", (e) => { e.preventDefault(); taClose(); send(input.value); });
  input.addEventListener("input", () => { autosize(); taQueue(); });
  input.addEventListener("keydown", (e) => {
    if (!ta.hidden && !e.isComposing) {
      if (e.key === "ArrowDown") { e.preventDefault(); return taHighlight((taIndex + 1) % taItems.length); }
      if (e.key === "ArrowUp") { e.preventDefault(); return taHighlight((taIndex - 1 + taItems.length) % taItems.length); }
      if (e.key === "Escape") { e.preventDefault(); return taClose(); }
      if (e.key === "Enter" && !e.shiftKey && taIndex >= 0) { e.preventDefault(); return taPick(taIndex); }
    }
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); form.requestSubmit(); }
  });
  $("suggest").addEventListener("click", (e) => {
    const b = e.target.closest("button[data-q]"); if (b) send(b.dataset.q);
  });
  $("newChat").addEventListener("click", () => {
    if (busy) return;
    conversationId = null;
    try { localStorage.removeItem(STORE); } catch {}
    thread.innerHTML = ""; welcome.hidden = false; input.focus();
  });
  thread.addEventListener("click", (e) => {
    const c = e.target.closest("a.cite");
    if (c) {
      e.preventDefault();
      const t = document.getElementById(c.dataset.src);
      if (t) { t.scrollIntoView({ behavior: "smooth", block: "center" }); t.classList.remove("flash"); void t.offsetWidth; t.classList.add("flash"); }
      return;
    }
    const cp = e.target.closest("[data-copy]");
    if (cp) {
      const text = cp.closest(".msg").dataset.text || "";
      (navigator.clipboard ? navigator.clipboard.writeText(text) : Promise.reject())
        .then(() => { cp.textContent = "কপি হয়েছে ✓"; setTimeout(() => (cp.textContent = "কপি করুন"), 1500); })
        .catch(() => {});
    }
  });

  // Restore the previous conversation, if the server still has it.
  (async () => {
    if (!conversationId) return;
    try {
      const headers = API_KEY ? { "X-API-Key": API_KEY } : {};
      const res = await fetch(`${API_BASE}/api/conversations/${encodeURIComponent(conversationId)}`, { headers });
      if (!res.ok) throw new Error();
      const { turns } = await res.json();
      if (!turns.length) return;
      welcome.hidden = true;
      for (const t of turns) {
        if (t.role === "user") addMessage("user", esc(t.content));
        else { const id = `m${++uid}`; addMessage("bot", renderAnswer(t.content, id)).dataset.text = t.content; }
      }
    } catch {
      conversationId = null;
      try { localStorage.removeItem(STORE); } catch {}
    }
  })();

  input.focus();
})();

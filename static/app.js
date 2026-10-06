const $ = (id) => document.getElementById(id);
const log = $("log"), input = $("input"), send = $("send"), modelList = $("modelList"), modelPill = $("modelPill");

// name: sidebar group heading; short: shown in the header pill
const PROVIDERS = {
  bedrock: { name: "Claude · Amazon Bedrock", short: "Amazon Bedrock", color: "var(--bedrock)" },
  bedrock_converse: { name: "More models · Amazon Bedrock", short: "Amazon Bedrock", color: "var(--bedrock)" },
  openai: { name: "OpenAI", short: "OpenAI", color: "var(--openai)" },
};
let models = [];
let currentModel = null;
let sessionId = null;
let busy = false;
let storeInitials = "VK";

// ---------- Models ----------
async function loadModels() {
  const res = await fetch("/api/models");
  const data = await res.json();
  models = data.models;
  applyStoreName(data.store_name);
  let saved = null;
  try { saved = localStorage.getItem("chat-model"); } catch {}
  const usable = (id) => models.some((m) => m.id === id && m.available);
  currentModel = usable(saved) ? saved : data.default;
  renderModels();
}

function applyStoreName(name) {
  if (!name) return;
  // "VoltKart" -> "VK", "Tech World" -> "TW", "shop" -> "SH"
  const caps = name.match(/[A-Z]/g) || [];
  const words = name.split(/\s+/).map((w) => w[0]);
  storeInitials = (caps.length >= 2 ? caps : words.length >= 2 ? words : [...name]).slice(0, 2).join("").toUpperCase();
  document.title = `${name} Assistant`;
  $("brandName").textContent = name;
  $("logo").textContent = storeInitials;
  document.querySelectorAll(".row.bot .avatar").forEach((a) => (a.textContent = storeInitials));
}

function renderModels() {
  modelList.innerHTML = "";
  for (const [key, p] of Object.entries(PROVIDERS)) {
    const group = models.filter((m) => m.provider === key);
    if (!group.length) continue;
    const head = document.createElement("div");
    head.className = "model-group";
    head.innerHTML = `<span class="badge" style="background:${p.color}"></span>${p.name}`;
    modelList.appendChild(head);
    for (const m of group) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "model";
      btn.setAttribute("role", "radio");
      btn.setAttribute("aria-checked", String(m.id === currentModel));
      btn.disabled = !m.available;
      btn.title = m.available ? "" : "Set OPENAI_API_KEY on the server to enable";
      btn.innerHTML = `<span class="radio"></span><span><div class="m-name"></div><div class="m-note"></div></span>`;
      btn.querySelector(".m-name").textContent = m.label;
      btn.querySelector(".m-note").textContent = m.available ? m.note : "Not configured";
      btn.addEventListener("click", () => selectModel(m.id));
      modelList.appendChild(btn);
    }
  }
  const m = models.find((x) => x.id === currentModel);
  modelPill.textContent = m ? `${m.label} · ${PROVIDERS[m.provider].short}` : "—";
}

function selectModel(id) {
  currentModel = id;
  try { localStorage.setItem("chat-model", id); } catch {}
  renderModels();
  closeSidebar();
}

// ---------- Rendering ----------
function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// Minimal, safe markdown: **bold**, "- " bullets, paragraphs.
function renderMarkdown(text) {
  const inline = (s) => escapeHtml(s).replace(/\*\*(.+?)\*\*/g, "<b>$1</b>");
  const out = [];
  let list = null, para = [];
  const flushPara = () => { if (para.length) { out.push(`<p>${para.join("<br>")}</p>`); para = []; } };
  const flushList = () => { if (list) { out.push(`<ul>${list.join("")}</ul>`); list = null; } };
  for (const raw of text.split("\n")) {
    const line = raw.trimEnd();
    const bullet = line.match(/^\s*[-*•]\s+(.*)$/);
    if (bullet) { flushPara(); (list ||= []).push(`<li>${inline(bullet[1])}</li>`); }
    else if (!line.trim()) { flushPara(); flushList(); }
    else { flushList(); para.push(inline(line.replace(/^#+\s*/, ""))); }
  }
  flushPara(); flushList();
  return out.join("");
}

function addRow(role) {
  $("welcome")?.remove();
  const row = document.createElement("div");
  row.className = `row ${role}`;
  row.innerHTML = `<div class="avatar">${role === "user" ? "You" : storeInitials}</div><div class="body"><div class="bubble"></div></div>`;
  log.appendChild(row);
  log.scrollTop = log.scrollHeight;
  return row;
}

function addUser(text) {
  addRow("user").querySelector(".bubble").textContent = text;
}

function addTyping() {
  const row = addRow("bot");
  row.querySelector(".bubble").innerHTML = `<div class="typing"><span></span><span></span><span></span></div>`;
  return row;
}

function fillBot(row, data) {
  row.querySelector(".bubble").innerHTML = renderMarkdown(data.reply);
  const m = models.find((x) => x.id === data.model);
  const meta = document.createElement("div");
  meta.className = "meta";
  meta.innerHTML = `<span class="badge" style="background:${PROVIDERS[m?.provider]?.color || "var(--muted)"}"></span>`;
  meta.append(`${data.model_label} · ${(data.latency_ms / 1000).toFixed(1)}s`);
  row.querySelector(".body").appendChild(meta);
  log.scrollTop = log.scrollHeight;
}

function fillError(row, message) {
  const bubble = row.querySelector(".bubble");
  bubble.classList.add("error");
  bubble.textContent = message;
}

// ---------- Chat ----------
async function ask(text) {
  text = text.trim();
  if (!text || busy) return;
  busy = true; send.disabled = true;
  addUser(text);
  input.value = ""; autosize();
  const row = addTyping();
  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text, session_id: sessionId, model: currentModel }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || `Request failed (${res.status})`);
    sessionId = data.session_id;
    fillBot(row, data);
  } catch (e) {
    fillError(row, e.message);
  } finally {
    busy = false; send.disabled = false; input.focus();
  }
}

const welcomeHtml = log.innerHTML;

function wireQuickActions() {
  document.querySelectorAll(".quick button").forEach((b) => b.addEventListener("click", () => ask(b.dataset.q)));
}

function newConversation() {
  if (busy) return;
  if (sessionId) {
    fetch("/api/reset", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ message: "", session_id: sessionId }) });
  }
  sessionId = null;
  log.innerHTML = welcomeHtml;
  wireQuickActions();
  closeSidebar();
  input.focus();
}

// ---------- UI wiring ----------
function autosize() {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 160) + "px";
}
function closeSidebar() { $("sidebar").classList.remove("open"); $("scrim").classList.remove("open"); }

$("form").addEventListener("submit", (e) => { e.preventDefault(); ask(input.value); });
input.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); ask(input.value); } });
input.addEventListener("input", autosize);
wireQuickActions();
$("newChat").addEventListener("click", newConversation);
$("menuBtn").addEventListener("click", () => { $("sidebar").classList.add("open"); $("scrim").classList.add("open"); });
$("scrim").addEventListener("click", closeSidebar);

loadModels().catch(() => { modelPill.textContent = "Server unavailable"; });

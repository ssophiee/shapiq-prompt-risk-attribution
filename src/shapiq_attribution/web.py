"""Static HTML for the prompt-risk attribution web interface.

Kept as a Python string so the package has no static-file plumbing and the
single-page UI ships inside the wheel / Docker image with no extra COPY steps.
"""

from __future__ import annotations

INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Prompt-Risk Attribution</title>
<style>
  :root {
    --bg: #0f1117;
    --panel: #181b24;
    --border: #272b37;
    --text: #e6e8ee;
    --muted: #9aa1b1;
    --accent: #6d6dff;
    --safe: #2ea66b;
    --risky: #e0533d;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.5;
  }
  .wrap { max-width: 820px; margin: 0 auto; padding: 48px 24px 96px; }
  header h1 { font-size: 24px; margin: 0 0 6px; }
  header p { color: var(--muted); margin: 0 0 32px; font-size: 14px; }
  .card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 20px;
  }
  textarea {
    width: 100%;
    min-height: 110px;
    resize: vertical;
    background: #0f1117;
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px 14px;
    font-size: 15px;
    font-family: inherit;
  }
  textarea:focus { outline: none; border-color: var(--accent); }
  .row { display: flex; gap: 12px; align-items: center; margin-top: 14px; flex-wrap: wrap; }
  button {
    background: var(--accent);
    color: #fff;
    border: none;
    border-radius: 10px;
    padding: 11px 20px;
    font-size: 15px;
    font-weight: 600;
    cursor: pointer;
  }
  button:disabled { opacity: 0.5; cursor: not-allowed; }
  .toggle { display: flex; align-items: center; gap: 8px; color: var(--muted); font-size: 14px; }
  select {
    background: #0f1117; color: var(--text); border: 1px solid var(--border);
    border-radius: 8px; padding: 6px 8px; font-size: 14px; font-family: inherit;
  }
  select:focus { outline: none; border-color: var(--accent); }
  .hidden { display: none; }
  .result-head { display: flex; align-items: center; gap: 14px; margin-bottom: 16px; }
  .badge {
    font-weight: 700;
    font-size: 13px;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    padding: 5px 12px;
    border-radius: 999px;
  }
  .badge.safe { background: rgba(46,166,107,0.15); color: var(--safe); }
  .badge.risky { background: rgba(224,83,61,0.15); color: var(--risky); }
  .prob { font-size: 28px; font-weight: 700; }
  .prob small { font-size: 14px; color: var(--muted); font-weight: 400; }
  .bar { height: 10px; background: #0f1117; border-radius: 999px; overflow: hidden; margin: 4px 0 20px; }
  .bar > div { height: 100%; border-radius: 999px; transition: width 0.4s ease; }
  .section-title {
    font-size: 13px; text-transform: uppercase; letter-spacing: 0.05em;
    color: var(--muted); margin: 0 0 12px;
  }
  .tokens { line-height: 2.2; font-size: 16px; }
  .tok {
    padding: 3px 5px;
    border-radius: 6px;
    margin: 0 1px;
    white-space: pre-wrap;
  }
  .legend { display: flex; gap: 18px; margin-top: 16px; font-size: 12px; color: var(--muted); align-items: center; }
  .legend span { display: inline-flex; align-items: center; gap: 6px; }
  .sw { width: 14px; height: 14px; border-radius: 4px; display: inline-block; }
  .interactions { margin-top: 28px; }
  .pair {
    display: flex; align-items: center; gap: 12px;
    padding: 9px 12px; border-radius: 9px; background: #0f1117;
    border: 1px solid var(--border); margin-bottom: 8px;
  }
  .pair .terms { flex: 1; font-size: 15px; }
  .pair .terms b { color: var(--text); }
  .pair .arrow { color: var(--muted); margin: 0 6px; }
  .pair .val { font-variant-numeric: tabular-nums; font-weight: 700; font-size: 14px; }
  .pair .tag { font-size: 11px; text-transform: uppercase; letter-spacing: 0.04em; color: var(--muted); }
  .error { color: var(--risky); font-size: 14px; margin-top: 12px; }
  .hint { color: var(--muted); font-size: 13px; }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>Prompt-Risk Attribution</h1>
    <p>Classify how risky a prompt is and see which words drive the score, using Shapley interaction values.</p>
  </header>

  <div class="card">
    <textarea id="prompt" placeholder="Enter a prompt to analyze..."></textarea>
    <div class="row">
      <button id="go">Analyze</button>
      <label class="toggle">
        <input type="checkbox" id="explain" checked />
        Explain words (slower)
      </label>
      <label class="toggle" id="budget-wrap">
        Budget
        <select id="budget">
          <option value="64">64 (fastest)</option>
          <option value="128">128</option>
          <option value="256" selected>256</option>
          <option value="512">512 (most accurate)</option>
        </select>
      </label>
      <span class="hint" id="status"></span>
    </div>
    <div class="error hidden" id="error"></div>
  </div>

  <div class="card hidden" id="result">
    <div class="result-head">
      <span class="badge" id="badge"></span>
      <div class="prob"><span id="prob"></span><small> P(unsafe)</small></div>
    </div>
    <div class="bar"><div id="barfill"></div></div>

    <div id="attr" class="hidden">
      <p class="section-title">Word contributions</p>
      <div class="tokens" id="tokens"></div>
      <div class="legend">
        <span><i class="sw" style="background:#e0533d"></i> pushes toward risky</span>
        <span><i class="sw" style="background:#2ea66b"></i> pushes toward safe</span>
      </div>

      <div class="interactions hidden" id="interactions">
        <p class="section-title">Top interactions</p>
        <div id="pairs"></div>
        <div class="legend">
          <span><i class="sw" style="background:#e0533d"></i> reinforcing (more risk together than apart)</span>
          <span><i class="sw" style="background:#2ea66b"></i> offsetting (less risk together than apart)</span>
        </div>
      </div>
    </div>
  </div>
</div>

<script>
const $ = (id) => document.getElementById(id);

function cleanToken(t) {
  return t.replace(/^\\u0120/, "").replace(/^##/, "").replace(/^\\u2581/, "");
}

function colorFor(value, maxAbs) {
  if (maxAbs === 0) return "transparent";
  const intensity = Math.min(Math.abs(value) / maxAbs, 1);
  const alpha = 0.12 + intensity * 0.55;
  return value >= 0
    ? `rgba(224,83,61,${alpha})`   // risky
    : `rgba(46,166,107,${alpha})`; // safe
}

function renderAttribution(words) {
  const maxAbs = Math.max(...words.map((w) => Math.abs(w.shapley_value)), 0);
  const tokens = $("tokens");
  tokens.innerHTML = "";
  words.forEach((w) => {
    const el = document.createElement("span");
    el.className = "tok";
    el.textContent = w.word;
    el.style.background = colorFor(w.shapley_value, maxAbs);
    el.title = w.shapley_value.toFixed(4);
    tokens.appendChild(el);
    tokens.appendChild(document.createTextNode(" "));
  });
}

function renderInteractions(items) {
  const box = $("interactions");
  const list = $("pairs");
  list.innerHTML = "";
  if (!items || items.length === 0) {
    box.classList.add("hidden");
    return;
  }
  box.classList.remove("hidden");
  const maxAbs = Math.max(...items.map((it) => Math.abs(it.value)), 0);
  items.forEach((it) => {
    const positive = it.value >= 0;
    const color = positive ? "#e0533d" : "#2ea66b";
    const row = document.createElement("div");
    row.className = "pair";
    row.style.borderLeft = "4px solid " + colorFor(it.value, maxAbs);
    row.innerHTML =
      '<div class="terms"><b>' + cleanToken(it.tokens[0]) + '</b>' +
      '<span class="arrow">&#8644;</span><b>' + cleanToken(it.tokens[1]) + '</b></div>' +
      '<span class="tag">' + (positive ? "reinforcing" : "offsetting") + '</span>' +
      '<span class="val" style="color:' + color + '">' +
      (positive ? "+" : "") + it.value.toFixed(3) + '</span>';
    list.appendChild(row);
  });
}

function showResult(data, hasAttr) {
  $("result").classList.remove("hidden");
  const risky = data.label === "risky";
  const badge = $("badge");
  badge.textContent = data.label;
  badge.className = "badge " + data.label;
  const pct = (data.risk_probability * 100).toFixed(1);
  $("prob").textContent = pct + "%";
  const fill = $("barfill");
  fill.style.width = pct + "%";
  fill.style.background = risky ? "var(--risky)" : "var(--safe)";

  if (hasAttr && data.words) {
    $("attr").classList.remove("hidden");
    renderAttribution(data.words);
    renderInteractions(data.top_interactions);
  } else {
    $("attr").classList.add("hidden");
  }
}

async function analyze() {
  const prompt = $("prompt").value.trim();
  $("error").classList.add("hidden");
  if (!prompt) {
    $("error").textContent = "Please enter a prompt.";
    $("error").classList.remove("hidden");
    return;
  }
  const explain = $("explain").checked;
  $("go").disabled = true;
  $("status").textContent = explain ? "Computing attribution..." : "Classifying...";
  try {
    const endpoint = explain ? "/attribute" : "/predict";
    const payload = { prompt };
    if (explain) payload.budget = parseInt($("budget").value, 10);
    const res = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || ("Request failed (" + res.status + ")"));
    }
    showResult(await res.json(), explain);
  } catch (err) {
    $("error").textContent = err.message;
    $("error").classList.remove("hidden");
  } finally {
    $("go").disabled = false;
    $("status").textContent = "";
  }
}

function syncBudget() {
  const on = $("explain").checked;
  $("budget").disabled = !on;
  $("budget-wrap").style.opacity = on ? "1" : "0.4";
}
$("explain").addEventListener("change", syncBudget);
syncBudget();

$("go").addEventListener("click", analyze);
$("prompt").addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") analyze();
});
</script>
</body>
</html>
"""

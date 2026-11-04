/* RobustHFT-AgentLab live demo. Talks to scripts/demo_server.py (GET /api/setup, POST /api/analyze),
   which runs the experiments' own agent code. No libraries; every number shown comes from the server. */
(function () {
  "use strict";
  const $ = (s) => document.querySelector(s);
  const root = document.documentElement;
  const MINUS = "−";

  // ---------------------------------------------------------------- helpers
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
  const sg = (x, d = 2) => (x > 0 ? "+" : x < 0 ? MINUS : "") + Math.abs(x).toFixed(d);
  const pct = (x, d = 1) => (x > 0 ? "+" : x < 0 ? MINUS : "") + Math.abs(100 * x).toFixed(d) + "%";
  const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
  const pos = (x) => 50 + 50 * clamp(x, -1, 1); // [-1, 1] -> % across a meter
  const lines = (s) => s.split("\n").map((t) => t.trim()).filter(Boolean);
  const css = (name) => getComputedStyle(root).getPropertyValue(name).trim();
  const BIAS_NAMES = { none: "", overconfidence: "overconfidence", loss_aversion: "loss aversion" };

  // ---------------------------------------------------------------- theme (shared with the showcase)
  try { const t = localStorage.getItem("rhal-theme"); if (t) root.dataset.theme = t; } catch (e) { /* storage blocked */ }
  $("#theme-toggle").addEventListener("click", () => {
    const dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
    root.dataset.theme = dark ? "light" : "dark";
    try { localStorage.setItem("rhal-theme", root.dataset.theme); } catch (e) { /* storage blocked */ }
    if (state.setup) drawChart(); // the chart reads theme colours when drawn
  });

  const state = { setup: null, scenario: "rising", preset: null, last: null, busy: false };

  async function api(path, body) {
    const res = await fetch(path, body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {});
    let data = {};
    try { data = await res.json(); } catch (e) { /* non-JSON error page */ }
    if (!res.ok) throw new Error(data.error || "the demo server returned an error (" + res.status + ")");
    return data;
  }

  // ---------------------------------------------------------------- setup
  async function init() {
    try {
      state.setup = await api("/api/setup");
    } catch (e) {
      $("#offline").hidden = false;
      $("#run").disabled = true;
      $("#verdict").innerHTML = '<span class="muted">Start the demo server to use this page (see above).</span>';
      return;
    }
    renderPresets();
    renderScenarioTabs();
    selectScenario(state.scenario);
    $("#reader").addEventListener("change", readerHint);
    $("#run").addEventListener("click", run);
    for (const id of ["genuine", "injected"]) $("#" + id).addEventListener("input", () => markPreset(null));
    readerHint();
    applyPreset(state.setup.presets[0]);
  }

  function renderPresets() {
    const box = $("#presets");
    box.innerHTML = "";
    for (const p of state.setup.presets) {
      const b = document.createElement("button");
      b.className = "preset";
      b.type = "button";
      b.textContent = p.title;
      b.dataset.id = p.id;
      b.setAttribute("aria-pressed", "false");
      b.addEventListener("click", () => applyPreset(p));
      box.appendChild(b);
    }
    const own = document.createElement("button");
    own.className = "preset";
    own.type = "button";
    own.textContent = "Type your own";
    own.dataset.id = "own";
    own.setAttribute("aria-pressed", "false");
    own.addEventListener("click", () => {
      markPreset("own");
      $("#genuine").value = "";
      $("#injected").value = "";
      $("#preset-note").textContent = "Write one genuine headline and the attacker's version of it, pick a market day, then press Run. The agent trades the stock of Nordwind Systems (a fictional company), so write news about it; with free wording, use the local LLM reader.";
      $("#genuine").focus();
    });
    box.appendChild(own);
  }

  function markPreset(id) {
    state.preset = id;
    for (const b of document.querySelectorAll(".preset")) b.setAttribute("aria-pressed", String(b.dataset.id === id));
  }

  function applyPreset(p) {
    markPreset(p.id);
    $("#genuine").value = p.genuine.join("\n");
    $("#injected").value = p.injected.join("\n");
    $("#bias").value = p.bias || "none";
    $("#risk").value = "off";
    $("#reader").value = p.reader || "lexicon";
    readerHint();
    $("#preset-note").textContent = p.note;
    selectScenario(p.scenario);
    run();
  }

  function renderScenarioTabs() {
    const box = $("#scenarios");
    box.innerHTML = "";
    for (const s of state.setup.scenarios) {
      const b = document.createElement("button");
      b.type = "button";
      b.setAttribute("role", "tab");
      b.dataset.id = s.id;
      b.textContent = s.name;
      b.addEventListener("click", () => { selectScenario(s.id); if (state.last) run(); });
      box.appendChild(b);
    }
  }

  const scenario = () => state.setup.scenarios.find((s) => s.id === state.scenario);

  function selectScenario(id) {
    state.scenario = id;
    for (const b of document.querySelectorAll("#scenarios button")) b.setAttribute("aria-selected", String(b.dataset.id === id));
    drawChart();
    renderEvidence();
  }

  function drawChart() {
    const s = scenario();
    const svg = $("#spark");
    const W = 520, H = 150, L = 44, R = 12, T = 12, B = 24;
    const ys = s.chart.close, n = ys.length;
    const lo = Math.min(...ys), hi = Math.max(...ys), pad = (hi - lo) * 0.08 || 1;
    const x = (i) => L + (i * (W - L - R)) / Math.max(n - 1, 1);
    const y = (v) => T + ((hi + pad - v) * (H - T - B)) / (hi - lo + 2 * pad);
    const col = s.s_mkt > 0.15 ? css("--buy") : s.s_mkt < -0.15 ? css("--sell") : css("--hold");
    const pts = ys.map((v, i) => x(i).toFixed(1) + "," + y(v).toFixed(1)).join(" ");
    const area = `M${x(0)},${H - B} L` + ys.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" L") + ` L${x(n - 1)},${H - B} Z`;
    svg.innerHTML =
      `<path class="area" d="${area}" fill="${col}"/>` +
      `<polyline class="line" points="${pts}" stroke="${col}"/>` +
      `<line class="day" x1="${x(n - 1)}" x2="${x(n - 1)}" y1="${T - 4}" y2="${H - B}"/>` +
      `<circle class="dot" cx="${x(n - 1)}" cy="${y(ys[n - 1])}" r="5" fill="${col}"/>` +
      `<text x="${L - 6}" y="${y(hi) + 4}" text-anchor="end">${hi.toFixed(0)}</text>` +
      `<text x="${L - 6}" y="${y(lo) + 4}" text-anchor="end">${lo.toFixed(0)}</text>` +
      `<text x="${L}" y="${H - 6}">${esc(s.chart.dates[0])}</text>` +
      `<text x="${x(n - 1)}" y="${H - 6}" text-anchor="end">decision day ${esc(s.date)}</text>`;
  }

  function renderEvidence() {
    const s = scenario();
    const word = s.s_mkt > 0.15 ? "points up" : s.s_mkt < -0.15 ? "points down" : "is about neutral";
    const analog = s.n_analogs
      ? `<b>${pct(s.analog_mean_forward_return)}</b>average next-5-day move of the ${s.n_analogs} most similar past windows (${Math.round(100 * s.analog_hit_rate)}% rose)`
      : "<b>n/a</b>no similar past windows";
    $("#evidence").innerHTML =
      `<div class="ev main"><b>${sg(s.s_mkt)}</b>Market evidence used by the defense: the market ${word}. It is built only from prices up to this day.</div>` +
      `<div class="ev">${analog}</div>` +
      `<div class="ev"><b>${sg(s.basic_read)}</b>what the undefended agent sees instead: only the last return and drawdown</div>`;
  }

  function readerHint() {
    const llm = $("#reader").value === "llm";
    $("#reader-hint").innerHTML = llm
      ? `The local LLM (<code>${esc(state.setup.llm_model)}</code>) reads any wording. Start it first with <code>make llm-up</code>; the first answer can take a few seconds. It rates news for Nordwind Systems, the stock the agent trades, so news that names another company reads as neutral.`
      : "The paper's agent reads news with a fixed list of finance words (surged, plunged, upgraded, missed, ...). It counts “record” as positive and does not know “low”, so for free-form headlines such as “sales hit a record low” switch to the local LLM.";
  }

  // ---------------------------------------------------------------- run
  async function run() {
    if (state.busy) return;
    const body = {
      scenario: state.scenario,
      genuine: lines($("#genuine").value),
      injected: lines($("#injected").value),
      reader: $("#reader").value,
      bias: $("#bias").value,
      risk: $("#risk").value,
    };
    const err = $("#err");
    err.hidden = true;
    state.busy = true;
    const btn = $("#run");
    btn.disabled = true;
    btn.textContent = body.reader === "llm" ? "Reading the news with the local LLM ..." : "Running ...";
    try {
      const r = await api("/api/analyze", body);
      state.last = r;
      render(r);
    } catch (e) {
      err.textContent = e.message;
      err.hidden = false;
      state.last = null;
      $("#verdict").className = "verdict-box";
      $("#verdict").innerHTML = '<span class="muted">This run could not complete: see the message under the Run button.</span>';
      $("#pairs").innerHTML = "";
      $("#heads-card").hidden = true;
    } finally {
      state.busy = false;
      btn.disabled = false;
      btn.textContent = "▶ Run both agents";
    }
  }

  // ---------------------------------------------------------------- results
  function render(r) {
    renderVerdict(r);
    renderPairs(r);
    renderHeadlines(r);
    renderHow(r);
  }

  function renderVerdict(r) {
    const n = r.pairs.none, d = r.pairs.defended, da = d.attacked;
    const box = $("#verdict");
    const hold = da.overrides.some((o) => o.startsWith("hold_on_conflict"));
    const shrink = da.overrides.some((o) => o.startsWith("shrink_on_conflict"));
    const anyFake = r.headlines.some((h) => h.role === "injected");
    let cls = "", h = "", p = [];
    if (!anyFake) {
      h = "No fake headline: both agents read the same news.";
      p.push("Add a fake headline to see whether it flips the decision.");
    } else if (n.flipped && !d.flipped) {
      cls = "good";
      h = `The fake news flipped the undefended agent (${n.clean.final} → ${n.attacked.final}); the defense layer kept ${d.clean.final}.`;
      if (da.conflict) p.push(`It spotted the conflict: the attacked agent's combined news signal (${sg(da.s_text)}) disagrees with the market evidence (${sg(da.s_mkt)}), so their consistency is ${da.consistency.toFixed(2)}, below ${r.params.conflict_threshold}. Grounding pulled the decision back toward the market.`);
      else p.push(`The defended agent already follows the market evidence (${sg(da.s_mkt)}): its clean twin decided ${d.clean.final} without the fake, and the fake headline pushes the same way, so it changes nothing. There was no conflict to spot.`);
      if (shrink) p.push("The risk layer shrank the trade instead of cancelling it, so the decision stayed the same.");
    } else if (n.flipped && d.flipped) {
      cls = "bad";
      h = `The fake news flipped both agents (without defense ${n.clean.final} → ${n.attacked.final}, with defense ${d.clean.final} → ${d.attacked.final}).`;
      if (hold) p.push("The hold-on-conflict rule fired because the news contradicts the market, and that changed the decision: the attacker controls when the rule fires. This is the backfire the paper reports. Try “Shrink on conflict”.");
      else if (!da.conflict) p.push(`The news did not contradict the market evidence enough to be discounted (consistency ${da.consistency.toFixed(2)} ≥ ${r.params.conflict_threshold}). With no clear market signal, grounding has nothing to check the news against.`);
      else p.push(`The conflict was spotted (consistency ${da.consistency.toFixed(2)}) and grounding pulled the signal toward the market, but not far enough: grounding removes at most ${Math.round(100 * r.params.kappa)}% of the gap.`);
    } else if (!n.flipped && d.flipped) {
      cls = "bad";
      h = `Only the defended agent flipped (${d.clean.final} → ${d.attacked.final}).`;
      if (hold) p.push("The hold-on-conflict rule changed its decision: a defense that reacts to conflict hands the attacker a lever. Try “Shrink on conflict”.");
    } else {
      h = "The fake news did not change either decision.";
      const gen = r.headlines.filter((x) => x.role === "genuine"), fake = r.headlines.filter((x) => x.role === "injected");
      const same = fake.every((f) => gen.some((g) => Math.abs(g.score - f.score) < 0.05));
      const neutral = r.headlines.every((x) => Math.abs(x.score) < 0.05);
      if (r.reader === "llm" && neutral) p.push("The LLM read every headline as neutral. It rates news for Nordwind Systems, the stock the agent trades: name that company in your headlines.");
      else if (r.reader === "lexicon" && same) p.push(`The word list read the fake headline the same way as the genuine one (${sg(fake[0].score)}): it does not understand this wording (see the counted words below). Switch the news reader to the local LLM, which reads any wording.`);
      else p.push("Its effect on the agents' signals was too small to cross the decision threshold.");
    }
    box.className = "verdict-box " + cls;
    box.innerHTML = `<h3>${esc(h)}</h3>` + p.filter(Boolean).map((t) => `<p>${esc(t)}</p>`).join("");
  }

  function meter(agent, twin, threshold) {
    const z = pos(-threshold), w = pos(threshold) - z;
    return `<div class="meter"><div class="meter-track">` +
      `<div class="meter-hold" style="left:${z}%;width:${w}%"></div><div class="meter-zero"></div>` +
      `<div class="needle twin" style="left:${pos(twin.s_final)}%" title="clean twin's decision signal ${sg(twin.s_final)}"></div>` +
      `<div class="needle text" style="left:${pos(agent.s_text)}%" title="attacked agent's news signal ${sg(agent.s_text)}"></div>` +
      `<div class="needle market" style="left:${pos(agent.s_mkt)}%" title="market signal ${sg(agent.s_mkt)}"></div>` +
      `<div class="needle final" style="left:${pos(agent.s_final)}%" title="attacked agent's decision signal ${sg(agent.s_final)}"></div>` +
      `</div><div class="meter-scale"><span>Sell ${MINUS}1</span><span>Hold</span><span>+1 Buy</span></div></div>`;
  }

  function pairCard(key, pr, r) {
    const a = pr.attacked, c = pr.clean, def = key === "defended";
    const riskName = { off: "", hold: " + risk layer (hold on conflict)", shrink: " + risk layer (shrink on conflict)" }[r.risk];
    const title = def ? "With the defense layer" : "Without defense";
    const sub = def ? "Market evidence + temporal grounding" + riskName : "Reads the news and a crude price read";
    let html = `<div class="pair ${def ? "def" : ""}"><h3><span>${esc(title)}</span>` +
      `<span class="flag ${pr.flipped ? "flip" : "ok"}">${pr.flipped ? "⚡ flipped" : "✓ not flipped"}</span></h3>` +
      `<div class="sub">${esc(sub)}</div>` +
      `<div class="who"><span>Clean twin (genuine news only)</span><span class="act ${c.final}">${c.final}</span>` +
      `<span>Attacked agent (genuine + fake${r.bias !== "none" ? ", " + BIAS_NAMES[r.bias] + " prompt" : ""})</span><span class="act ${a.final}">${a.final}</span></div>` +
      meter(a, c, a.threshold) +
      `<div class="meter-legend"><span><i style="background:var(--s1)"></i>news</span><span><i style="background:var(--s2);border-radius:2px"></i>market</span><span><i style="background:var(--s3);border-radius:2px"></i>decision</span><span><i style="border:2px dashed var(--ink-2)"></i>clean twin</span></div>` +
      `<div class="nums">` +
      `<div class="num-cell"><b>${sg(a.s_text)}</b>news signal</div>` +
      `<div class="num-cell"><b>${sg(a.s_mkt)}</b>${def ? "market evidence" : "crude price read"}</div>`;
    if (def) html += `<div class="num-cell ${a.conflict ? "warn" : ""}"><b>${a.consistency.toFixed(2)}</b>news–market consistency${a.conflict ? " ⚠" : ""}</div>`;
    html += `<div class="num-cell"><b>${sg(a.s_final)}</b>decision signal (act beyond ±${a.threshold.toFixed(2)})</div></div>`;
    if (def && a.conflict) {
      html += `<div class="callout conflict">⚠ <b>Conflict spotted.</b> The news says ${sg(a.s_text)}, the market evidence says ${sg(a.s_mkt)}. Grounding moved the signal from ${sg(a.s_fused)} toward the market, to ${sg(a.s_final)}.</div>`;
    }
    for (const o of a.overrides) {
      const name = o.startsWith("hold_on_conflict") ? "Hold on conflict: the trade was cancelled" : o.startsWith("shrink_on_conflict") ? "Shrink on conflict: the trade was kept but made smaller" : o.startsWith("low_confidence") ? "Low confidence: the trade was cancelled" : o;
      html += `<div class="callout risk">Risk layer: ${esc(name)}.</div>`;
    }
    return html + "</div>";
  }

  function renderPairs(r) {
    $("#pairs").innerHTML = pairCard("none", r.pairs.none, r) + pairCard("defended", r.pairs.defended, r);
  }

  function renderHeadlines(r) {
    $("#heads-card").hidden = false;
    const s = scenario();
    let html = "";
    for (const h of r.headlines) {
      const col = h.score >= 0 ? "var(--buy)" : "var(--sell)";
      const left = Math.min(pos(h.score), 50), width = Math.abs(pos(h.score) - 50);
      const contra = h.agreement < r.params.conflict_threshold;
      html += `<div class="head"><div class="head-top"><span class="role ${h.role}">${h.role === "genuine" ? "genuine" : "fake"}</span><span class="head-text">${esc(h.text)}</span></div>` +
        `<div class="head-metrics">` +
        `<div class="metric">Read as <b>${sg(h.score)}</b><div class="mini"><span class="mid"></span><i style="left:${left}%;width:${width}%;background:${col}"></i></div></div>` +
        `<div class="metric">Agreement with the market <b>${h.agreement.toFixed(2)}</b>${contra ? " ⚠ contradicts it" : ""}<div class="mini"><i style="left:0;width:${100 * h.agreement}%;background:${contra ? "var(--serious)" : "var(--good)"}"></i></div></div>` +
        `<div class="metric">Text-only fake-news detector <b>${h.detector.score.toFixed(2)}</b> ${h.detector.flagged ? "⚑ flagged" : "not flagged"} <span class="muted">(flags above ${r.params.detector_threshold.toFixed(2)})</span></div>` +
        `</div>`;
      if (r.reader === "lexicon") {
        html += h.words.length
          ? `<div class="words">${h.words.map((w) => `<span class="word ${w.weight >= 0 ? "pos" : "neg"}">${esc(w.phrase)} ${sg(w.weight)}${w.notes.length ? ` <em>${esc(w.notes.join(", "))}</em>` : ""}</span>`).join("")}</div>`
          : `<div class="reader-tip">No finance words recognized, so the paper's agent reads this headline as neutral (0). Try the local LLM reader.</div>`;
      }
      html += "</div>";
    }
    const injected = r.headlines.filter((h) => h.role === "injected");
    if (injected.length && injected.every((h) => !h.detector.flagged)) {
      html += `<div class="reader-tip">The text-only detector does not flag the fake: a small edit leaves the wording looking normal. The defense instead checks the news against the market (${sg(s.s_mkt)} on this day).</div>`;
    }
    $("#heads").innerHTML = html;
  }

  function renderHow(r) {
    const n = r.pairs.none.attacked, d = r.pairs.defended.attacked, p = r.params;
    const lam = p.text_weight, k = p.kappa, th = d.threshold;
    $("#how-steps").innerHTML = [
      `<b>Read the news.</b> Each headline gets a score from ${MINUS}1 (bearish) to +1 (bullish); the agent averages them into its news signal.` +
        `<span class="eq">news signal = ${sg(d.s_text)}${r.bias !== "none" ? "   (after the " + BIAS_NAMES[r.bias] + " prompt)" : ""}</span>`,
      `<b>Read the market.</b> Without defense the agent only glances at the last return and drawdown. With the defense it uses the market evidence: price features plus how similar past windows ended, from prices up to this day only.` +
        `<span class="eq">crude read = ${sg(n.s_mkt)}      market evidence = ${sg(d.s_mkt)}</span>`,
      `<b>Combine.</b> Like an LLM that leans on the news, the agent gives the news ${Math.round(100 * lam)}% of the weight.` +
        `<span class="eq">s = ${lam}·news + ${(1 - lam).toFixed(2)}·market = ${sg(n.s_fused)} (no defense), ${sg(d.s_fused)} (defense)</span>`,
      `<b>Ground (defense only).</b> Consistency measures how well news and market agree; inconsistent news is pulled toward the market by up to κ = ${k}.` +
        `<span class="eq">c = 1 ${MINUS} |news ${MINUS} market| / 2 = ${d.consistency.toFixed(2)}</span>` +
        `<span class="eq">s_g = s ${MINUS} κ·(1 ${MINUS} c)·(s ${MINUS} market) = ${sg(d.s_final)}</span>`,
      `<b>Decide.</b> Buy if the signal is above +${th.toFixed(2)}, Sell if below ${MINUS}${th.toFixed(2)}, otherwise Hold.` +
        `<span class="eq">no defense: ${sg(n.s_final)} → ${n.decision}      defense: ${sg(d.s_final)} → ${d.decision}</span>`,
      `<b>Risk layer (optional).</b> When consistency is below ${p.conflict_threshold}, the original rule holds; the fix keeps the decision but moves only a fraction c of the way to the new position.` +
        `<span class="eq">${r.risk === "off" ? "off in this run" : esc(d.overrides.join(", ") || "no rule fired") + " → " + d.final}</span>`,
    ].map((t) => `<li>${t}</li>`).join("");
  }

  init();
})();

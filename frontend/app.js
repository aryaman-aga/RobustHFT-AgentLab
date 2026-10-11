/* RobustHFT-AgentLab showcase. Every number comes from frontend/data.js
   (written by scripts/export_frontend_data.py); nothing here is hand-typed. */
(() => {
  "use strict";
  const D = window.RHAL_DATA;
  const $ = (s, r = document) => r.querySelector(s);
  const NS = "http://www.w3.org/2000/svg";
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const root = document.documentElement;

  function svg(tag, attrs, parent) {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs || {}) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  function h(tag, attrs, html) {
    const e = document.createElement(tag);
    for (const k in attrs || {}) k === "class" ? (e.className = attrs[k]) : e.setAttribute(k, attrs[k]);
    if (html != null) e.innerHTML = html;
    return e;
  }
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const fmt = (x, d = 1) => (x == null ? "–" : x.toFixed(d));
  const signed = (x, d = 1) => (x == null ? "–" : (x > 0 ? "+" : x < 0 ? "−" : "") + Math.abs(x).toFixed(d));
  const ciTxt = (c) => (c ? `[${signed(c.lo)}, ${signed(c.hi)}]` : "");
  const pTxt = (c) => (c ? (c.p < 0.001 ? "p < 0.001" : `p = ${c.p.toFixed(3)}`) : "");
  const css = (v) => getComputedStyle(root).getPropertyValue(v).trim();

  if (!D) {
    document.body.insertAdjacentHTML("afterbegin", '<p style="padding:16px">frontend/data.js is missing. Run <code>make frontend</code>.</p>');
    return;
  }
  const RP = D.replay;
  const R = D.results;
  const N = RP.days;

  // ---------------------------------------------------------------- theme
  try {
    const t = localStorage.getItem("rhal-theme");
    if (t) root.dataset.theme = t;
  } catch (e) { /* storage unavailable: follow the OS */ }
  $("#theme-toggle").addEventListener("click", () => {
    const dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
    root.dataset.theme = dark ? "light" : "dark";
    try { localStorage.setItem("rhal-theme", root.dataset.theme); } catch (e) { /* ignore */ }
    document.dispatchEvent(new CustomEvent("rhal-theme"));
  });

  // ---------------------------------------------------------------- reveal + count-up
  const revealIO = new IntersectionObserver((entries) => entries.forEach((en) => {
    if (!en.isIntersecting) return;
    en.target.classList.add("in");
    en.target.dispatchEvent(new CustomEvent("revealed"));
    revealIO.unobserve(en.target);
  }), { threshold: 0.15 });
  document.querySelectorAll(".reveal").forEach((el) => revealIO.observe(el));
  function onVisible(el, fn, threshold = 0.25) {
    const io = new IntersectionObserver((es) => es.forEach((en) => { if (en.isIntersecting) { fn(); io.disconnect(); } }), { threshold });
    io.observe(el);
  }
  function countUp(el, to, { dec = 1, prefix = "", suffix = "", dur = 1400 } = {}) {
    if (to == null) { el.textContent = "–"; return; }
    if (reduced) { el.textContent = prefix + to.toFixed(dec) + suffix; return; }
    const t0 = performance.now();
    const step = (t) => {
      const k = Math.min(1, (t - t0) / dur);
      el.textContent = prefix + (to * (1 - Math.pow(1 - k, 3))).toFixed(dec) + suffix;
      if (k < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }
  const flips = (arm, upto = N) => RP.arms[arm].slice(0, upto).filter((s) => s.a !== s.s).length;

  // ---------------------------------------------------------------- hero
  (function heroTiles() {
    const box = $("#hero-tiles");
    const tiles = [
      { v: R ? R.asr.naive : (100 * flips("attack")) / N, s: "%", t: "of the undefended agent's decisions flip under attack" },
      { v: R && R.decomposition.meaning ? R.decomposition.meaning.mean : null, s: " pp", t: "of those flips are caused by what the fake news actually says" },
      { v: R && R.gain.grounding_adaptive ? R.gain.grounding_adaptive.mean : null, s: " pp", pre: "−", t: "misinformation-driven flips with temporal grounding" },
    ];
    tiles.forEach((tl, i) => {
      const b = h("b", {}, "0");
      const d = h("div", { class: "tile" });
      d.append(b, h("span", {}, tl.t));
      box.append(d);
      setTimeout(() => countUp(b, tl.v, { prefix: tl.pre || "", suffix: tl.s }), 300 + i * 200);
    });
  })();

  (function heroCanvas() {
    const cv = $("#hero-canvas");
    const ctx = cv.getContext("2d");
    let W = 0, H = 0, running = !reduced, pulse = 0, lastSpawn = 0;
    const parts = [], emits = [];
    function resize() {
      const dpr = Math.min(2, window.devicePixelRatio || 1);
      W = cv.clientWidth; H = cv.clientHeight;
      cv.width = W * dpr; cv.height = H * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
    resize();
    window.addEventListener("resize", resize);
    const agentAt = () => ({ x: W * (W > 900 ? 0.84 : 0.9), y: H * 0.46 });
    function spawn(x) {
      const y = H * (0.1 + Math.random() * 0.8);
      parts.push({ x: x == null ? -40 : x, y0: y, v: 0.8 + Math.random() * 1.3, syn: Math.random() < 0.35, w: 26 + Math.random() * 10, j: 0 });
    }
    for (let i = 0; i < 16; i++) spawn(Math.random() * W * 0.75);
    function roundRect(x, y, w, hgt, r) {
      ctx.beginPath();
      ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + hgt, r); ctx.arcTo(x + w, y + hgt, x, y + hgt, r);
      ctx.arcTo(x, y + hgt, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
    }
    function frame(t) {
      if (!running) return;
      ctx.clearRect(0, 0, W, H);
      const a = agentAt(), ink = css("--muted"), red = css("--crit"), blue = css("--s1"), buy = css("--buy"), sell = css("--sell");
      if (t - lastSpawn > 260) { spawn(); lastSpawn = t; }
      // agent node
      pulse *= 0.93;
      const r = 24 + pulse * 9;
      const g = ctx.createRadialGradient(a.x, a.y, 2, a.x, a.y, r * 3.2);
      g.addColorStop(0, blue + "66"); g.addColorStop(1, blue + "00");
      ctx.fillStyle = g; ctx.beginPath(); ctx.arc(a.x, a.y, r * 3.2, 0, 6.283); ctx.fill();
      ctx.strokeStyle = blue; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(a.x, a.y, r, 0, 6.283); ctx.stroke();
      ctx.fillStyle = blue; ctx.font = "600 12px system-ui, sans-serif"; ctx.textAlign = "center"; ctx.fillText("agent", a.x, a.y + 4);
      // news cards flowing in
      for (let i = parts.length - 1; i >= 0; i--) {
        const p = parts[i];
        p.x += p.v * 1.6;
        const k = clamp(p.x / (a.x - 30), 0, 1);
        const y = p.y0 + (a.y - p.y0) * k * k * (3 - 2 * k);
        const glitch = p.syn && k > 0.35 ? (Math.random() - 0.5) * 3 : 0;
        ctx.globalAlpha = W > 900 && p.x < W * 0.56 ? 0.1 + 0.12 * k : 0.2 + 0.6 * k;  // faint behind the hero text
        ctx.strokeStyle = p.syn && k > 0.3 ? red : ink; ctx.lineWidth = 1.5;
        roundRect(p.x + glitch, y - 9, p.w, 18, 4); ctx.stroke();
        ctx.fillStyle = p.syn && k > 0.3 ? red : ink;
        ctx.fillRect(p.x + 5 + glitch, y - 4, p.w * 0.6, 2.2); ctx.fillRect(p.x + 5 + glitch, y + 1.5, p.w * 0.4, 2.2);
        ctx.globalAlpha = 1;
        if (p.x >= a.x - 34) {
          parts.splice(i, 1);
          pulse = 1;
          const flipped = p.syn && Math.random() < 0.45;
          const act = Math.random();
          emits.push({ x: a.x + r + 6, y: a.y + (Math.random() - 0.5) * 30, life: 1, c: flipped ? css("--serious") : act < 0.4 ? buy : act < 0.75 ? sell : ink,
            g: flipped ? "⚡" : act < 0.4 ? "▲" : act < 0.75 ? "▼" : "●" });
        }
      }
      // decisions flowing out
      ctx.font = "700 14px system-ui, sans-serif";
      for (let i = emits.length - 1; i >= 0; i--) {
        const e = emits[i];
        e.x += 1.2; e.life -= 0.012;
        if (e.life <= 0) { emits.splice(i, 1); continue; }
        ctx.globalAlpha = e.life; ctx.fillStyle = e.c; ctx.fillText(e.g, e.x, e.y);
      }
      ctx.globalAlpha = 1;
      requestAnimationFrame(frame);
    }
    const io = new IntersectionObserver((es) => es.forEach((en) => {
      const vis = en.isIntersecting && !reduced;
      if (vis && !running) { running = true; requestAnimationFrame(frame); } else if (!vis) running = false;
    }));
    io.observe(cv);
    if (running) requestAnimationFrame(frame);
  })();

  // ---------------------------------------------------------------- pipeline
  (function pipeline() {
    const S = $("#pipeline-svg");
    const W = 128, Hn = 56;
    const nodes = {
      newsA: [16, 40, "Clean news", "5 items a day"], misgen: [176, 40, "MisGen attacker", "edit · rewrite · bias"],
      retrA: [336, 40, "Retrieval", "top-4, last 3 days"], agentA: [496, 40, "Agent", "reads + remembers"],
      groundA: [656, 40, "Grounding", "check vs market"], riskA: [816, 40, "Risk layer", "gates & sizing"], decA: [976, 40, "Decision", "▲ buy ▼ sell ● hold"],
      market: [336, 172, "Market data", "prices & volume"], features: [496, 172, "Evidence", "trend · analogs"], same: [976, 172, "Same decision?", "no → ⚡ flip"],
      newsB: [16, 304, "Clean news", "same items"], retrB: [336, 304, "Retrieval", "clean pool only"], agentB: [496, 304, "Clean twin", "identical agent"],
      groundB: [656, 304, "Grounding", "check vs market"], riskB: [816, 304, "Risk layer", "gates & sizing"], decB: [976, 304, "Decision", "▲ buy ▼ sell ● hold"],
    };
    const cx = (id) => nodes[id][0] + W / 2, cy = (id) => nodes[id][1] + Hn / 2;
    const edges = {
      "newsA-misgen": ["h"], "misgen-retrA": ["h", true], "retrA-agentA": ["h", true], "agentA-groundA": ["h", true], "groundA-riskA": ["h"], "riskA-decA": ["h"],
      "newsB-retrB": ["h"], "retrB-agentB": ["h"], "agentB-groundB": ["h"], "groundB-riskB": ["h"], "riskB-decB": ["h"],
      "market-features": ["h"], "features-groundA": ["up"], "features-groundB": ["down"], "decA-same": ["vdown"], "decB-same": ["vup"],
    };
    function pathFor(id) {
      const [a, b] = id.split("-"), kind = edges[id][0];
      if (kind === "h") return `M ${nodes[a][0] + W} ${cy(a)} L ${nodes[b][0]} ${cy(b)}`;
      if (kind === "up") return `M ${cx(a)} ${nodes[a][1]} C ${cx(a)} ${nodes[a][1] - 40}, ${cx(b)} ${nodes[b][1] + Hn + 40}, ${cx(b)} ${nodes[b][1] + Hn}`;
      if (kind === "down") return `M ${cx(a)} ${nodes[a][1] + Hn} C ${cx(a)} ${nodes[a][1] + Hn + 40}, ${cx(b)} ${nodes[b][1] - 40}, ${cx(b)} ${nodes[b][1]}`;
      if (kind === "vdown") return `M ${cx(a)} ${nodes[a][1] + Hn} L ${cx(b)} ${nodes[b][1]}`;
      return `M ${cx(a)} ${nodes[a][1]} L ${cx(b)} ${nodes[b][1] + Hn}`;
    }
    const lanes = [["Attacked agent", 30], ["Market evidence", 162], ["Clean twin (control)", 294]];
    lanes.forEach(([t, y]) => { const e = svg("text", { x: 16, y, class: "lane-label" }, S); e.textContent = t; });
    const edgeEls = {};
    Object.keys(edges).forEach((id) => { edgeEls[id] = svg("path", { d: pathFor(id), class: "pedge" + (edges[id][1] ? " attack" : "") }, S); });
    const nodeEls = {};
    Object.entries(nodes).forEach(([id, [x, y, title, sub]]) => {
      const g = svg("g", { class: "pnode" + (id === "misgen" ? " attack" : ""), tabindex: 0, role: "button", "aria-label": title }, S);
      svg("rect", { x, y, width: W, height: Hn, rx: 12 }, g);
      const t1 = svg("text", { x: x + 12, y: y + 24 }, g); t1.textContent = title;
      const t2 = svg("text", { x: x + 12, y: y + 42, class: "sub" }, g); t2.textContent = sub;
      nodeEls[id] = g;
    });
    const k = RP.threshold;
    const steps = [
      [["newsA", "newsB"], [], "Clean news arrives", "Each simulated day publishes five template news items about a fictitious company. Their tone is genuinely (but noisily) related to the next day's return, so reading news is useful."],
      [["misgen"], ["newsA-misgen"], "The attacker edits it", "MisGen copies today's items and makes small, plausible edits: it flips a direction, inflates a number, swaps a cause or a company, and sometimes rewrites the style. Edits that look too different or too detectable are rejected. It also slips a behavioural-bias prompt (e.g. overconfidence) into the agent's instructions, and learns from which edits caused flips."],
      [["retrA", "retrB"], ["misgen-retrA", "newsB-retrB"], "Each agent retrieves its news", "Both agents fetch the 4 most relevant recent articles. The attacked agent's pool now mixes genuine and synthetic items; the clean twin's does not."],
      [["agentA", "agentB"], ["retrA-agentA", "retrB-agentB"], "The agent forms a view", "The agent scores the articles' sentiment, applies its memory and any bias prompt, and blends the text signal with what it sees in the price chart. Text dominates, just like an LLM agent that overreacts to news."],
      [["market", "features"], ["market-features"], "Market evidence is gathered", "Trend, volatility, drawdown, microstructure proxies and the outcomes of similar past market windows. Only windows whose outcome is already known count (e + H ≤ t), so nothing peeks into the future."],
      [["groundA", "groundB"], ["agentA-groundA", "agentB-groundB", "features-groundA", "features-groundB"], "Grounding checks the story", `Text that contradicts the market evidence is pulled toward it: s_g = s − κ(1 − c)(s − s_mkt). Consistent text passes untouched; a confident fake story that the market does not support loses weight.`],
      [["riskA", "riskB"], ["groundA-riskA", "groundB-riskB"], "The risk layer can veto", "A confidence gate, hold-on-conflict, volatility-targeted sizing, stop-loss / take-profit and a drawdown throttle. It can only shrink or close positions, never open new ones."],
      [["decA", "decB"], ["riskA-decA", "riskB-decB"], "Decide and trade", `Buy, sell or hold. The agent acts when its final signal leaves the ±${k} hold zone, and orders fill at the next day's open, so the decision cannot use today's close twice.`],
      [["same"], ["decA-same", "decB-same"], "Did the fake news change anything?", "Compare the two decisions. If they differ, that day is a flip. ASR is the share of flipped days. Net ASR subtracts the flips that harmless paraphrases also cause, which leaves only what the misinformation's content did."],
    ];
    let cur = -1, hold = 0;
    const dots = [];
    function travel(pathEl, red) {
      if (reduced) return;
      const len = pathEl.getTotalLength();
      const dot = svg("circle", { r: 5, class: "pdot" + (red ? " red" : "") }, S);
      const t0 = performance.now();
      dots.push(dot);
      (function mv(t) {
        const f = Math.min(1, (t - t0) / 900);
        const p = pathEl.getPointAtLength(len * f);
        dot.setAttribute("cx", p.x); dot.setAttribute("cy", p.y);
        if (f < 1) requestAnimationFrame(mv); else { dot.remove(); }
      })(t0);
    }
    function go(i) {
      cur = (i + steps.length) % steps.length;
      const [ns, es, title, text] = steps[cur];
      Object.values(nodeEls).forEach((g) => g.classList.remove("active"));
      Object.values(edgeEls).forEach((e) => e.classList.remove("flow"));
      ns.forEach((id) => nodeEls[id].classList.add("active"));
      es.forEach((id) => { edgeEls[id].classList.add("flow"); travel(edgeEls[id], edges[id][1] || id === "misgen-retrA"); });
      $("#pipe-step").textContent = cur + 1;
      $("#pipe-title").textContent = title;
      $("#pipe-text").textContent = text;
    }
    Object.entries(nodeEls).forEach(([id, g]) => {
      const pick = () => { const i = steps.findIndex((s) => s[0].includes(id)); if (i >= 0) { go(i); hold = performance.now() + 9000; } };
      g.addEventListener("click", pick);
      g.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick(); } });
    });
    go(0);
    let visible = false;
    new IntersectionObserver((es) => es.forEach((en) => { visible = en.isIntersecting; })).observe(S);
    setInterval(() => { if (visible && !reduced && performance.now() > hold) go(cur + 1); }, 2600);
  })();

  // ---------------------------------------------------------------- replay
  const MINOR = { flipping: "direction flipped", sentiment: "tone exaggerated", numerical: "number distorted", temporal: "date shifted",
    entity: "company swapped", concept: "metric swapped", causal: "cause swapped", other: "vague attribution" };
  const REWRITE = { news_style: "news-style rewrite", academic_style: "academic rewrite" };
  const BIAS = { overconfidence: "overconfidence", loss_aversion: "loss aversion", anchoring: "anchoring", herding: "herding", confirmation: "confirmation bias" };
  const VETO = { hold_on_conflict: "Risk layer held: the text contradicts the market", low_confidence: "Risk layer held: confidence too low",
    stop_loss: "Stop-loss exit", take_profit: "Take-profit exit", dd_throttle: "Position halved: drawdown limit reached" };

  const gainRisk = R && R.gain.risk_on_top;
  const riskLine = gainRisk
    ? (gainRisk.sig && gainRisk.mean < 0
      ? `Across ${R.n_seeds} simulated markets it slightly <b>increases</b> misinformation-driven flips (${signed(-gainRisk.mean)} pp, 95% CI ${ciTxt({ lo: -gainRisk.hi, hi: -gainRisk.lo })}).`
      : gainRisk.sig ? `Across ${R.n_seeds} markets it lowers misinformation-driven flips by ${fmt(gainRisk.mean)} pp ${ciTxt(gainRisk)}.`
        : `Across ${R.n_seeds} markets its effect on misinformation-driven flips is not detectable (${signed(gainRisk.mean)} pp ${ciTxt(gainRisk)}).`)
    : "";
  const g = R && R.gain.grounding_adaptive;
  const SCEN = [
    { tab: "1 · The attack", left: ["clean", "Without", "Clean news only"], right: ["attack", "With", "Fake news injected"],
      desc: "Same agent, same market, same days. On the right, the attacker slips synthetic articles (red cards) into the agent's news feed. Every ⚡ is a day the attacked agent decided differently from its clean twin.",
      take: R ? `Across ${R.n_seeds} simulated markets, ${fmt(R.asr.naive)}% of the undefended agent's decisions flip.` : "" },
    { tab: "2 · Placebo check", left: ["placebo", "Without", "Harmless paraphrases"], right: ["attack", "With", "Fake news"],
      desc: "Adding <i>any</i> extra articles reshuffles what the agent retrieves. The left panel injects meaning-preserving paraphrases (amber cards) instead of misinformation, so its flips are pure noise. Only the difference between the panels is caused by the fake news.",
      take: R ? `Across ${R.n_seeds} markets, paraphrases alone flip ${fmt(R.asr.placebo)}% of decisions; the misinformation's content adds ${fmt(R.decomposition.meaning && R.decomposition.meaning.mean)} pp.` : "" },
    { tab: "3 · Temporal grounding", left: ["attack", "Without", "No defence"], right: ["grounded", "With", "Temporal grounding"],
      desc: "Both agents read the exact same recorded attack. Grounding checks each story against market evidence (trends, microstructure, similar past windows) and shrinks text that contradicts it. Watch the green final-signal bar stay closer to the orange market marker.",
      take: g ? `Across ${R.n_seeds} markets, grounding cuts misinformation-driven flips by ${fmt(g.mean)} pp (95% CI ${ciTxt(g)}), at a cost of ${fmt(R.cost.grounding && R.cost.grounding.mean)} pp of clean return.` : "" },
    { tab: "4 · Risk layer", left: ["grounded", "Without", "Grounding only"], right: ["grounded_risk", "With", "Grounding + risk layer"],
      desc: "Same recorded attack again. The risk layer can veto trades (🛡): hold when text and market conflict, a confidence gate, stops and a drawdown throttle. Do the vetoes actually prevent flips?",
      take: riskLine },
  ];

  const tip = $("#tooltip");
  function showTip(html, x, y) {
    tip.innerHTML = html; tip.classList.add("show");
    const r = tip.getBoundingClientRect();
    tip.style.left = clamp(x + 14, 8, window.innerWidth - r.width - 8) + "px";
    tip.style.top = clamp(y - r.height - 12, 8, window.innerHeight - r.height - 8) + "px";
  }
  const hideTip = () => tip.classList.remove("show");
  const glyph = (a) => (a === "Buy" ? "▲ Buy" : a === "Sell" ? "▼ Sell" : "● Hold");

  class Panel {
    constructor() {
      this.el = h("article", { class: "panel" });
      this.el.innerHTML = `
        <div class="panel-head"><div><span class="ptag"></span><h3></h3></div>
          <div class="kpis"><div><b class="kpi-flips">0</b><span>flips</span></div><div><b class="kpi-asr">0%</b><span>ASR</span></div><div><b class="kpi-ret">+0.0%</b><span>return</span></div></div></div>
        <svg class="chart" viewBox="0 0 600 236" role="img"></svg>
        <div class="meter"><div class="meter-track"><div class="meter-zone"></div><div class="meter-zero"></div>
          <div class="needle market" title="market signal"></div><div class="needle text" title="text signal"></div><div class="needle final" title="final signal"></div></div>
          <div class="meter-scale"><span>−1 · sell</span><span>hold zone</span><span>buy · +1</span></div>
          <div class="meter-legend"><span><i style="background:var(--s1)"></i>text signal</span><span><i style="background:var(--s2);border-radius:2px"></i>market signal</span><span><i style="background:var(--s3);border-radius:2px;width:5px"></i>final signal</span></div></div>
        <div class="feed"><div class="feed-head"><span>Retrieved news today</span><span class="bias-badge" hidden></span></div><ul class="cards"></ul></div>
        <div class="veto" hidden></div>`;
      this.svg = $(".chart", this.el);
      const k = RP.threshold;
      Object.assign($(".meter-zone", this.el).style, { left: `${((1 - k) / 2) * 100}%`, width: `${k * 100}%` });
      this.needles = { text: $(".needle.text", this.el), market: $(".needle.market", this.el), final: $(".needle.final", this.el) };
      this.svg.addEventListener("mousemove", (e) => this.hover(e));
      this.svg.addEventListener("mouseleave", () => { hideTip(); if (this.cross) this.cross.setAttribute("opacity", 0); });
    }
    setArm([arm, tag, title]) {
      this.arm = arm; this.steps = RP.arms[arm]; this.shown = 0;
      const t = $(".ptag", this.el); t.textContent = tag.toUpperCase(); t.className = "ptag" + (tag === "With" ? " with" : "");
      $("h3", this.el).textContent = title;
      this.buildChart();
    }
    buildChart() {
      const S = this.svg; S.innerHTML = "";
      const W = 600, H = 236, L = 44, Rr = 10, T = 24, B = 24;
      const px = this.steps.map((s) => s.px);
      const lo = Math.min(...px), hi = Math.max(...px), pad = (hi - lo) * 0.18 || 1;
      const y0 = lo - pad, y1 = hi + pad;
      this.x = (i) => L + (i * (W - L - Rr)) / (N - 1);
      this.y = (p) => T + ((y1 - p) / (y1 - y0)) * (H - T - B);
      this.top = T; this.bot = H - B;
      const defs = svg("defs", {}, S);
      const gr = svg("linearGradient", { id: `areaGrad-${this.uid || (this.uid = Math.random().toString(36).slice(2))}`, x1: 0, x2: 0, y1: 0, y2: 1 }, defs);
      svg("stop", { offset: "0%", "stop-color": css("--s1"), "stop-opacity": 0.22 }, gr);
      svg("stop", { offset: "100%", "stop-color": css("--s1"), "stop-opacity": 0 }, gr);
      const clip = svg("clipPath", { id: `clip-${this.uid}` }, defs);
      this.clipRect = svg("rect", { x: 0, y: 0, width: L, height: H }, clip);
      const grid = svg("g", { class: "grid" }, S), axis = svg("g", { class: "axis" }, S);
      for (let i = 0; i <= 3; i++) {
        const v = y0 + ((y1 - y0) * (i + 0.5)) / 4, yy = this.y(v);
        svg("line", { x1: L, x2: W - Rr, y1: yy, y2: yy }, grid);
        const tx = svg("text", { x: L - 6, y: yy + 4, "text-anchor": "end" }, axis); tx.textContent = v.toFixed(0);
      }
      [0, 19, 39, N - 1].forEach((i, j, all) => {
        const tx = svg("text", { x: this.x(i), y: H - 6, "text-anchor": j === 0 ? "start" : j === all.length - 1 ? "end" : "middle" }, axis);
        tx.textContent = `day ${i + 1}`;
      });
      const d = px.map((p, i) => `${i ? "L" : "M"} ${this.x(i).toFixed(1)} ${this.y(p).toFixed(1)}`).join(" ");
      const body = svg("g", { "clip-path": `url(#clip-${this.uid})` }, S);
      svg("path", { d: `${d} L ${this.x(N - 1)} ${this.bot} L ${this.x(0)} ${this.bot} Z`, fill: `url(#areaGrad-${this.uid})` }, body);
      svg("path", { d, class: "price" }, body);
      this.bands = svg("g", {}, S);
      this.marks = svg("g", {}, S);
      this.cursor = svg("line", { class: "cursor", y1: T - 8, y2: this.bot }, S);
      this.cross = svg("line", { class: "crosshair", y1: T - 8, y2: this.bot, opacity: 0 }, S);
      this.svg.setAttribute("aria-label", `${$("h3", this.el).textContent}: price over ${N} days with the agent's decisions`);
    }
    marker(i) {
      const s = this.steps[i], cx = this.x(i), cy = this.y(s.px), g = svg("g", { "data-i": i }, this.marks);
      if (s.a !== s.s) {
        svg("rect", { class: "flipband", x: cx - 4, y: this.top - 4, width: 8, height: this.bot - this.top + 4 }, this.bands).dataset.i = i;
        const b = svg("text", { class: "flipbolt", x: cx, y: this.top - 8 }, this.bands); b.textContent = "⚡"; b.dataset.i = i;
        // the clean twin's decision, as a dashed outline
        if (s.s === "Buy") svg("path", { class: "mk twin", d: `M ${cx} ${cy - 18} L ${cx + 5} ${cy - 9} L ${cx - 5} ${cy - 9} Z` }, g);
        else if (s.s === "Sell") svg("path", { class: "mk twin", d: `M ${cx} ${cy + 18} L ${cx + 5} ${cy + 9} L ${cx - 5} ${cy + 9} Z` }, g);
        else svg("circle", { class: "mk twin", cx, cy, r: 5 }, g);
      }
      if (s.a === "Buy") svg("path", { class: "mk buy", d: `M ${cx} ${cy - 19} L ${cx + 6} ${cy - 8} L ${cx - 6} ${cy - 8} Z` }, g);
      else if (s.a === "Sell") svg("path", { class: "mk sell", d: `M ${cx} ${cy + 19} L ${cx + 6} ${cy + 8} L ${cx - 6} ${cy + 8} Z` }, g);
      else svg("circle", { class: "mk hold", cx, cy, r: 3.5 }, g);
    }
    show(day, animate) {
      // markers: add the newly revealed days, drop the ones after `day` when scrubbing back
      if (day < this.shown) {
        [...this.marks.children, ...this.bands.children].forEach((e) => { if (+e.dataset.i >= day) e.remove(); });
        this.shown = day;
      }
      while (this.shown < day) this.marker(this.shown++);
      const i = day - 1, s = this.steps[i], xr = this.x(i);
      this.clipRect.setAttribute("width", xr + 2);
      this.cursor.setAttribute("x1", xr); this.cursor.setAttribute("x2", xr);
      // KPIs
      const f = flips(this.arm, day), kf = $(".kpi-flips", this.el);
      if (+kf.textContent !== f) { kf.textContent = f; if (animate && s.a !== s.s) { kf.classList.remove("bump"); void kf.offsetWidth; kf.classList.add("bump"); } }
      $(".kpi-asr", this.el).textContent = `${Math.round((100 * f) / day)}%`;
      const ret = 100 * (s.eq - 1);
      $(".kpi-ret", this.el).textContent = `${ret >= 0 ? "+" : "−"}${Math.abs(ret).toFixed(1)}%`;
      if (animate && s.a !== s.s && !reduced) { this.el.classList.remove("flash"); void this.el.offsetWidth; this.el.classList.add("flash"); }
      // signal meter
      const pos = (v) => `${((clamp(v, -1, 1) + 1) / 2) * 100}%`;
      this.needles.text.style.left = pos(s.ts); this.needles.market.style.left = pos(s.ms); this.needles.final.style.left = pos(s.sig);
      // news feed
      const ul = $(".cards", this.el); ul.innerHTML = "";
      s.docs.forEach((d, j) => {
        const li = h("li", { class: "card" + (d.k === "s" ? " syn" : d.k === "p" ? " plc" : "") });
        if (animate) li.style.animationDelay = `${j * 70}ms`; else li.style.animation = "none";
        const lbl = d.k === "s" ? "⚠ SYNTHETIC" : d.k === "p" ? "≈ PARAPHRASE" : "📰 CLEAN";
        const tag = d.k === "s" ? [MINOR[d.m] || d.m, d.r ? REWRITE[d.r] : null].filter(Boolean).join(" · ") : d.k === "p" ? "same meaning, new words" : "genuine article";
        li.innerHTML = `<div class="lbl">${lbl}</div><div class="lines"><i></i><i></i><i></i></div><div class="tagline">${tag}</div>`;
        ul.append(li);
      });
      const bb = $(".bias-badge", this.el);
      bb.hidden = !s.bias; if (s.bias) bb.textContent = `⚙ attacker prompt: ${BIAS[s.bias] || s.bias}`;
      const v = $(".veto", this.el);
      v.hidden = !s.ov.length;
      if (s.ov.length) { v.textContent = "🛡 " + (VETO[s.ov[0]] || s.ov[0]); if (animate) { v.style.animation = "none"; void v.offsetWidth; v.style.animation = ""; } }
    }
    hover(e) {
      const r = this.svg.getBoundingClientRect(), sx = ((e.clientX - r.left) / r.width) * 600;
      const i = clamp(Math.round(((sx - 44) / (600 - 54)) * (N - 1)), 0, this.shown - 1);
      if (i < 0) return;
      const s = this.steps[i], xr = this.x(i);
      this.cross.setAttribute("x1", xr); this.cross.setAttribute("x2", xr); this.cross.setAttribute("opacity", 1);
      const nSyn = s.docs.filter((d) => d.k === "s").length, nPl = s.docs.filter((d) => d.k === "p").length;
      showTip(`<b>Day ${i + 1}</b> · ${s.d}<br>price ${s.px.toFixed(2)}<br>this agent: <b>${glyph(s.a)}</b><br>clean twin: ${glyph(s.s)}${s.a !== s.s ? " &nbsp;<b>⚡ flip</b>" : ""}
        <br>signals: text ${signed(s.ts, 2)} · market ${signed(s.ms, 2)} · final ${signed(s.sig, 2)}<br>news: ${nSyn} synthetic${nPl ? `, ${nPl} paraphrase` : ""}, ${s.docs.length - nSyn - nPl} clean${s.ov.length ? `<br>🛡 ${VETO[s.ov[0]] || s.ov[0]}` : ""}`, e.clientX, e.clientY);
    }
  }

  const panels = [new Panel(), new Panel()];
  const panelBox = $("#panels");
  panels.forEach((p) => panelBox.append(p.el));
  const state = { scen: 0, day: 1, playing: false, speed: 2, last: 0, acc: 0 };
  const scrub = $("#scrub"), playBtn = $("#btn-play"), verdict = $("#verdict");
  scrub.max = N;

  function setDay(d, animate = true) {
    state.day = clamp(d, 1, N);
    panels.forEach((p) => p.show(state.day, animate));
    scrub.value = state.day;
    $("#daylbl").textContent = `Day ${state.day} / ${N} · ${RP.arms.clean[state.day - 1].d}`;
    if (state.day === N) {
      const sc = SCEN[state.scen], [l, r] = [sc.left[0], sc.right[0]];
      const fl = flips(l), fr = flips(r);
      verdict.innerHTML = `<b>Over these ${N} days:</b> “${sc.right[2]}” changed <b>${fr}</b> decisions (${Math.round((100 * fr) / N)}%), “${sc.left[2]}” changed <b>${fl}</b> (${Math.round((100 * fl) / N)}%). ${sc.take}`;
      verdict.classList.add("show");
    } else verdict.classList.remove("show");
  }
  function loop(t) {
    if (!state.playing) return;
    if (!state.last) state.last = t;
    state.acc += t - state.last; state.last = t;
    const stepMs = 1000 / state.speed;
    while (state.acc >= stepMs) {
      state.acc -= stepMs;
      if (state.day >= N) { pause(); return; }
      setDay(state.day + 1);
    }
    requestAnimationFrame(loop);
  }
  function play() {
    if (state.day >= N) setDay(1, false);
    state.playing = true; state.last = 0; state.acc = 0;
    playBtn.textContent = "❚❚"; playBtn.setAttribute("aria-label", "Pause");
    requestAnimationFrame(loop);
  }
  function pause() { state.playing = false; playBtn.textContent = "▶"; playBtn.setAttribute("aria-label", "Play"); }
  function setScenario(i, autoplay) {
    state.scen = i;
    document.querySelectorAll(".tab").forEach((t, j) => t.setAttribute("aria-selected", j === i ? "true" : "false"));
    const sc = SCEN[i];
    $("#scenario-desc").innerHTML = sc.desc;
    panels[0].setArm(sc.left); panels[1].setArm(sc.right);
    setDay(1, false);
    if (autoplay && !reduced) play();
  }
  const tabs = $("#tabs");
  SCEN.forEach((sc, i) => {
    const b = h("button", { class: "tab", role: "tab", "aria-selected": "false" }, sc.tab);
    b.addEventListener("click", () => { pause(); setScenario(i, true); });
    tabs.append(b);
  });
  playBtn.addEventListener("click", () => (state.playing ? pause() : play()));
  $("#btn-restart").addEventListener("click", () => { pause(); setDay(1, false); play(); });
  $("#btn-back").addEventListener("click", () => { pause(); setDay(state.day - 1, false); });
  $("#btn-fwd").addEventListener("click", () => { pause(); setDay(state.day + 1, true); });
  scrub.addEventListener("input", () => { pause(); setDay(+scrub.value, false); });
  $("#speed").addEventListener("change", (e) => { state.speed = +e.target.value; });
  document.addEventListener("keydown", (e) => {
    if (e.target.closest("input, select, textarea")) return;
    const r = $("#replay").getBoundingClientRect();
    if (r.bottom < 0 || r.top > window.innerHeight) return;
    if (e.key === " ") { e.preventDefault(); state.playing ? pause() : play(); }
    if (e.key === "ArrowRight") { pause(); setDay(state.day + 1, true); }
    if (e.key === "ArrowLeft") { pause(); setDay(state.day - 1, false); }
  });
  setScenario(0, false);
  onVisible($("#panels"), () => { if (!reduced) play(); }, 0.35);
  // SVG gradients read the theme's colours when built: rebuild the charts on a theme switch
  document.addEventListener("rhal-theme", () => panels.forEach((p) => { p.buildChart(); p.shown = 0; p.show(state.day, false); }));
  $("#replay-note").innerHTML = `Replay of one simulated market (seed ${RP.seed}). In comparisons 3 and 4 both panels face the <i>exact same</i> recorded attack. Cards show only labels (edit type, rewrite style): the generated articles' text is never displayed. Hover a chart for day-by-day details; keyboard: space to play/pause, ← → to step.`;

  // ---------------------------------------------------------------- results
  (function results() {
    if (!R) { $("#results").querySelector(".res-grid").innerHTML = '<p class="muted">Run the experiments and <code>make frontend</code> to fill in the results.</p>'; return; }
    $("#results-title").textContent = `What ${R.n_seeds} simulated markets say`;
    const dec = R.decomposition;
    const segs = [
      ["Retrieval noise (any extra articles)", R.asr.placebo, "--s1", null],
      ["Matching the article count", dec.count && dec.count.mean, "--s2", dec.count],
      ["Attacker's bias prompts", dec.bias && dec.bias.mean, "--s3", dec.bias],
      ["The fake news' content", dec.meaning && dec.meaning.mean, "--s4", dec.meaning],
    ];
    const total = R.asr.naive, axisMax = Math.ceil(total / 10) * 10;
    $("#stack-sub").innerHTML = `Naive ASR is <b>${fmt(total)}%</b>. Placebo and matched controls split it into four parts. Only the last one is caused by what the fake news says.`;
    $("#stack-max").textContent = `${axisMax}%`;
    const stack = $("#stack"), legend = $("#stack-legend");
    segs.forEach(([name, v, color, c]) => {
      const seg = h("div", { class: "seg" });
      seg.style.background = `var(${color})`;
      seg.dataset.w = `${(Math.max(v || 0, 0) / axisMax) * 100}%`;
      if ((v || 0) / axisMax > 0.07) seg.innerHTML = `<span>${fmt(v)}</span>`;
      seg.addEventListener("mousemove", (e) => showTip(`<b>${name}</b><br>${fmt(v)} pp${c ? `<br>95% CI ${ciTxt(c)} · ${pTxt(c)}` : ""}`, e.clientX, e.clientY));
      seg.addEventListener("mouseleave", hideTip);
      stack.append(seg);
      legend.append(h("span", {}, `<i style="background:var(${color})"></i>${name}: <b>${fmt(v)}</b>${c ? " pp" : "%"}`));
    });
    onVisible(stack, () => { stack.querySelectorAll(".seg").forEach((s, i) => setTimeout(() => { s.style.width = s.dataset.w; }, i * 350)); setTimeout(() => stack.classList.add("grown"), 400); });

    const box = $("#netbars");
    const rows = [["No defence", R.net.none], ["Grounding", R.net.grounded], ["Grounding + risk", R.net.grounded_risk]].filter((r) => r[1]);
    const mx = Math.max(...rows.map((r) => r[1].hi)) * 1.1;
    rows.forEach(([name, c]) => {
      const row = h("div", { class: "bar-row" });
      row.innerHTML = `<span>${name}</span><div class="bar-track"><div class="bar-fill"></div><div class="bar-ci"></div></div><span class="bar-val">${fmt(c.mean)}%</span>`;
      const fill = $(".bar-fill", row), cib = $(".bar-ci", row);
      fill.dataset.w = `${(c.mean / mx) * 100}%`;
      Object.assign(cib.style, { left: `${(c.lo / mx) * 100}%`, width: `${((c.hi - c.lo) / mx) * 100}%` });
      row.addEventListener("mousemove", (e) => showTip(`<b>${name}</b><br>net ASR ${fmt(c.mean)}%<br>95% CI ${ciTxt(c)}`, e.clientX, e.clientY));
      row.addEventListener("mouseleave", hideTip);
      box.append(row);
    });
    onVisible(box, () => { box.querySelectorAll(".bar-fill").forEach((f, i) => setTimeout(() => { f.style.width = f.dataset.w; }, i * 250)); box.classList.add("grown"); });

    const F = [];
    const card = (kind, tag, num, title, text) => F.push({ kind, tag, num, title, text });
    if (g) card("yes", "✓ works", `−${fmt(g.mean)} pp`, "Temporal grounding", `Cuts misinformation-driven flips by ${fmt(g.mean)} pp (95% CI ${ciTxt(g)}), and by ${fmt(R.gain.grounding_frozen && R.gain.grounding_frozen.mean)} pp against the identical frozen attack.`);
    if (dec.meaning) card("yes", "✓ key finding", `${fmt(dec.meaning.mean)} of ${fmt(total)}`, "Naive ASR overstates the attack", `Only ${fmt(dec.meaning.mean)} of the ${fmt(total)} points of naive ASR come from what the fake news says. The rest is retrieval noise and bias prompts, which is why a placebo control is essential.`);
    if (dec.bias) card("warn", "! caution", `+${fmt(dec.bias.mean)} pp`, "Bias prompts alone flip decisions", `Giving harmless paraphrases the attacker's behavioural-bias prompts already adds ${fmt(dec.bias.mean)} pp ${ciTxt(dec.bias)} of flips.`);
    if (R.microstructure) card(R.microstructure.sig ? "yes" : "no", R.microstructure.sig ? "✓ works" : "✗ no effect", `${signed(R.microstructure.mean)} pp`, "Microstructure features", `Price-only grounding vs grounding with microstructure proxies: ${signed(R.microstructure.mean)} pp ${ciTxt(R.microstructure)} (${R.ablation_seeds} seeds). On synthetic daily bars they add ${R.microstructure.sig ? "a detectable gain" : "no detectable robustness"}.`);
    if (gainRisk) card(gainRisk.sig && gainRisk.mean < 0 ? "warn" : gainRisk.sig ? "yes" : "no", gainRisk.sig && gainRisk.mean < 0 ? "! backfires" : gainRisk.sig ? "✓ works" : "✗ no effect", `${signed(-gainRisk.mean)} pp`, "Risk layer on top of grounding", riskLine);
    if (R.cost.grounding) card("warn", "! trade-off", `−${fmt(R.cost.grounding.mean)} pp`, "Robustness costs return", `Grounding lowers clean-market return by ${fmt(R.cost.grounding.mean)} pp ${ciTxt(R.cost.grounding)}: the synthetic news is genuinely informative, and grounding sometimes discounts true stories.`);
    if (R.drift) card("no", R.drift.sig ? "≈ tiny" : "✗ no effect", `${signed(R.drift.mean)} pp`, "Drift between the twins", `After the first flip the two agents' portfolios differ. Re-scoring each day from the clean twin's state changes ASR by only ${signed(R.drift.mean)} pp ${ciTxt(R.drift)}.`);
    if (R.kappa) card(R.kappa.sig ? "yes" : "no", R.kappa.sig ? "✓ matters" : "✗ no effect", `${signed(R.kappa.mean)} pp`, "Grounding strength κ", `κ = 0.3 vs 0.7: ${signed(R.kappa.mean)} pp ${ciTxt(R.kappa)}. Switching grounding on matters far more than tuning its strength.`);
    const fbox = $("#findings");
    F.forEach((f, i) => {
      const el = h("div", { class: `finding reveal ${f.kind}` });
      el.innerHTML = `<span class="verdict-tag">${f.tag}</span><h4>${f.title}</h4><div class="num">${f.num}</div><p>${f.text}</p>`;
      el.style.transitionDelay = `${(i % 3) * 90}ms`;
      fbox.append(el); revealIO.observe(el);
    });

    const tbl = $("#numtable");
    const trow = (label, c, unit = "pp") => (c ? `<tr><td>${label}</td><td class="r">${signed(c.mean)} ${unit}</td><td class="r">${ciTxt(c)}</td><td class="r">${pTxt(c)}</td></tr>` : "");
    tbl.innerHTML = `<thead><tr><th>Quantity (paired over ${R.n_seeds} seeds)</th><th class="r">Estimate</th><th class="r">95% CI</th><th class="r">Holm p</th></tr></thead><tbody>
      <tr><td>Naive ASR, undefended</td><td class="r">${fmt(total)}%</td><td></td><td></td></tr>
      <tr><td>Placebo ASR (retrieval noise)</td><td class="r">${fmt(R.asr.placebo)}%</td><td></td><td></td></tr>
      ${trow("Net ASR, no defence", R.net.none, "%")}${trow("Net ASR, grounding", R.net.grounded, "%")}${trow("Net ASR, grounding + risk", R.net.grounded_risk, "%")}
      ${trow("Count matching", dec.count)}${trow("Bias prompts", dec.bias)}${trow("Meaning (misinformation content)", dec.meaning)}
      ${trow("Grounding gain, adaptive attack", g)}${trow("Grounding gain, frozen attack", R.gain.grounding_frozen)}
      ${trow("Risk layer on top (− = more flips)", gainRisk)}${trow("Clean-return cost of grounding", R.cost.grounding)}
      ${trow("ASR − counterfactual ASR (drift)", R.drift)}${trow(`Microstructure (${R.ablation_seeds} seeds)`, R.microstructure)}${trow(`κ 0.3 vs 0.7 (${R.ablation_seeds} seeds)`, R.kappa)}</tbody>`;
  })();

  $("#footer-meta").textContent = `Replay: seed ${RP.seed}, ${N} trading days. Results: ${R ? R.n_seeds : "–"} seeds` +
    (R && R.ablation_seeds && R.ablation_seeds !== R.n_seeds ? ` (ablations: ${R.ablation_seeds})` : "") + ". Every number on this page is generated by scripts/export_frontend_data.py.";
})();

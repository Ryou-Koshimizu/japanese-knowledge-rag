/* 索引室 — 画面の動き。外部ライブラリなし。文字は textContent で入れる（HTML として解釈しない）。 */
(() => {
  "use strict";

  const $ = (sel, root = document) => root.querySelector(sel);
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const ICONS = "/static/icons.svg";
  const METHOD_LABELS = { terms: "語の一致", heading: "見出し", trigram: "部分一致", vector: "埋め込み" };

  const state = { meta: null, persona: null, personas: [], library: null, busy: false, lastCited: [] };

  const store = {
    get(key) { try { return window.localStorage.getItem(key); } catch { return null; } },
    set(key, value) { try { window.localStorage.setItem(key, value); } catch { /* 保存できなくても動く */ } },
  };

  function el(tag, attrs = {}, ...children) {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(attrs)) {
      if (value === undefined || value === null || value === false) continue;
      if (key === "class") node.className = value;
      else if (key === "text") node.textContent = value;
      else if (key === "delay") node.style.animationDelay = `${value}ms`; // CSSOM のプロパティで設定（CSP で許される）
      else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
      else node.setAttribute(key, value === true ? "" : String(value));
    }
    for (const child of children.flat()) {
      if (child === null || child === undefined || child === false) continue;
      node.append(child instanceof Node ? child : document.createTextNode(String(child)));
    }
    return node;
  }

  function icon(name, cls = "icon") {
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("class", cls);
    svg.setAttribute("aria-hidden", "true");
    const use = document.createElementNS("http://www.w3.org/2000/svg", "use");
    use.setAttribute("href", `${ICONS}#i-${name}`);
    svg.append(use);
    return svg;
  }

  async function api(path, options = {}) {
    const headers = { "Content-Type": "application/json" };
    if (state.persona) headers["X-Demo-Persona"] = state.persona.id;
    const response = await fetch(path, { ...options, headers: { ...headers, ...(options.headers || {}) } });
    const body = await response.json().catch(() => ({}));
    if (!response.ok) {
      const detail = typeof body.detail === "string" ? body.detail : "通信に失敗しました。時間をおいてお試しください。";
      throw new Error(detail);
    }
    return body;
  }

  /* ── 数字のカウント ─────────────────────────── */
  function countUp(node, to, { decimals = 0, duration = 1200 } = {}) {
    if (reduced) { node.textContent = to.toFixed(decimals); return; }
    const from = Number(node.dataset.value || 0);
    const start = performance.now();
    node.dataset.value = String(to);
    const step = (now) => {
      const t = Math.min(1, (now - start) / duration);
      const eased = 1 - Math.pow(1 - t, 4);
      node.textContent = (from + (to - from) * eased).toFixed(decimals);
      if (t < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }

  /* ── 閲覧者の切り替え ─────────────────────────── */
  const personaButton = $("#persona-button");
  const personaList = $("#persona-list");

  function renderPersonas() {
    personaList.replaceChildren(
      ...state.personas.map((p) =>
        el("li", {
          class: "persona-option",
          role: "option",
          id: `persona-${p.id}`,
          "aria-selected": state.persona && p.id === state.persona.id ? "true" : "false",
          "data-id": p.id,
          onclick: () => choosePersona(p.id, true),
        },
        el("span", { class: "po-name", text: p.name }),
        el("span", { class: "po-docs", text: `${p.department}` }),
        el("span", { class: "po-meta", text: `${p.role}として読める規程で答えます` }))
      )
    );
  }

  function togglePersonaList(open) {
    const willOpen = open ?? personaList.hidden;
    personaList.hidden = !willOpen;
    personaButton.setAttribute("aria-expanded", String(willOpen));
    if (willOpen) {
      const current = personaList.querySelector('[aria-selected="true"]') || personaList.firstElementChild;
      setActiveOption(current);
      personaList.setAttribute("tabindex", "-1");
      personaList.focus({ preventScroll: true });
    }
  }

  function setActiveOption(option) {
    personaList.querySelectorAll(".persona-option").forEach((o) => o.classList.toggle("active", o === option));
    if (option) {
      personaList.setAttribute("aria-activedescendant", option.id);
      option.scrollIntoView({ block: "nearest" });
    }
  }

  personaButton.addEventListener("click", () => togglePersonaList());
  personaList.addEventListener("keydown", (event) => {
    const options = [...personaList.querySelectorAll(".persona-option")];
    const index = options.findIndex((o) => o.classList.contains("active"));
    if (event.key === "ArrowDown") { event.preventDefault(); setActiveOption(options[(index + 1) % options.length]); }
    else if (event.key === "ArrowUp") { event.preventDefault(); setActiveOption(options[(index - 1 + options.length) % options.length]); }
    else if (event.key === "Enter" || event.key === " ") { event.preventDefault(); if (options[index]) choosePersona(options[index].dataset.id, true); }
    else if (event.key === "Escape") { togglePersonaList(false); personaButton.focus(); }
  });
  document.addEventListener("click", (event) => {
    if (!personaList.hidden && !event.target.closest(".persona")) togglePersonaList(false);
  });

  async function choosePersona(id, fromUser = false) {
    const persona = state.personas.find((p) => p.id === id) || state.personas[0];
    if (!persona) return;
    state.persona = persona;
    store.set("persona", persona.id);
    $("#persona-name").textContent = persona.name;
    renderPersonas();
    if (fromUser) { togglePersonaList(false); personaButton.focus(); }
    try {
      state.library = await api("/api/library");
      countUp($("#readable-count"), state.library.documents.length, { duration: 700 });
      $("#readable-note").textContent = `${persona.name}として読める規程です。ほかの部署の文書は検索の段階で除きます。`;
      renderSuggestions(state.library.suggestions);
    } catch (error) {
      $("#readable-note").textContent = error.message;
    }
    if (fromUser && !$("#reading").hidden) {
      clearReading();
    }
  }

  function renderSuggestions(questions) {
    const list = $("#suggest-list");
    const picks = questions.slice(0, 8);
    list.replaceChildren(
      ...picks.map((q) => el("li", {}, el("button", { class: "chip", type: "button", text: q, onclick: () => ask(q) })))
    );
  }

  /* ── 質問 ─────────────────────────────────── */
  const form = $("#ask");
  const input = $("#question");
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const question = input.value.trim();
    if (question) ask(question);
  });

  function clearReading() {
    $("#reading").hidden = true;
    $("#answer").hidden = true;
    $("#hits").hidden = true;
    const old = $("#reading .notice");
    if (old) old.remove();
  }

  const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

  async function ask(question) {
    if (state.busy || !state.persona) return;
    state.busy = true;
    input.value = question;
    const submit = form.querySelector(".ask-submit");
    submit.disabled = true;
    const reading = $("#reading");
    clearReading();
    reading.hidden = false;
    reading.setAttribute("aria-busy", "true");
    $("#ink").hidden = false;
    reading.scrollIntoView({ behavior: reduced ? "auto" : "smooth", block: "start" });
    try {
      const [result] = await Promise.all([
        api("/api/ask", { method: "POST", body: JSON.stringify({ question }) }),
        wait(reduced ? 0 : 650),
      ]);
      $("#ink").hidden = true;
      if (result.status === "not_recorded") renderNotRecorded(result);
      else renderAnswer(result);
      renderHits(result.hits || []);
    } catch (error) {
      $("#ink").hidden = true;
      reading.prepend(notice("circle-alert", "通信エラー", error.message));
    } finally {
      reading.setAttribute("aria-busy", "false");
      submit.disabled = false;
      state.busy = false;
    }
  }

  function notice(iconName, title, text, extra) {
    return el("div", { class: "notice", role: "status" },
      icon(iconName),
      el("div", {}, el("p", { class: "notice-title", text: title }), el("p", { text }), extra || null));
  }

  function renderNotRecorded(result) {
    const suggestions = el("ul", { class: "suggest-list notice-suggest" },
      ...(result.suggestions || []).slice(0, 6).map((q) =>
        el("li", {}, el("button", { class: "chip", type: "button", text: q, onclick: () => ask(q) }))));
    $("#reading").prepend(notice(
      "history",
      "公開デモは、実AIで記録した回答を再生しています",
      "この質問の記録はありません。下の「検索した箇所」はこの場で検索した結果です。記録済みの質問はこちらです。",
      suggestions,
    ));
  }

  function renderAnswer(result) {
    const article = $("#answer");
    article.hidden = false;
    article.classList.toggle("not-found", !result.found);
    $("#answer-persona").textContent = state.persona ? state.persona.name : "";
    $("#answer-question").textContent = result.question;

    const meta = $("#answer-meta");
    meta.replaceChildren();
    if (result.replayed_at) {
      const when = new Date(result.replayed_at);
      meta.append(el("span", { class: "stamp" }, icon("history"), "記録の再生"));
      meta.append(el("span", { text: `${result.model} で ${when.toLocaleDateString("ja-JP")} に記録した回答` }));
    } else if (result.model) {
      meta.append(el("span", { class: "stamp" }, icon("badge-check"), "実AIの回答"));
      meta.append(el("span", { text: `${result.model}・${result.seconds}秒` }));
    }
    if (!result.found) meta.append(el("span", { text: "根拠になる条文が見つからないため、推測で答えていません" }));

    typeset($("#answer-body"), result.answer, result.found);
    $("#answer-foot").textContent = result.found
      ? "引用は、渡した条文の本文と一字一句照合して、一致したものだけを載せています。"
      : "読める規程の中に答えが書かれていないときは、この一文で答えます。";
    renderSources(result.citations || []);
  }

  /* 回答を一文字ずつ組版する。[n] は根拠への印に置き換える */
  function typeset(container, text, found) {
    container.replaceChildren();
    const paragraph = el("p");
    container.append(paragraph);
    const parts = text.split(/(\[\d+\])/g).filter(Boolean);
    const total = text.replace(/\[\d+\]/g, "").length || 1;
    const step = reduced ? 0 : Math.min(18, 1400 / total);
    let index = 0;
    let first = true;
    for (const part of parts) {
      const cite = part.match(/^\[(\d+)\]$/);
      if (cite) {
        const n = cite[1];
        const button = el("button", {
          class: "cite ch", type: "button", "aria-label": `根拠 ${n} を見る`, text: n,
          delay: index * step,
          onclick: () => focusSource(n),
        });
        paragraph.append(button);
        continue;
      }
      for (const ch of part) {
        const span = el("span", { class: first && found ? "ch dropcap" : "ch", text: ch });
        span.style.animationDelay = `${index * step}ms`;
        paragraph.append(span);
        index += 1;
        first = false;
      }
    }
    // 読み上げには文字の分割ではなく文全体を渡す
    container.setAttribute("aria-label", text.replace(/\[(\d+)\]/g, "（根拠$1）"));
    container.setAttribute("role", "text");
  }

  function renderSources(citations) {
    const list = $("#source-list");
    list.replaceChildren();
    state.lastCited = citations.map((c) => c.chunk_id);
    if (!citations.length) {
      list.append(el("li", {}, el("p", { class: "sources-empty", text: "根拠として示せる条文はありません。" })));
      return;
    }
    const groups = new Map();
    for (const c of citations) {
      if (!groups.has(c.number)) groups.set(c.number, { ...c, quotes: [] });
      groups.get(c.number).quotes.push(c.quote);
    }
    let order = 0;
    for (const group of groups.values()) {
      const quotes = group.quotes.map((q) => el("p", { class: "source-quote" }, "「", el("mark", { text: q }), "」"));
      const card = el("li", { class: "source", id: `source-${group.number}`, delay: 200 + order * 140 },
        el("span", { class: "source-num", "aria-hidden": "true", text: group.number }),
        el("p", { class: "source-doc" }, el("span", { class: "sr-only", text: `根拠 ${group.number}：` }), group.doc_title),
        el("p", { class: "source-heading", text: group.heading }),
        ...quotes,
        el("button", {
          class: "open-doc", type: "button",
          onclick: () => openDocument(group.doc_id, group.chunk_id, group.quotes),
        }, "文書をひらく", icon("arrow-up-right")));
      list.append(card);
      const marks = card.querySelectorAll("mark");
      setTimeout(() => marks.forEach((m) => m.classList.add("on")), reduced ? 0 : 700 + order * 220);
      order += 1;
    }
  }

  function focusSource(n) {
    const card = document.getElementById(`source-${n}`);
    if (!card) return;
    card.scrollIntoView({ behavior: reduced ? "auto" : "smooth", block: "nearest" });
    card.classList.remove("flash");
    void card.offsetWidth;
    card.classList.add("flash");
    setTimeout(() => card.classList.remove("flash"), 1600);
    const button = card.querySelector(".open-doc");
    if (button) button.focus({ preventScroll: true });
  }

  function renderHits(hits) {
    const section = $("#hits");
    const list = $("#hit-list");
    list.replaceChildren();
    if (!hits.length) { section.hidden = true; return; }
    section.hidden = false;
    hits.forEach((hit, i) => {
      const ranks = Object.entries(METHOD_LABELS).map(([key, label]) => {
        const r = hit.ranks[key];
        return el("li", { class: r === 1 ? "top" : "" , text: `${label} ${r ? `${r}位` : "—"}` });
      });
      list.append(el("li", { class: "hit", delay: i * 50 },
        el("span", { class: "hit-rank", "aria-hidden": "true", text: i + 1 }),
        el("div", {},
          el("p", { class: "hit-title" },
            el("span", { class: "sr-only", text: `${i + 1}位：` }),
            el("button", { type: "button", text: hit.doc_title, onclick: () => openDocument(hit.doc_id, hit.chunk_id, []) })),
          el("p", { class: "hit-heading", text: hit.heading || "（見出しなし）" }),
          el("p", { class: "hit-body", text: hit.body }),
          el("ul", { class: "hit-ranks", "aria-label": "方式ごとの順位" }, ranks))));
    });
  }

  /* ── 文書（紙がめくれて開く） ─────────────────── */
  const sheet = $("#sheet");
  const backdrop = $("#sheet-backdrop");
  let lastFocus = null;

  async function openDocument(docId, chunkId, quotes) {
    lastFocus = document.activeElement;
    let doc;
    try {
      doc = await api(`/api/documents/${encodeURIComponent(docId)}`);
    } catch (error) {
      $("#reading").prepend(notice("circle-alert", "文書の取得でエラー", error.message));
      return;
    }
    const formats = { markdown: "Markdown", pdf: "PDF", docx: "Word", html: "HTML" };
    $("#sheet-kicker").textContent = `${formats[doc.format] || doc.format}・所管 ${doc.owner}・改定 ${doc.updated}`;
    $("#sheet-title").textContent = doc.title;
    const body = $("#sheet-body");
    body.replaceChildren();
    let target = null;
    for (const section of doc.sections) {
      const cited = section.chunk_id === chunkId;
      const text = el("p");
      highlightInto(text, section.body, cited ? quotes : []);
      const node = el("section", { class: cited ? "doc-section cited" : "doc-section" },
        el("h3", { text: section.heading || doc.title }), text);
      if (cited) target = node;
      body.append(node);
    }
    sheet.classList.remove("closing");
    backdrop.classList.remove("closing");
    sheet.hidden = false;
    backdrop.hidden = false;
    document.body.style.overflow = "hidden";
    body.scrollTop = 0;
    if (target) setTimeout(() => target.scrollIntoView({ block: "center", behavior: reduced ? "auto" : "smooth" }), reduced ? 0 : 520);
    $("#sheet-close").focus({ preventScroll: true });
  }

  function highlightInto(node, text, quotes) {
    const squash = (s) => s.replace(/\s+/g, "");
    let rest = text;
    for (const quote of quotes) {
      const at = rest.indexOf(quote);
      if (at < 0) {
        if (!squash(rest).includes(squash(quote))) continue;
        continue;
      }
      node.append(rest.slice(0, at), el("mark", { class: "on", text: quote }));
      rest = rest.slice(at + quote.length);
    }
    node.append(rest);
  }

  function closeSheet() {
    if (sheet.hidden) return;
    const finish = () => {
      sheet.hidden = true;
      backdrop.hidden = true;
      document.body.style.overflow = "";
      if (lastFocus && document.contains(lastFocus)) lastFocus.focus({ preventScroll: true });
    };
    if (reduced) { finish(); return; }
    sheet.classList.add("closing");
    backdrop.classList.add("closing");
    setTimeout(finish, 420);
  }

  $("#sheet-close").addEventListener("click", closeSheet);
  backdrop.addEventListener("click", closeSheet);
  document.addEventListener("keydown", (event) => {
    if (sheet.hidden) return;
    if (event.key === "Escape") { closeSheet(); return; }
    if (event.key === "Tab") {
      const focusables = [...sheet.querySelectorAll('button, [tabindex="0"]')];
      const first = focusables[0];
      const last = focusables[focusables.length - 1];
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
    }
  });

  /* ── 奥付（測った数字） ─────────────────────── */
  function renderMetrics(summary) {
    const metrics = $("#metrics");
    metrics.replaceChildren();
    if (!summary) {
      metrics.append(el("div", { class: "metric" }, el("dt", { text: "測定" }), el("dd", {}, el("span", { class: "metric-note", text: "測定の記録は docs/evidence にあります。" }))));
      return;
    }
    const items = [];
    const a = summary.answers;
    if (a) {
      items.push(["回答の正答", a.all_ok, a.questions, `実AI（${a.model}）で${a.questions}問すべてを採点`]);
      items.push(["誤った引用", a.citations_wrong, null, `表示した引用${a.citations}件のうち、正解の文書以外を引いた数`]);
      items.push(["「記載なし」の正答", a.not_found_ok, a.unanswerable, "書かれていない・読めない・乗っ取りの質問"]);
    }
    const r = summary.retrieval;
    if (r) items.push(["検索 Recall@5", r.recall_at_5, null, `${r.questions}問・MRR ${r.mrr.toFixed(3)}（${r.label}）`, 3]);
    const acl = summary.acl;
    if (acl) items.push(["権限の漏れ", acl.leaks, null, `${acl.searches}回の検索・${acl.hits_checked}件の結果を照合`]);
    for (const [label, value, total, note, decimals = 0] of items) {
      const number = el("span", { class: "count", text: "0" });
      const valueNode = el("span", { class: "metric-value" }, number, total !== null ? el("small", { text: `／${total}` }) : null);
      metrics.append(el("div", { class: "metric" }, el("dt", { text: label }), el("dd", {}, valueNode, el("span", { class: "metric-note", text: note }))));
      number.dataset.target = String(value);
      number.dataset.decimals = String(decimals);
    }
    const observer = new IntersectionObserver((entries) => {
      for (const entry of entries) {
        if (!entry.isIntersecting) continue;
        entry.target.querySelectorAll(".count[data-target]").forEach((n) =>
          countUp(n, Number(n.dataset.target), { decimals: Number(n.dataset.decimals) }));
        observer.unobserve(entry.target);
      }
    }, { threshold: 0.3 });
    observer.observe(metrics);
    $("#colophon-note").textContent = summary.note || "";
  }

  /* ── 磁石のボタン ─────────────────────────── */
  function magnetize(node) {
    if (reduced || !window.matchMedia("(hover: hover)").matches) return;
    node.addEventListener("pointermove", (event) => {
      const rect = node.getBoundingClientRect();
      const x = (event.clientX - rect.left - rect.width / 2) * 0.18;
      const y = (event.clientY - rect.top - rect.height / 2) * 0.3;
      node.style.transform = `translate(${x}px, ${y}px)`;
    });
    node.addEventListener("pointerleave", () => { node.style.transform = ""; });
  }
  document.querySelectorAll(".magnetic").forEach(magnetize);

  /* 狭い画面では例文を短くして、入力欄で文字が欠けないようにする */
  const LONG_HINT = "例）育休は子どもが何歳になるまで取れますか";
  const HINTS = [LONG_HINT, "例）育休は何歳まで？", "質問を入力"];
  const fitPlaceholder = () => {
    const size = parseFloat(getComputedStyle(input).fontSize) || 16;
    input.placeholder = HINTS.find((h) => h.length * size * 1.04 <= input.clientWidth) || HINTS[HINTS.length - 1];
  };
  fitPlaceholder();
  if ("ResizeObserver" in window) new ResizeObserver(fitPlaceholder).observe(input);

  /* ── 起動 ─────────────────────────────────── */
  async function boot() {
    try {
      state.meta = await api("/api/meta");
    } catch (error) {
      $("#issue-mode").textContent = error.message;
      return;
    }
    const meta = state.meta;
    $("#issue-docs").textContent = `規程 ${meta.documents}篇`;
    $("#deck-docs").textContent = String(meta.documents);
    $("#issue-mode").textContent = meta.mode === "demo" ? "公開デモ：実AIの記録を再生" : `実AI：${meta.model}`;
    state.personas = meta.personas;
    renderMetrics(meta.summary);
    if (!state.personas.length) {
      $("#readable-note").textContent = "この環境では、ログインした利用者の権限で検索します。";
      return;
    }
    await choosePersona(store.get("persona") || state.personas[0].id);
  }
  boot();
})();

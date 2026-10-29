(function () {
  const heuristic = globalThis.VerifaiHeuristic;
  const runtime = (globalThis.browser || globalThis.chrome).runtime;
  if (!heuristic || !document.body) {
    return;
  }

  const article = document.querySelector("article");
  const root = article || document.body;
  const excerpt = (root.innerText || "").replace(/\s+/g, " ").trim().slice(0, 4000);
  const decision = heuristic.shouldActivate({
    url: location.href,
    title: document.title,
    excerpt: excerpt,
    hasArticle: Boolean(article),
  });
  if (!decision.activate) {
    return;
  }

  const badge = document.createElement("button");
  badge.type = "button";
  badge.textContent = "VerifAI";
  badge.setAttribute("aria-label", "Check claims on this page with VerifAI");
  Object.assign(badge.style, {
    position: "fixed",
    right: "16px",
    bottom: "16px",
    zIndex: "2147483646",
    border: "1px solid #1c1915",
    background: "#1c1915",
    color: "#f4efe4",
    font: "600 13px sans-serif",
    padding: "8px 12px",
    cursor: "pointer",
  });
  badge.addEventListener("click", function () {
    const text = selectedOrHighlighted();
    if (!text) {
      showPanel("Highlight a sentence, or open a page with a clearer claim.");
      return;
    }
    badge.disabled = true;
    runtime.sendMessage({ type: "check", text: text }, function (response) {
      badge.disabled = false;
      if (!response || !response.ok) {
        showPanel((response && response.error) || "VerifAI did not answer.");
        return;
      }
      showPanel(renderBody(response.body));
    });
  });
  document.body.appendChild(badge);

  function selectedOrHighlighted() {
    const selection = window.getSelection();
    const selected = selection ? selection.toString().trim() : "";
    if (selected.length >= 30) {
      return selected.slice(0, 4000);
    }
    const sentences = excerpt
      .split(/(?<=[.!?])\s+/)
      .map(function (sentence) {
        return sentence.trim();
      })
      .filter(function (sentence) {
        return sentence.length >= 40 && !sentence.endsWith("?");
      })
      .slice(0, 5);
    return sentences.join("\n");
  }

  function renderBody(body) {
    const lines = [];
    if (body && body.explanation) {
      lines.push(body.explanation);
    }
    const claims = (body && body.combined) || [];
    claims.forEach(function (claim) {
      lines.push((claim.verdict_label || claim.verdict || "Unverified") + ": " + claim.claim);
      (claim.evidence || []).forEach(function (item) {
        if (item.url) {
          lines.push((item.speaker ? item.speaker + ": " : "") + (item.title || item.url));
          lines.push(item.url);
        }
      });
    });
    return lines.join("\n\n") || "VerifAI returned no verdict.";
  }

  function showPanel(text) {
    let panel = document.getElementById("verifai-panel");
    if (!panel) {
      panel = document.createElement("pre");
      panel.id = "verifai-panel";
      Object.assign(panel.style, {
        position: "fixed",
        right: "16px",
        bottom: "56px",
        zIndex: "2147483646",
        width: "min(420px, calc(100vw - 32px))",
        maxHeight: "50vh",
        overflow: "auto",
        margin: "0",
        padding: "12px",
        border: "1px solid #1c1915",
        background: "#fffdf8",
        color: "#1c1915",
        font: "14px/1.4 sans-serif",
        whiteSpace: "pre-wrap",
      });
      document.body.appendChild(panel);
    }
    panel.textContent = text;
  }
})();

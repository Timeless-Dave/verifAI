(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) {
    module.exports = api;
  } else {
    root.VerifaiHeuristic = api;
  }
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  const CHAT_HOSTS = [
    "chatgpt.com",
    "chat.openai.com",
    "claude.ai",
    "gemini.google.com",
    "perplexity.ai",
    "poe.com",
    "chat.deepseek.com",
  ];
  const HEALTH_HOSTS = [
    "webmd.com",
    "mayoclinic.org",
    "medlineplus.gov",
    "nih.gov",
    "who.int",
    "healthline.com",
    "cdc.gov",
    "cochrane.org",
  ];
  const NEWS_HOSTS = [
    "bbc.com",
    "bbc.co.uk",
    "reuters.com",
    "apnews.com",
    "nytimes.com",
    "theguardian.com",
    "washingtonpost.com",
    "wsj.com",
    "foxnews.com",
    "cnn.com",
    "politico.com",
    "npr.org",
    "aljazeera.com",
    "bloomberg.com",
    "thehill.com",
    "axios.com",
  ];
  const QUIET_HOSTS = [
    "github.com",
    "gitlab.com",
    "stackoverflow.com",
    "google.com",
    "docs.google.com",
    "mail.google.com",
    "youtube.com",
    "netflix.com",
    "amazon.com",
  ];
  const HEALTH_WORDS = [
    "vaccine",
    "cancer",
    "symptom",
    "disease",
    "treatment",
    "clinical",
    "trial",
    "medication",
    "virus",
    "covid",
    "diabetes",
    "supplement",
    "patient",
    "therapy",
    "infection",
  ];
  const NEWS_WORDS = [
    "election",
    "president",
    "minister",
    "parliament",
    "senator",
    "congress",
    "campaign",
    "ballot",
    "government",
    "protest",
    "invasion",
    "sanction",
    "democrat",
    "republican",
    "scandal",
  ];

  function hostname(url) {
    try {
      return new URL(url).hostname.toLowerCase();
    } catch (_error) {
      return "";
    }
  }

  function hostIn(host, names) {
    return names.some(function (name) {
      return host === name || host.endsWith("." + name);
    });
  }

  function wordScore(text, words) {
    const folded = text.toLowerCase();
    return words.reduce(function (count, word) {
      return count + (folded.includes(word) ? 1 : 0);
    }, 0);
  }

  function citationScore(text) {
    const dois = text.match(/\b10\.\d{4,9}\//g);
    const etal = text.match(/\bet\s+al\b/gi);
    if (dois && dois.length >= 1) {
      return 2;
    }
    return etal ? etal.length : 0;
  }

  function shouldActivate(page) {
    const input = page || {};
    const host = hostname(String(input.url || ""));
    const excerpt = String(input.excerpt || "");
    const text = String(input.title || "") + "\n" + excerpt;
    const chars = excerpt.trim().length;
    const reasons = [];
    if (!host) {
      return { activate: false, reasons: reasons };
    }
    if (hostIn(host, CHAT_HOSTS) && chars >= 40) {
      reasons.push("chat");
    }
    if (hostIn(host, HEALTH_HOSTS) && chars >= 400) {
      reasons.push("health-site");
    }
    if (hostIn(host, NEWS_HOSTS) && chars >= 400) {
      reasons.push("news-site");
    }
    if (citationScore(text) >= 2) {
      reasons.push("citations");
    }
    const quiet = hostIn(host, QUIET_HOSTS);
    if (!quiet && chars >= 500 && wordScore(text, HEALTH_WORDS) >= 3) {
      reasons.push("health-text");
    }
    if (!quiet && chars >= 500 && wordScore(text, NEWS_WORDS) >= 3) {
      reasons.push("news-text");
    }
    if (
      !quiet &&
      input.hasArticle &&
      chars >= 700 &&
      (wordScore(text, HEALTH_WORDS) >= 1 ||
        wordScore(text, NEWS_WORDS) >= 1 ||
        citationScore(text) >= 1)
    ) {
      reasons.push("article");
    }
    return { activate: reasons.length > 0, reasons: reasons };
  }

  return { shouldActivate, hostname, citationScore };
});

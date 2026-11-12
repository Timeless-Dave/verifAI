(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) {
    module.exports = api;
  } else {
    root.VerifaiConfig = api;
  }
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  const DEFAULT_SERVER = "http://127.0.0.1:8741";

  function normalizeServer(value) {
    if (typeof value !== "string" || value.trim() === "") {
      return DEFAULT_SERVER;
    }
    let parsed;
    try {
      parsed = new URL(value.trim());
    } catch (_error) {
      return null;
    }
    if (parsed.username || parsed.password) {
      return null;
    }
    if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
      return null;
    }
    return parsed.origin;
  }

  // The manifest grants localhost. Any other server is asked for when saved.
  function needsPermission(origin) {
    const host = new URL(origin).hostname;
    return host !== "127.0.0.1" && host !== "localhost";
  }

  function originPattern(origin) {
    return origin + "/*";
  }

  // VerifAI answers errors as {"detail": "..."}. Show that sentence.
  function errorMessage(status, body) {
    let detail = "";
    try {
      const parsed = JSON.parse(body);
      if (parsed && typeof parsed.detail === "string") {
        detail = parsed.detail;
      }
    } catch (_error) {
      detail = "";
    }
    if (!detail && typeof body === "string" && body.length < 300 && !body.includes("<")) {
      detail = body.trim();
    }
    return detail || "VerifAI returned " + status + ".";
  }

  return { DEFAULT_SERVER, normalizeServer, needsPermission, originPattern, errorMessage };
});

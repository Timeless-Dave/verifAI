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

  return { DEFAULT_SERVER, normalizeServer };
});

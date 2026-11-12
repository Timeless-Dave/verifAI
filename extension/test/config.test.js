const test = require("node:test");
const assert = require("node:assert/strict");
const { DEFAULT_SERVER, normalizeServer } = require("../src/config.js");

test("the default server is the public VerifAI server", function () {
  assert.equal(DEFAULT_SERVER, "https://useverifai.vercel.app");
  assert.equal(normalizeServer(""), DEFAULT_SERVER);
  assert.equal(normalizeServer("   "), DEFAULT_SERVER);
});

test("http and https origins are kept and secrets are refused", function () {
  assert.equal(normalizeServer("http://127.0.0.1:8741/check"), "http://127.0.0.1:8741");
  assert.equal(normalizeServer("https://verifai.example/app"), "https://verifai.example");
  assert.equal(normalizeServer("javascript:alert(1)"), null);
  assert.equal(normalizeServer("http://user:secret@127.0.0.1:8741"), null);
  assert.equal(normalizeServer("not a url"), null);
});

const { needsPermission, originPattern, errorMessage } = require("../src/config.js");

test("only servers other than localhost need a permission prompt", function () {
  assert.equal(needsPermission("http://127.0.0.1:8741"), false);
  assert.equal(needsPermission("http://localhost:8741"), false);
  assert.equal(needsPermission("https://verifai.example"), true);
  assert.equal(needsPermission("https://useverifai.vercel.app"), false);
  assert.equal(originPattern("https://verifai.example"), "https://verifai.example/*");
});

test("error messages use the server's detail sentence", function () {
  const limited = JSON.stringify({ detail: "This server allows 5 checks per minute." });
  assert.equal(errorMessage(429, limited), "This server allows 5 checks per minute.");
  assert.equal(errorMessage(502, "<html>Bad gateway</html>"), "VerifAI returned 502.");
  assert.equal(errorMessage(500, "plain failure"), "plain failure");
  assert.equal(errorMessage(503, ""), "VerifAI returned 503.");
});

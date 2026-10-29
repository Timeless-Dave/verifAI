const test = require("node:test");
const assert = require("node:assert/strict");
const { DEFAULT_SERVER, normalizeServer } = require("../src/config.js");

test("the default server is the local VerifAI port", function () {
  assert.equal(DEFAULT_SERVER, "http://127.0.0.1:8741");
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

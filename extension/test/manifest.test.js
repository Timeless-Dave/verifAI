const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

test("the manifest is version 3 for Chrome and Firefox", function () {
  const manifest = JSON.parse(
    fs.readFileSync(path.join(__dirname, "..", "manifest.json"), "utf8"),
  );
  assert.equal(manifest.manifest_version, 3);
  assert.equal(manifest.background.service_worker, "src/background.js");
  assert.equal(manifest.browser_specific_settings.gecko.id, "verifai@localhost");
  assert.ok(manifest.host_permissions.includes("http://127.0.0.1/*"));
  assert.deepEqual(manifest.content_scripts[0].js, [
    "src/config.js",
    "src/heuristic.js",
    "src/content.js",
  ]);
  assert.equal(JSON.stringify(manifest).includes("openai.com"), false);
});

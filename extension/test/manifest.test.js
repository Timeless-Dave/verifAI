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
  assert.deepEqual(manifest.background.scripts, ["src/config.js", "src/background.js"]);
  assert.equal(manifest.browser_specific_settings.gecko.id, "verifai@useverifai.vercel.app");
  assert.deepEqual(manifest.browser_specific_settings.gecko.data_collection_permissions, {
    required: ["websiteContent"],
  });
  for (const size of ["16", "32", "48", "128"]) {
    assert.ok(fs.existsSync(path.join(__dirname, "..", manifest.icons[size])), size);
  }
  assert.ok(manifest.host_permissions.includes("http://127.0.0.1/*"));
  assert.ok(manifest.host_permissions.includes("https://useverifai.vercel.app/*"));
  assert.deepEqual(manifest.optional_host_permissions, ["https://*/*", "http://*/*"]);
  assert.equal(manifest.host_permissions.includes("<all_urls>"), false);
  assert.deepEqual(manifest.content_scripts[0].js, [
    "src/config.js",
    "src/heuristic.js",
    "src/content.js",
  ]);
  assert.equal(JSON.stringify(manifest).includes("openai.com"), false);
});

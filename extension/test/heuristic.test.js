const test = require("node:test");
const assert = require("node:assert/strict");
const { shouldActivate } = require("../src/heuristic.js");

const filler = "word ".repeat(200);

test("news sites with a real article activate", function () {
  const result = shouldActivate({
    url: "https://www.bbc.com/news/world-123",
    title: "Council vote",
    excerpt: filler,
  });
  assert.equal(result.activate, true);
  assert.ok(result.reasons.includes("news-site"));
});

test("a health site activates", function () {
  const result = shouldActivate({
    url: "https://www.mayoclinic.org/diseases-conditions/flu/symptoms",
    title: "Flu",
    excerpt: filler,
  });
  assert.equal(result.activate, true);
  assert.ok(result.reasons.includes("health-site"));
});

test("chat transcripts activate and a bare search page does not", function () {
  const chat = shouldActivate({
    url: "https://chatgpt.com/c/abc",
    title: "Chat",
    excerpt: "The model claimed the tower is 400 metres tall and gave no source for that figure.",
  });
  assert.equal(chat.activate, true);
  assert.ok(chat.reasons.includes("chat"));

  const search = shouldActivate({
    url: "https://www.google.com/search?q=election+president+government",
    title: "election president government",
    excerpt: filler + " election president government scandal",
  });
  assert.equal(search.activate, false);
});

test("ordinary pages stay quiet", function () {
  assert.equal(
    shouldActivate({
      url: "https://github.com/example/repo",
      title: "README",
      excerpt: filler + "install the package and run the tests",
    }).activate,
    false,
  );
  assert.equal(
    shouldActivate({
      url: "https://example.com/recipe",
      title: "Soup",
      excerpt: filler + "simmer the vegetables and serve warm",
    }).activate,
    false,
  );
  assert.equal(shouldActivate({ url: "", excerpt: filler }).activate, false);
  assert.equal(
    shouldActivate({
      url: "https://www.youtube.com/watch?v=abc",
      title: "video",
      excerpt: filler,
    }).activate,
    false,
  );
});

test("a page with a DOI activates even on an unknown host", function () {
  const result = shouldActivate({
    url: "https://example.com/notes",
    title: "Notes",
    excerpt: "See 10.1126/science.1225829 for the enzyme paper. " + filler,
  });
  assert.equal(result.activate, true);
  assert.ok(result.reasons.includes("citations"));
});

test("repeated news words activate an unknown article", function () {
  const result = shouldActivate({
    url: "https://townpaper.example/story",
    title: "Election night",
    excerpt:
      "The election result left the government and the president arguing about the campaign. " +
      filler,
  });
  assert.equal(result.activate, true);
  assert.ok(result.reasons.includes("news-text"));
});

test("an article element with one health word activates", function () {
  const result = shouldActivate({
    url: "https://clinic.example/post",
    title: "Visit",
    excerpt: "The therapy took months and the follow up was quiet. " + filler,
    hasArticle: true,
  });
  assert.equal(result.activate, true);
  assert.ok(result.reasons.includes("article"));
});

test("gemini on a google host still counts as chat", function () {
  const result = shouldActivate({
    url: "https://gemini.google.com/app",
    title: "Gemini",
    excerpt: "Here is a longer answer about the claim you pasted into the chat box just now.",
  });
  assert.equal(result.activate, true);
  assert.ok(result.reasons.includes("chat"));
});

importScripts("config.js");

const config = globalThis.VerifaiConfig;
const storage = (globalThis.browser || globalThis.chrome).storage;

function serverUrl() {
  return storage.sync.get({ serverUrl: config.DEFAULT_SERVER }).then(function (stored) {
    return config.normalizeServer(stored.serverUrl) || config.DEFAULT_SERVER;
  });
}

function checkText(text) {
  return serverUrl().then(function (server) {
    return fetch(server + "/api/v1/check", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ text: text }),
    }).then(function (response) {
      if (!response.ok) {
        return response.text().then(function (body) {
          throw new Error(body || "VerifAI returned " + response.status);
        });
      }
      return response.json();
    });
  });
}

(globalThis.browser || globalThis.chrome).runtime.onMessage.addListener(function (
  message,
  _sender,
  sendResponse,
) {
  if (!message || message.type !== "check") {
    return false;
  }
  checkText(String(message.text || ""))
    .then(function (body) {
      sendResponse({ ok: true, body: body });
    })
    .catch(function (error) {
      sendResponse({ ok: false, error: error.message });
    });
  return true;
});

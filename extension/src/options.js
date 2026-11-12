(function () {
  const config = globalThis.VerifaiConfig;
  const api = globalThis.browser || globalThis.chrome;
  const storage = api.storage;
  const input = document.querySelector("#server");
  const status = document.querySelector("#status");
  const form = document.querySelector("#server-form");

  storage.sync.get({ serverUrl: config.DEFAULT_SERVER }).then(function (stored) {
    input.value = config.normalizeServer(stored.serverUrl) || config.DEFAULT_SERVER;
  });

  form.addEventListener("submit", function (event) {
    event.preventDefault();
    const normalized = config.normalizeServer(input.value);
    if (!normalized) {
      status.textContent = "Use an http or https URL without a password.";
      return;
    }
    // permissions.request must run inside the click, before any await.
    const granted = config.needsPermission(normalized)
      ? api.permissions.request({ origins: [config.originPattern(normalized)] })
      : Promise.resolve(true);
    granted
      .then(function (ok) {
        if (!ok) {
          status.textContent = "Not saved. The browser needs permission to reach " + normalized + ".";
          return null;
        }
        return storage.sync.set({ serverUrl: normalized }).then(function () {
          input.value = normalized;
          status.textContent = "Saved " + normalized;
        });
      })
      .catch(function (error) {
        status.textContent = "Not saved: " + error.message;
      });
  });
})();

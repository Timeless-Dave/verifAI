(function () {
  const config = globalThis.VerifaiConfig;
  const storage = (globalThis.browser || globalThis.chrome).storage;
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
    storage.sync.set({ serverUrl: normalized }).then(function () {
      input.value = normalized;
      status.textContent = "Saved " + normalized;
    });
  });
})();

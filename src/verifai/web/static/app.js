(function () {
  var button = document.querySelector("[data-load-example]");
  var area = document.querySelector("#claim");
  var example = document.querySelector("#example-text");
  var results = document.querySelector("#results");

  if (button && area && example) {
    button.addEventListener("click", function () {
      area.value = JSON.parse(example.textContent);
      area.focus();
    });
  }
  if (area && area.form) {
    area.addEventListener("keydown", function (event) {
      if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
        event.preventDefault();
        area.form.requestSubmit();
      }
    });
  }

  // htmx does not swap error responses in, so a refused or failed check
  // would leave the page silent. Say what happened instead.
  function notice(text) {
    if (!results) {
      return;
    }
    var box = document.createElement("div");
    box.className = "notice tone-warning";
    box.setAttribute("role", "alert");
    var p = document.createElement("p");
    p.textContent = text;
    box.appendChild(p);
    results.replaceChildren(box);
  }

  document.body.addEventListener("htmx:responseError", function (event) {
    var status = event.detail.xhr ? event.detail.xhr.status : 0;
    if (status === 429) {
      notice("Too many checks from your network right now. Wait a minute, then try again.");
    } else if (status === 413) {
      notice("That paste is too long. Check a shorter piece at a time.");
    } else {
      notice("The check could not finish (error " + status + "). Try again in a moment.");
    }
  });
  document.body.addEventListener("htmx:sendError", function () {
    notice("Could not reach VerifAI. Check your connection and try again.");
  });
  document.body.addEventListener("htmx:afterSwap", function (event) {
    if (event.detail.target === results && results.firstElementChild) {
      results.firstElementChild.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  });
})();

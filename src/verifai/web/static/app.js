(function () {
  var button = document.querySelector("[data-load-example]");
  var area = document.querySelector("#claim");
  var example = document.querySelector("#example-text");
  var results = document.querySelector("#results");
  var toggle = document.querySelector("[data-theme-toggle]");

  function showTheme() {
    if (!toggle) {
      return;
    }
    var dark = document.documentElement.dataset.theme === "dark";
    var label = dark ? "Switch to light mode" : "Switch to dark mode";
    toggle.setAttribute("aria-label", label);
    toggle.setAttribute("title", label);
  }
  if (toggle) {
    showTheme();
    toggle.addEventListener("click", function () {
      var dark = document.documentElement.dataset.theme !== "dark";
      if (dark) {
        document.documentElement.dataset.theme = "dark";
      } else {
        delete document.documentElement.dataset.theme;
      }
      try {
        localStorage.setItem("verifai-theme", dark ? "dark" : "light");
      } catch (error) {
        // Private windows can refuse storage. The choice still applies now.
      }
      showTheme();
    });
  }

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

  // A follow-up batch refused at the edge pauses in place, with a button to
  // resume, instead of wiping the results already on the page.
  function pauseBatch(form, text) {
    form.classList.add("more-paused");
    form.setAttribute("hx-trigger", "submit");
    Array.prototype.slice.call(form.children).forEach(function (child) {
      if (child.tagName !== "INPUT") {
        child.remove();
      }
    });
    var label = document.createElement("span");
    label.className = "more-label";
    label.textContent = text;
    var resume = document.createElement("button");
    resume.type = "submit";
    resume.className = "ghost";
    resume.textContent = "Resume checking";
    form.append(label, resume);
    if (window.htmx) {
      window.htmx.process(form);
    }
  }

  document.body.addEventListener("htmx:responseError", function (event) {
    var status = event.detail.xhr ? event.detail.xhr.status : 0;
    var form = event.detail.elt && event.detail.elt.closest ? event.detail.elt.closest("[data-more]") : null;
    if (form) {
      pauseBatch(form, status === 429 ? "Paused: too many checks from your network. Wait a minute." : "Paused: the next batch could not finish.");
      return;
    }
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

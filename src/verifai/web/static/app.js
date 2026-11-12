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
    toggle.setAttribute("aria-checked", dark ? "true" : "false");
    toggle.setAttribute("title", dark ? "Switch to light mode" : "Switch to dark mode");
  }

  function applyTheme(dark) {
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
  }

  // Peel the old page back from the top-right corner toward the bottom-left.
  // The new theme shows through above the fold line while the old page still
  // covers the rest, so both are on screen until the fold has crossed it.
  function peelTo(dark) {
    var root = document.documentElement;
    var still = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (!document.startViewTransition || still) {
      applyTheme(dark);
      return;
    }
    var styles = getComputedStyle(root);
    var back = styles.getPropertyValue("--paper-2").trim() || "#e9e3d5";
    var light = styles.getPropertyValue("--sheet").trim() || "#f8f5ee";
    var width = window.innerWidth;
    var height = window.innerHeight;
    var travel = width + height;
    var peel = document.createElement("div");
    peel.className = "peel";
    peel.setAttribute("aria-hidden", "true");
    var flap = document.createElement("div");
    flap.className = "peel-flap";
    flap.style.setProperty("--flap-back", back);
    flap.style.setProperty("--flap-light", light);
    flap.style.setProperty("--flap-edge", "rgba(0, 0, 0, 0.3)");
    flap.style.setProperty("--flap-under", styles.getPropertyValue("--rule-soft").trim() || "#cfc7b5");
    var curl = Math.max(70, Math.min(150, Math.min(width, height) * 0.18));
    peel.appendChild(flap);

    function place(distance) {
      // The fold runs from (width - d, 0) to (width, d). The flap is the
      // peeled corner folded back over the fold, so it reaches (width - d, d).
      // Only a band beside the fold lifts, like paper curling, so the old
      // page stays visible past it.
      var side = distance * Math.SQRT2;
      var depth = Math.min(side / 2, curl);
      root.style.setProperty("--peel-d", distance + "px");
      flap.style.left = width - distance / 2 - side / 2 + "px";
      flap.style.top = distance / 2 + "px";
      flap.style.width = side + "px";
      flap.style.height = depth + "px";
      flap.style.clipPath =
        "polygon(0 0, 100% 0, calc(100% - " + depth + "px) 100%, " + depth + "px 100%)";
    }

    root.classList.add("peeling");
    root.style.setProperty("--peel-d", "0px");
    var transition = document.startViewTransition(function () {
      applyTheme(dark);
      document.body.appendChild(peel);
      place(0);
    });
    transition.ready.then(function () {
      var start = performance.now();
      var duration = 1100;
      function frame(now) {
        var t = Math.min(1, (now - start) / duration);
        var eased = t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
        place(eased * travel);
        if (t < 1) {
          requestAnimationFrame(frame);
        } else {
          finish();
        }
      }
      requestAnimationFrame(frame);
    }, finish);
    function finish() {
      transition.skipTransition();
    }
    transition.finished.then(cleanUp, cleanUp);
    function cleanUp() {
      root.classList.remove("peeling");
      root.style.removeProperty("--peel-d");
      if (peel.parentNode) {
        peel.parentNode.removeChild(peel);
      }
    }
  }

  if (toggle) {
    showTheme();
    toggle.addEventListener("click", function () {
      peelTo(document.documentElement.dataset.theme !== "dark");
    });
  }

  if (button && area && example) {
    button.addEventListener("click", function () {
      area.value = JSON.parse(example.textContent);
      area.focus();
    });
  }
  if (area) {
    // An empty box never reaches the server. Clear the last results so the
    // page does not look as if they answer the new, empty check.
    area.addEventListener("invalid", function () {
      notice("Paste some text first.");
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
    var progress = document.getElementById("claims-progress");
    if (progress) {
      progress.textContent = "Paused. Resume at the end of the claims list.";
    }
    if (window.htmx) {
      window.htmx.process(form);
    }
  }

  // What to tell someone who was rate limited: the server's own reason if
  // it sent one, else its Retry-After, else the firewall's one-minute window.
  function waitMessage(xhr) {
    try {
      var body = JSON.parse(xhr.responseText);
      if (body && typeof body.detail === "string") {
        return body.detail;
      }
    } catch (error) {
      // Not JSON, for example the edge firewall's own page.
    }
    var after = parseInt(xhr.getResponseHeader("Retry-After") || "", 10);
    var wait = after > 0 ? (after < 90 ? after + " seconds" : Math.ceil(after / 60) + " minutes") : "about a minute";
    return "Too many checks from your network right now. Try again in " + wait + ".";
  }

  document.body.addEventListener("htmx:responseError", function (event) {
    var status = event.detail.xhr ? event.detail.xhr.status : 0;
    var form = event.detail.elt && event.detail.elt.closest ? event.detail.elt.closest("[data-more]") : null;
    if (form) {
      pauseBatch(form, status === 429 ? "Paused. " + waitMessage(event.detail.xhr) : "Paused: the next batch could not finish.");
      return;
    }
    if (status === 429) {
      notice(waitMessage(event.detail.xhr));
    } else if (status === 413) {
      notice("That paste is too long. Check a shorter piece at a time.");
    } else {
      notice("The check could not finish (error " + status + "). Try again in a moment.");
    }
  });
  document.body.addEventListener("htmx:sendError", function (event) {
    var form = event.detail.elt && event.detail.elt.closest ? event.detail.elt.closest("[data-more]") : null;
    if (form) {
      pauseBatch(form, "Paused: VerifAI could not be reached. Check your connection.");
      return;
    }
    notice("Could not reach VerifAI. Check your connection and try again.");
  });
  document.body.addEventListener("htmx:afterSwap", function (event) {
    if (event.detail.target === results && results.firstElementChild) {
      results.firstElementChild.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  });
})();

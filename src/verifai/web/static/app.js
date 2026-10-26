(function () {
  var button = document.querySelector("[data-load-example]");
  var area = document.querySelector("#claim");
  var example = document.querySelector("#example-text");
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
})();

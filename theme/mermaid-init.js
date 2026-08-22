/* Book chrome and diagram renderer.
 *
 * Loads the vendored Mermaid build (no network dependency), renders every
 * ```mermaid block as a numbered figure whose caption is the bold thesis
 * line above it, applies the book's component/movement palette natively,
 * and re-renders when the reading theme toggles between light and dark.
 * Also upgrades chapter openers with part-aware eyebrows and styles the
 * part-divider pages.
 */

(function () {
  "use strict";

  var DARK_THEMES = ["navy", "ayu", "coal"];
  var BOOK_LABEL = "Inference Systems";

  /* mdBook copies non-markdown files from src/ verbatim, so the vendored
     bundle lives at <book-root>/assets/vendor/. path_to_root is mdBook's
     page-depth-relative prefix (a top-level const in the generated head). */
  function mermaidBundleUrl() {
    var root = typeof path_to_root === "string" ? path_to_root : "";
    return new URL(root + "assets/vendor/mermaid.min.js", document.baseURI);
  }

  function cssVar(name, fallback) {
    var value = getComputedStyle(document.documentElement)
      .getPropertyValue(name)
      .trim();
    return value || fallback;
  }

  function rootHasAny(classNames) {
    var roots = [document.documentElement, document.body];
    for (var i = 0; i < roots.length; i++) {
      if (!roots[i] || !roots[i].classList) continue;
      for (var j = 0; j < classNames.length; j++) {
        if (roots[i].classList.contains(classNames[j])) return true;
      }
    }
    return false;
  }

  function isDark() {
    return rootHasAny(DARK_THEMES);
  }

  /* ------------------------------------------------------------------ */
  /* Chapter openers                                                     */
  /* ------------------------------------------------------------------ */

  function sidebarPartOfActiveLink() {
    var currentPart = "";
    var found = null;
    var nodes = document.querySelectorAll(
      ".sidebar .part-title, .sidebar a"
    );
    for (var i = 0; i < nodes.length; i++) {
      var el = nodes[i];
      if (el.classList.contains("part-title")) {
        currentPart = el.textContent.replace(/^#+\s*/, "").trim();
      } else if (el.classList.contains("active")) {
        found = currentPart;
        break;
      }
    }
    return found;
  }

  function enhanceOpeners() {
    var main = document.querySelector(".content main") || document.querySelector("main");
    var h1 = main ? main.querySelector("h1") : null;
    if (!main || !h1 || h1.dataset.chrome === "done") return;
    h1.dataset.chrome = "done";

    var title = h1.textContent.trim();
    var chapterMatch = title.match(/^(\d+)\.\s+(.*)$/);
    var appendixMatch = title.match(/^Appendix ([A-G])\b\.?\s*(.*)$/i);
    var partMatch = title.match(/^Part\s+([IVX]+)\b(.*)$/i);
    var hasBodySections = !!main.querySelector("h2");

    if (partMatch && !hasBodySections) {
      main.classList.add("part-divider");
      var range = main.querySelector("p:last-of-type strong");
      if (range && /^Chapters\b/i.test(range.textContent.trim())) {
        range.parentElement.classList.add("divider-range");
      }
      return;
    }

    var eyebrow = null;
    var part = sidebarPartOfActiveLink();
    if (chapterMatch) {
      eyebrow = "Chapter " + chapterMatch[1];
      if (part) eyebrow += " · " + part;
      h1.textContent = title.replace(/^\d+\.\s+/, "");
    } else if (appendixMatch) {
      eyebrow = "Appendix " + appendixMatch[1].toUpperCase();
      h1.textContent = appendixMatch[2];
      if (appendixMatch[2] && !/^appendices$/i.test(part || "")) {
        eyebrow += " · " + part;
      } else if (appendixMatch[2]) {
        eyebrow += " · " + appendixMatch[2];
      }
    } else if (part) {
      eyebrow = part;
    } else {
      eyebrow = BOOK_LABEL;
    }

    var label = document.createElement("div");
    label.className = "opening-eyebrow";
    label.textContent = eyebrow.toUpperCase();
    h1.insertAdjacentElement("beforebegin", label);
    main.classList.add("has-opener");
  }

  /* ------------------------------------------------------------------ */
  /* Figures                                                             */
  /* ------------------------------------------------------------------ */

  var figures = [];

  function buildFigures() {
    var blocks = document.querySelectorAll("pre > code.language-mermaid");
    var counter = 0;
    blocks.forEach(function (code) {
      var pre = code.parentElement;
      var caption = null;
      var prev = pre.previousElementSibling;
      while (prev && prev.classList && prev.classList.contains("diagram-figure")) {
        prev = prev.previousElementSibling;
      }
      if (prev && prev.tagName === "P") {
        caption = prev.textContent.trim().replace(/[:：]\s*$/, "");
      }

      counter += 1;
      var figure = document.createElement("figure");
      figure.className = "diagram-figure";

      var box = document.createElement("div");
      box.className = "diagram-box";
      box.setAttribute("role", "img");
      box.setAttribute("aria-label", caption || "System block diagram");
      box.textContent = code.textContent;

      var cap = document.createElement("figcaption");
      var labelSpan = document.createElement("span");
      labelSpan.className = "fig-label";
      labelSpan.textContent = "Figure " + counter;
      cap.appendChild(labelSpan);
      if (caption) {
        cap.appendChild(document.createTextNode(caption));
      }

      figure.appendChild(box);
      figure.appendChild(cap);
      pre.replaceWith(figure);
      if (prev && prev.tagName === "P" && caption !== null) {
        prev.remove();
      }
      figures.push({ box: box, source: code.textContent });
    });
  }

  /* ------------------------------------------------------------------ */
  /* Rendering                                                           */
  /* ------------------------------------------------------------------ */

  var rendering = false;
  var pendingTheme = null;

  function themeConfig(dark) {
    return {
      startOnLoad: false,
      securityLevel: "strict",
      theme: "base",
      darkMode: dark,
      fontFamily: "'Inter', system-ui, sans-serif",
      themeVariables: {
        fontFamily: "'Inter', system-ui, sans-serif",
        fontSize: "13.5px",
        primaryColor: cssVar("--surface-raised", "#ffffff"),
        primaryTextColor: cssVar("--fg", "#18202d"),
        primaryBorderColor: cssVar("--trace-blue", "#245dcc"),
        lineColor: cssVar("--state-teal", "#087e78"),
        textColor: cssVar("--fg", "#18202d"),
        titleColor: cssVar("--fg", "#18202d"),
        edgeLabelBackground: cssVar("--surface-raised", "#ffffff"),
        clusterBkg: "transparent",
        clusterBorder: cssVar("--table-border-color", "#d8dee8"),
        mainBkg: cssVar("--surface-raised", "#ffffff"),
        nodeBorder: cssVar("--trace-blue", "#245dcc"),
        nodeTextColor: cssVar("--fg", "#18202d"),
        arrowheadColor: cssVar("--state-teal", "#087e78")
      },
      flowchart: {
        curve: "basis",
        htmlLabels: false,
        nodeSpacing: 22,
        rankSpacing: 40,
        useMaxWidth: true,
        padding: 6
      }
    };
  }

  function renderBoxes(boxes) {
    return window.mermaid.run({ nodes: boxes, suppressErrors: false });
  }

  function failFigures(message) {
    console.error(message);
    figures.forEach(function (item) {
      item.box.classList.add("diagram-error");
      item.box.removeAttribute("role");
      item.box.removeAttribute("aria-label");
      item.box.textContent = item.source;
    });
  }

  function renderAll(dark) {
    if (!figures.length) return Promise.resolve();
    if (rendering) {
      pendingTheme = dark;
      return Promise.resolve();
    }
    rendering = true;
    window.mermaid.initialize(themeConfig(dark));
    return renderBoxes(figures.map(function (item) { return item.box; }))
      .catch(function (error) {
        failFigures("Unable to render block diagrams: " + error);
      })
      .then(function () {
        rendering = false;
        if (pendingTheme !== null && pendingTheme !== dark) {
          var next = pendingTheme;
          pendingTheme = null;
          rerenderFor(next);
        } else {
          pendingTheme = null;
        }
      });
  }

  function rerenderFor(dark) {
    figures.forEach(function (item) {
      /* mermaid marks processed nodes and skips them on later runs */
      item.box.removeAttribute("data-processed");
      item.box.classList.remove("diagram-error");
      item.box.textContent = item.source;
    });
    return renderAll(dark);
  }

  function loadMermaid() {
    return new Promise(function (resolve, reject) {
      if (window.mermaid) {
        resolve(window.mermaid);
        return;
      }
      var tag = document.createElement("script");
      tag.src = mermaidBundleUrl().href;
      tag.async = true;
      tag.onload = function () {
        if (window.mermaid) resolve(window.mermaid);
        else reject(new Error("mermaid global missing after load"));
      };
      tag.onerror = function () {
        reject(new Error("vendored mermaid.min.js failed to load from " + tag.src));
      };
      document.head.appendChild(tag);
    });
  }

  /* ------------------------------------------------------------------ */
  /* Boot                                                                */
  /* ------------------------------------------------------------------ */

  document.addEventListener("DOMContentLoaded", function () {
    enhanceOpeners();
    buildFigures();
    if (!figures.length) return;

    loadMermaid()
      .then(function () {
        return renderAll(isDark());
      })
      .catch(function (error) {
        failFigures(error && error.message ? error.message : String(error));
      });

    var lastDark = isDark();
    var observer = new MutationObserver(function () {
      var dark = isDark();
      if (dark !== lastDark) {
        lastDark = dark;
        if (window.mermaid) rerenderFor(dark);
      }
    });
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["class"]
    });
    if (document.body) {
      observer.observe(document.body, {
        attributes: true,
        attributeFilter: ["class"]
      });
    }
  });
})();

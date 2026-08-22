/* Book chrome and diagram renderer.
 *
 * Loads the vendored Mermaid build (no network dependency), renders every
 * ```mermaid block as a numbered figure whose caption is the bold thesis
 * line above it, and applies the book's single signal-blue palette natively.
 * Also upgrades chapter openers with part-aware eyebrows and styles the
 * part-divider pages.
 */

(function () {
  "use strict";

  var BOOK_LABEL = "Inference Systems";
  var THEME_CLASSES = ["rust", "coal", "navy", "ayu"];

  function enforceSingleTheme() {
    [document.documentElement, document.body].forEach(function (root) {
      if (!root || !root.classList) return;
      THEME_CLASSES.forEach(function (name) { root.classList.remove(name); });
      root.classList.add("light");
    });
    try {
      localStorage.setItem("mdbook-theme", "light");
    } catch (_error) {
      /* Storage can be unavailable in hardened or private contexts. */
    }
  }

  enforceSingleTheme();

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

    if (chapterMatch) {
      main.dataset.chapterNumber = chapterMatch[1];
      main.dataset.chapterTitle = chapterMatch[2];
    }

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
      h1.textContent = chapterMatch[2];
      main.dataset.partShort =
        part ? part.split("—")[0].trim() : "";
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
    } else if (title !== BOOK_LABEL) {
      eyebrow = BOOK_LABEL;
    }

    if (!eyebrow) return;

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
  var LABEL_MIN_CHARS = 14;
  var LABEL_MAX_CHARS = 22;

  function balanceLabel(label, maxChars, minChars) {
    maxChars = maxChars || LABEL_MAX_CHARS;
    minChars = minChars || LABEL_MIN_CHARS;
    var words = label.trim().split(/\s+/);
    var lines = [];
    var line = "";

    words.forEach(function (word) {
      var candidate = line ? line + " " + word : word;
      if (line && candidate.length > maxChars) {
        lines.push(line);
        line = word;
      } else {
        line = candidate;
      }
    });
    if (line) lines.push(line);

    if (lines.length === 1 && lines[0].length < minChars) {
      var missing = minChars - lines[0].length;
      var before = Math.floor(missing / 2);
      var after = missing - before;
      lines[0] = "\u00a0".repeat(before) + lines[0] + "\u00a0".repeat(after);
    }

    return "`" + lines.join("\n") + "`";
  }

  function normalizeDiagramSource(source) {
    return source.replace(
      /(\b[A-Za-z][\w-]*)(\[|\{)"([^"]+)"(\]|\})/g,
      function (_match, id, open, label, close) {
        var balanced = open === "{"
          ? balanceLabel(label, 15, 12)
          : balanceLabel(label);
        return id + open + '"' + balanced + '"' + close;
      }
    );
  }

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
      var main = document.querySelector(".content main") || document.querySelector("main");
      var chapterNo = main ? main.dataset.chapterNumber : null;
      var figureLabel = chapterNo
        ? "Figure " + chapterNo + "." + counter
        : "Figure " + counter;
      var figure = document.createElement("figure");
      figure.className = "diagram-figure";

      var box = document.createElement("div");
      box.className = "diagram-box";
      box.setAttribute("role", "img");
      box.setAttribute("aria-label", caption || "System block diagram");
      var originalSource = code.textContent;
      var renderSource = normalizeDiagramSource(originalSource);
      box.textContent = renderSource;

      var cap = document.createElement("figcaption");
      var labelSpan = document.createElement("span");
      labelSpan.className = "fig-label";
      labelSpan.textContent = figureLabel;
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
      figures.push({
        box: box,
        figure: figure,
        originalSource: originalSource,
        renderSource: renderSource
      });
    });
  }

  /* ------------------------------------------------------------------ */
  /* Rendering                                                           */
  /* ------------------------------------------------------------------ */

  function themeConfig() {
    return {
      startOnLoad: false,
      securityLevel: "strict",
      theme: "base",
      darkMode: false,
      fontFamily: "'Inter', system-ui, sans-serif",
      themeVariables: {
        fontFamily: "'Inter', system-ui, sans-serif",
        fontSize: "13.5px",
        primaryColor: cssVar("--surface-raised", "#ffffff"),
        primaryTextColor: cssVar("--fg", "#172033"),
        primaryBorderColor: cssVar("--signal", "#2458d3"),
        lineColor: cssVar("--signal", "#2458d3"),
        textColor: cssVar("--fg", "#172033"),
        titleColor: cssVar("--fg", "#172033"),
        edgeLabelBackground: cssVar("--surface-raised", "#ffffff"),
        clusterBkg: "transparent",
        clusterBorder: cssVar("--table-border-color", "#d9e1f0"),
        mainBkg: cssVar("--surface-raised", "#ffffff"),
        nodeBorder: cssVar("--signal", "#2458d3"),
        nodeTextColor: cssVar("--fg", "#172033"),
        arrowheadColor: cssVar("--signal", "#2458d3")
      },
      flowchart: {
        curve: "stepAfter",
        diagramPadding: 8,
        htmlLabels: false,
        inheritDir: true,
        nodeSpacing: 34,
        rankSpacing: 48,
        useMaxWidth: true,
        padding: 12,
        wrappingWidth: 154
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
      item.box.textContent = item.originalSource;
    });
  }

  function classifyFigures() {
    figures.forEach(function (item) {
      var svg = item.box.querySelector("svg");
      if (!svg) return;
      var viewBox = (svg.getAttribute("viewBox") || "")
        .trim()
        .split(/\s+/)
        .map(Number);
      var aspect = viewBox.length === 4 && viewBox[3] > 0
        ? viewBox[2] / viewBox[3]
        : 1;
      var nodes = item.box.querySelectorAll(".node").length;
      var isWide = nodes >= 5 && aspect >= 4;
      var isPannable = isWide || window.matchMedia("(max-width: 700px)").matches;
      item.figure.classList.toggle("diagram-wide", isWide);
      item.figure.classList.toggle("diagram-pannable", isPannable);
      if (isPannable && !item.figure.querySelector(".diagram-pan-hint")) {
        var hint = document.createElement("span");
        hint.className = "diagram-pan-hint";
        hint.setAttribute("aria-hidden", "true");
        hint.textContent = "Pan diagram →";
        item.figure.appendChild(hint);
      }
      if (isWide) {
        var naturalWidth = Math.min(1200, Math.max(960, viewBox[2]));
        item.figure.style.setProperty(
          "--diagram-min-width",
          Math.round(naturalWidth) + "px"
        );
      }
    });
  }

  function renderAll() {
    if (!figures.length) return Promise.resolve();
    window.mermaid.initialize(themeConfig());
    return renderBoxes(figures.map(function (item) { return item.box; }))
      .then(classifyFigures)
      .catch(function (error) {
        failFigures("Unable to render block diagrams: " + error);
      });
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
  /* Reading chrome: cross-links, pager, running head, progress          */
  /* ------------------------------------------------------------------ */

  function sidebarPageMap() {
    /* Ordered list of {href, text} for every content page in the sidebar. */
    var pages = [];
    document.querySelectorAll(".sidebar a").forEach(function (a) {
      var href = a.getAttribute("href");
      if (!href || href.startsWith("#")) return;
      pages.push({ href: href, text: a.textContent.trim(), el: a });
    });
    return pages;
  }

  function enhanceCrossReferences() {
    var main = document.querySelector(".content main");
    if (!main || main.dataset.xref === "done") return;
    main.dataset.xref = "done";

    var chapterHrefs = {};
    var appendixHrefs = {};
    sidebarPageMap().forEach(function (page) {
      var m = page.text.match(/^(\d+)\.\s/);
      if (m) chapterHrefs[m[1]] = page.href;
      var a = page.text.match(/^([A-G])\.\s/);
      if (a) appendixHrefs[a[1]] = page.href;
    });
    if (Object.keys(chapterHrefs).length === 0) return;

    var skip = { A: 1, CODE: 1, PRE: 1, SCRIPT: 1, STYLE: 1, BUTTON: 1, H1: 1 };
    var walker = document.createTreeWalker(main, NodeFilter.SHOW_TEXT, {
      acceptNode: function (node) {
        var parent = node.parentNode;
        if (!parent || skip[parent.nodeName]) return NodeFilter.FILTER_REJECT;
        if (parent.closest && parent.closest("a")) return NodeFilter.FILTER_REJECT;
        return /\bChapter \d|\bAppendix [A-G]\b/.test(node.nodeValue)
          ? NodeFilter.FILTER_ACCEPT
          : NodeFilter.FILTER_REJECT;
      }
    });

    var nodes = [];
    while (walker.nextNode()) nodes.push(walker.currentNode);

    nodes.forEach(function (node) {
      var text = node.nodeValue;
      var pattern = /\bChapter (\d{1,2})\b|\bAppendix ([A-G])\b/g;
      var match;
      var frag = document.createDocumentFragment();
      var last = 0;
      var changed = false;
      while ((match = pattern.exec(text)) !== null) {
        var href = match[1] !== undefined
          ? chapterHrefs[match[1]]
          : appendixHrefs[match[2]];
        if (!href) continue;
        changed = true;
        frag.appendChild(document.createTextNode(text.slice(last, match.index)));
        var link = document.createElement("a");
        link.href = href;
        link.textContent = match[0];
        link.className = "xref";
        frag.appendChild(link);
        last = match.index + match[0].length;
      }
      if (!changed) return;
      frag.appendChild(document.createTextNode(text.slice(last)));
      node.parentNode.replaceChild(frag, node);
    });
  }

  function styleSidebar() {
    /* Part-divider links replace mdBook's duplicate, non-link part labels. */
    document.querySelectorAll(".sidebar a").forEach(function (a) {
      var href = a.getAttribute("href") || "";
      if (/^(\.\.\/)*(part-\d+\/)?index\.html$/.test(href.split("#")[0]) ||
          /part-\d+\/index\.html$/.test(href)) {
        a.classList.add("divider-link");
      }
    });
  }

  function buildPager() {
    var main = document.querySelector(".content main");
    if (!main || main.dataset.pager === "done") return;
    main.dataset.pager = "done";

    var pages = sidebarPageMap();
    var index = -1;
    for (var i = 0; i < pages.length; i++) {
      if (pages[i].el.classList.contains("active")) { index = i; break; }
    }
    if (index === -1) return;

    var pager = document.createElement("nav");
    pager.className = "page-pager";
    pager.setAttribute("aria-label", "Chapter navigation");

    [["pager-prev", index - 1, "Previous"], ["pager-next", index + 1, "Next"]]
      .forEach(function (spec) {
        var cls = spec[0], at = spec[1], word = spec[2];
        if (at < 0 || at >= pages.length) {
          var spacer = document.createElement("div");
          spacer.className = cls + " pager-empty";
          spacer.setAttribute("aria-hidden", "true");
          pager.appendChild(spacer);
          return;
        }
        var a = document.createElement("a");
        a.className = cls;
        a.href = pages[at].href;
        var dir = document.createElement("span");
        dir.className = "pager-word";
        dir.textContent = word;
        var label = document.createElement("strong");
        label.textContent = pages[at].text;
        a.appendChild(dir);
        a.appendChild(label);
        pager.appendChild(a);
      });

    main.appendChild(pager);
  }

  function initRunningHead() {
    var menuTitle = document.querySelector(".menu-title");
    var main = document.querySelector(".content main");
    var h1 = main ? main.querySelector("h1") : null;
    if (!menuTitle || !main || !h1 || menuTitle.dataset.head === "done") return;
    menuTitle.dataset.head = "done";
    var bookTitle = menuTitle.textContent;

    var chapterTitle = main.dataset.chapterTitle;
    var partShort = main.dataset.partShort;
    var running = null;
    if (chapterTitle) {
      running = partShort
        ? partShort + " · " + chapterTitle
        : chapterTitle;
    }

    if (!running || !("IntersectionObserver" in window)) return;
    new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        menuTitle.textContent = entry.isIntersecting
          ? bookTitle
          : running;
      });
    }).observe(h1);
  }

  function initProgressBar() {
    if (document.getElementById("reading-progress")) return;
    var bar = document.createElement("div");
    bar.id = "reading-progress";
    bar.setAttribute("aria-hidden", "true");
    document.body.appendChild(bar);

    var ticking = false;
    function update() {
      ticking = false;
      var doc = document.documentElement;
      var max = doc.scrollHeight - window.innerHeight;
      var frac = max > 0 ? Math.min(1, window.scrollY / max) : 0;
      bar.style.transform = "scaleX(" + frac + ")";
    }
    window.addEventListener("scroll", function () {
      if (!ticking) {
        ticking = true;
        window.requestAnimationFrame(update);
      }
    }, { passive: true });
    update();
  }

  function tagCodeLanguages() {
    document.querySelectorAll("pre > code[class*='language-']").forEach(
      function (code) {
        var match = code.className.match(/language-([\w+#.-]+)/);
        var lang = match ? match[1].toLowerCase() : null;
        if (!lang || lang === "text" || lang === "mermaid") return;
        code.parentElement.dataset.lang = lang;
      }
    );
  }

  function alignNumericCells() {
    var numeric = /^[\s≈~$]*[-+]?[\d.,]+(?:\s*(?:ms|µs|us|s|GiB|MiB|KiB|GB|MB|KB|%|x|×|bps|GB\/s|GiB\/s|requests\/s?|req\/s?|tokens?|steps?|layers?|ranks?|experts?|heads?|bytes?)?\s*)[*]?[\s—–…]*$/;
    document.querySelectorAll(".content main td").forEach(function (td) {
      var t = td.textContent.trim();
      if (t && numeric.test(t)) td.classList.add("num");
    });
  }

  function addSkipLink() {
    if (document.querySelector(".skip-link")) return;
    var main = document.querySelector("main");
    if (!main) return;
    if (!main.id) main.id = "content-main";
    main.setAttribute("tabindex", "-1");
    var skip = document.createElement("a");
    skip.className = "skip-link";
    skip.href = "#" + main.id;
    skip.textContent = "Skip to content";
    document.body.insertBefore(skip, document.body.firstChild);
  }

  /* ------------------------------------------------------------------ */
  /* Boot                                                                */
  /* ------------------------------------------------------------------ */

  document.addEventListener("DOMContentLoaded", function () {
    enforceSingleTheme();
    addSkipLink();
    enhanceOpeners();
    buildFigures();
    styleSidebar();
    enhanceCrossReferences();
    buildPager();
    initRunningHead();
    initProgressBar();
    tagCodeLanguages();
    alignNumericCells();
    if (!figures.length) return;

    loadMermaid()
      .then(function () {
        return renderAll();
      })
      .catch(function (error) {
        failFigures(error && error.message ? error.message : String(error));
      });
  });
})();

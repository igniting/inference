/* Fixed-geometry block diagrams for the book.
 *
 * The manuscript keeps compact flowchart source because it is
 * readable in Git. This renderer parses the small subset the book uses, asks
 * Dagre only for coordinates, and draws the final SVG itself. Node dimensions,
 * label wrapping, decision shapes, connectors, and page-fit behavior therefore
 * stay under the book's control.
 */

(function () {
  "use strict";

  var SVG_NS = "http://www.w3.org/2000/svg";
  var TARGET_WIDTH = 800;
  var RECT_WIDTH = 176;
  var DECISION_WIDTH = 184;
  var NODE_LINE_HEIGHT = 17;

  function pathToRoot() {
    return typeof path_to_root === "string" ? path_to_root : "";
  }

  function dagreBundleUrl() {
    return new URL(
      pathToRoot() + "assets/vendor/dagre.min.js",
      document.baseURI
    );
  }

  function svgElement(name, attrs) {
    var element = document.createElementNS(SVG_NS, name);
    Object.keys(attrs || {}).forEach(function (key) {
      element.setAttribute(key, attrs[key]);
    });
    return element;
  }

  function wrapWords(text, limit) {
    var words = text.trim().split(/\s+/);
    var lines = [];
    var line = "";
    words.forEach(function (word) {
      var candidate = line ? line + " " + word : word;
      if (line && candidate.length > limit) {
        lines.push(line);
        line = word;
      } else {
        line = candidate;
      }
    });
    if (line) lines.push(line);
    return lines.length ? lines : [text];
  }

  function nodeDefinitionPattern() {
    return /\b([A-Za-z][\w-]*)(\[|\{)"([^"]+)"(\]|\})/g;
  }

  function parseDiagram(source) {
    var lines = source.split(/\r?\n/);
    var direction = "TB";
    var nodes = new Map();
    var groups = [];
    var groupStack = [];

    function ensureNode(id, label, shape) {
      if (!nodes.has(id)) {
        nodes.set(id, {
          id: id,
          label: label || id,
          shape: shape || "block",
          groups: groupStack.slice()
        });
      } else if (label) {
        var existing = nodes.get(id);
        existing.label = label;
        existing.shape = shape || existing.shape;
        groupStack.forEach(function (groupId) {
          if (existing.groups.indexOf(groupId) === -1) {
            existing.groups.push(groupId);
          }
        });
      }
      return nodes.get(id);
    }

    lines.forEach(function (rawLine) {
      var line = rawLine.trim();
      var flow = line.match(/^flowchart\s+(LR|RL|TB|BT)\b/i);
      if (flow) {
        direction = flow[1].toUpperCase();
        return;
      }
      var subgraph = line.match(/^subgraph\s+([A-Za-z][\w-]*)(?:\["([^"]+)"\])?/);
      if (subgraph) {
        groups.push({ id: subgraph[1], label: subgraph[2] || subgraph[1] });
        groupStack.push(subgraph[1]);
        return;
      }
      if (/^end\b/.test(line)) {
        groupStack.pop();
        return;
      }
      if (/^(direction|%%)/.test(line)) return;

      var pattern = nodeDefinitionPattern();
      var match;
      while ((match = pattern.exec(line)) !== null) {
        ensureNode(
          match[1],
          match[3],
          match[2] === "{" ? "decision" : "block"
        );
      }
    });

    var edges = [];
    lines.forEach(function (rawLine) {
      var line = rawLine.trim();
      if (!line || /^(flowchart|subgraph|direction|end\b|%%)/.test(line)) {
        return;
      }

      line = line.replace(nodeDefinitionPattern(), function (_m, id) {
        return id;
      });
      line = line
        .replace(/-\.\s*"([^"]*)"\s*\.->/g, "\u001edotted\t$1\u001e")
        .replace(/-->\|([^|]+)\|/g, "\u001esolid\t$1\u001e")
        .replace(/-->/g, "\u001esolid\t\u001e")
        .replace(/---/g, "\u001eline\t\u001e");

      var parts = line.split("\u001e").map(function (part) {
        return part.trim();
      }).filter(Boolean);
      if (parts.length < 3) return;

      for (var i = 0; i + 2 < parts.length; i += 2) {
        var fromMatch = parts[i].match(/[A-Za-z][\w-]*/);
        var descriptor = parts[i + 1].split("\t");
        var toMatch = parts[i + 2].match(/[A-Za-z][\w-]*/);
        if (!fromMatch || !toMatch) continue;
        ensureNode(fromMatch[0]);
        ensureNode(toMatch[0]);
        edges.push({
          from: fromMatch[0],
          to: toMatch[0],
          kind: descriptor[0] || "solid",
          label: descriptor.slice(1).join("\t").trim()
        });
      }
    });

    return {
      direction: direction,
      nodes: Array.from(nodes.values()),
      edges: edges,
      groups: groups
    };
  }

  function layoutModel(model, direction, compact) {
    var graph = new window.dagre.graphlib.Graph({ multigraph: true })
      .setGraph({
        rankdir: direction,
        ranker: "network-simplex",
        nodesep: compact ? 8 : (direction === "TB" || direction === "BT" ? 28 : 24),
        edgesep: 14,
        ranksep: compact ? 34 : (direction === "TB" || direction === "BT" ? 42 : 50),
        marginx: compact ? 16 : 24,
        marginy: compact ? 20 : 26
      })
      .setDefaultEdgeLabel(function () { return {}; });

    model.nodes.forEach(function (node) {
      var wrapLimit = compact
        ? (node.shape === "decision" ? 14 : 12)
        : (node.shape === "decision" ? 17 : 22);
      var lines = wrapWords(node.label, wrapLimit);
      var width = compact
        ? (node.shape === "decision" ? 104 : 92)
        : (node.shape === "decision" ? DECISION_WIDTH : RECT_WIDTH);
      var height = Math.max(
        node.shape === "decision" ? 64 : 50,
        lines.length * NODE_LINE_HEIGHT + (node.shape === "decision" ? 28 : 24)
      );
      graph.setNode(node.id, {
        id: node.id,
        label: node.label,
        lines: lines,
        shape: node.shape,
        groups: node.groups,
        width: width,
        height: height
      });
    });

    model.edges.forEach(function (edge, index) {
      var lines = edge.label ? wrapWords(edge.label, compact ? 12 : 18) : [];
      graph.setEdge(
        edge.from,
        edge.to,
        {
          kind: edge.kind,
          label: edge.label,
          lines: lines,
          width: lines.length
            ? Math.min(132, Math.max.apply(null, lines.map(function (line) {
                return line.length * 6.2 + 14;
              })))
            : 0,
          height: lines.length ? lines.length * 14 + 8 : 0
        },
        "edge-" + index
      );
    });

    window.dagre.layout(graph);
    return graph;
  }

  function chooseLayout(model, targetWidth) {
    var preferred = layoutModel(model, model.direction, false);
    var preferredBox = preferred.graph();
    if (preferredBox.width <= targetWidth) return preferred;

    var alternateDirection = /^(LR|RL)$/.test(model.direction) ? "TB" : "LR";
    var alternate = layoutModel(model, alternateDirection, false);
    var alternateBox = alternate.graph();
    if (alternateBox.width <= targetWidth) return alternate;
    if (targetWidth < 520 && alternateBox.width <= targetWidth * 1.12) {
      return alternate;
    }

    if (targetWidth < 520) {
      var compactPreferred = layoutModel(model, model.direction, true);
      var compactAlternate = layoutModel(model, alternateDirection, true);
      var compactPreferredBox = compactPreferred.graph();
      var compactAlternateBox = compactAlternate.graph();
      if (compactPreferredBox.width <= targetWidth) return compactPreferred;
      if (compactAlternateBox.width <= targetWidth) return compactAlternate;
      return compactAlternateBox.width < compactPreferredBox.width
        ? compactAlternate
        : compactPreferred;
    }

    return alternateBox.width < preferredBox.width ? alternate : preferred;
  }

  function appendText(parent, lines, x, y, className) {
    var text = svgElement("text", {
      x: x,
      y: y - ((lines.length - 1) * NODE_LINE_HEIGHT) / 2,
      class: className,
      "text-anchor": "middle",
      "dominant-baseline": "middle"
    });
    lines.forEach(function (line, index) {
      var span = svgElement("tspan", {
        x: x,
        dy: index === 0 ? 0 : NODE_LINE_HEIGHT
      });
      span.textContent = line;
      text.appendChild(span);
    });
    parent.appendChild(text);
  }

  function edgePath(points) {
    if (!points || !points.length) return "";
    return points.map(function (point, index) {
      return (index === 0 ? "M" : "L") +
        point.x.toFixed(1) + " " + point.y.toFixed(1);
    }).join(" ");
  }

  function renderGroups(svg, graph, model) {
    model.groups.forEach(function (group) {
      var members = graph.nodes().map(function (id) { return graph.node(id); })
        .filter(function (node) {
          return node.groups && node.groups.indexOf(group.id) !== -1;
        });
      if (!members.length) return;
      var left = Math.min.apply(null, members.map(function (n) { return n.x - n.width / 2; })) - 10;
      var right = Math.max.apply(null, members.map(function (n) { return n.x + n.width / 2; })) + 10;
      var top = Math.min.apply(null, members.map(function (n) { return n.y - n.height / 2; })) - 14;
      var bottom = Math.max.apply(null, members.map(function (n) { return n.y + n.height / 2; })) + 8;
      svg.appendChild(svgElement("rect", {
        x: left,
        y: top,
        width: right - left,
        height: bottom - top,
        rx: 8,
        class: "block-group"
      }));
      var label = svgElement("text", {
        x: left + 8,
        y: top + 10,
        class: "block-group-label"
      });
      label.textContent = group.label;
      svg.appendChild(label);
    });
  }

  function renderDiagram(item, index) {
    var model = parseDiagram(item.source);
    if (!model.nodes.length) throw new Error("Diagram contains no nodes");
    var availableWidth = Math.max(240, item.box.clientWidth - 36);
    var targetWidth = Math.min(TARGET_WIDTH, availableWidth);
    var graph = chooseLayout(model, targetWidth);
    var bounds = graph.graph();
    var width = Math.ceil(bounds.width);
    var height = Math.ceil(bounds.height);
    var markerId = "block-arrow-" + index;
    var svg = svgElement("svg", {
      viewBox: "0 0 " + width + " " + height,
      role: "img",
      "aria-labelledby": "block-title-" + index + " block-desc-" + index,
      class: "block-diagram",
      preserveAspectRatio: "xMidYMid meet"
    });
    svg.style.maxWidth = width + "px";

    var title = svgElement("title", { id: "block-title-" + index });
    title.textContent = item.caption || "System block diagram";
    svg.appendChild(title);
    var desc = svgElement("desc", { id: "block-desc-" + index });
    desc.textContent = model.nodes.map(function (node) { return node.label; }).join(", ");
    svg.appendChild(desc);

    var defs = svgElement("defs");
    var marker = svgElement("marker", {
      id: markerId,
      viewBox: "0 0 8 8",
      refX: 7,
      refY: 4,
      markerWidth: 6,
      markerHeight: 6,
      orient: "auto"
    });
    marker.appendChild(svgElement("path", { d: "M0 0 L8 4 L0 8 Z" }));
    defs.appendChild(marker);
    svg.appendChild(defs);

    renderGroups(svg, graph, model);

    graph.edges().forEach(function (edgeRef) {
      var edge = graph.edge(edgeRef);
      var attrs = {
        d: edgePath(edge.points),
        class: "block-edge block-edge-" + edge.kind,
        fill: "none"
      };
      if (edge.kind !== "line") attrs["marker-end"] = "url(#" + markerId + ")";
      svg.appendChild(svgElement("path", attrs));
      if (edge.lines && edge.lines.length) {
        var labelGroup = svgElement("g", { class: "block-edge-label" });
        labelGroup.appendChild(svgElement("rect", {
          x: edge.x - edge.width / 2,
          y: edge.y - edge.height / 2,
          width: edge.width,
          height: edge.height,
          rx: 3
        }));
        appendText(labelGroup, edge.lines, edge.x, edge.y, "block-edge-text");
        svg.appendChild(labelGroup);
      }
    });

    graph.nodes().forEach(function (id) {
      var node = graph.node(id);
      var group = svgElement("g", {
        class: "block-node block-node-" + node.shape,
        transform: "translate(" + node.x + " " + node.y + ")"
      });
      if (node.shape === "decision") {
        var inset = 16;
        group.appendChild(svgElement("polygon", {
          points: [
            -node.width / 2 + inset + "," + -node.height / 2,
            node.width / 2 - inset + "," + -node.height / 2,
            node.width / 2 + ",0",
            node.width / 2 - inset + "," + node.height / 2,
            -node.width / 2 + inset + "," + node.height / 2,
            -node.width / 2 + ",0"
          ].join(" ")
        }));
      } else {
        group.appendChild(svgElement("rect", {
          x: -node.width / 2,
          y: -node.height / 2,
          width: node.width,
          height: node.height,
          rx: 6
        }));
      }
      appendText(group, node.lines, 0, 0, "block-node-label");
      svg.appendChild(group);
    });

    item.box.replaceChildren(svg);
    item.layoutMode = availableWidth < 520 ? "narrow" : "wide";
  }

  function failFigure(item, error) {
    item.box.classList.add("diagram-error");
    item.box.removeAttribute("role");
    item.box.textContent = item.source;
    console.error("Unable to render block diagram", error);
  }

  function loadDagre() {
    return new Promise(function (resolve, reject) {
      if (window.dagre) {
        resolve(window.dagre);
        return;
      }
      var tag = document.createElement("script");
      tag.src = dagreBundleUrl().href;
      tag.async = true;
      tag.onload = function () {
        if (window.dagre) resolve(window.dagre);
        else reject(new Error("Dagre global missing after load"));
      };
      tag.onerror = function () {
        reject(new Error("Vendored Dagre bundle failed to load"));
      };
      document.head.appendChild(tag);
    });
  }

  function mount() {
    var blocks = document.querySelectorAll("pre > code.language-blockdiag");
    var items = [];
    var counter = 0;
    blocks.forEach(function (code) {
      var pre = code.parentElement;
      var previous = pre.previousElementSibling;
      var caption = previous && previous.tagName === "P"
        ? previous.textContent.trim().replace(/[:：]\s*$/, "")
        : "";
      counter += 1;
      var main = document.querySelector(".content main") || document.querySelector("main");
      var chapterNo = main ? main.dataset.chapterNumber : null;
      var label = chapterNo
        ? "Figure " + chapterNo + "." + counter
        : "Figure " + counter;

      var figure = document.createElement("figure");
      figure.className = "diagram-figure diagram-custom";
      var box = document.createElement("div");
      box.className = "diagram-box";
      box.setAttribute("role", "img");
      box.setAttribute("aria-label", caption || "System block diagram");
      var figcaption = document.createElement("figcaption");
      var labelSpan = document.createElement("span");
      labelSpan.className = "fig-label";
      labelSpan.textContent = label;
      figcaption.appendChild(labelSpan);
      if (caption) figcaption.appendChild(document.createTextNode(caption));
      figure.appendChild(box);
      figure.appendChild(figcaption);
      pre.replaceWith(figure);
      if (previous && previous.tagName === "P") previous.remove();
      items.push({
        box: box,
        caption: caption,
        source: code.textContent
      });
    });

    if (!items.length) return Promise.resolve();
    return loadDagre()
      .then(function () {
        items.forEach(function (item, index) {
          try {
            renderDiagram(item, index + 1);
          } catch (error) {
            failFigure(item, error);
          }
        });
        var resizeTimer;
        window.addEventListener("resize", function () {
          window.clearTimeout(resizeTimer);
          resizeTimer = window.setTimeout(function () {
            items.forEach(function (item, index) {
              var mode = item.box.clientWidth - 36 < 520 ? "narrow" : "wide";
              if (item.layoutMode !== mode) {
                try {
                  renderDiagram(item, index + 1);
                } catch (error) {
                  failFigure(item, error);
                }
              }
            });
          }, 120);
        }, { passive: true });
      })
      .catch(function (error) {
        items.forEach(function (item) { failFigure(item, error); });
      });
  }

  window.InferenceDiagramRenderer = { mount: mount };
})();

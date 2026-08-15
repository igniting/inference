document.addEventListener("DOMContentLoaded", async () => {
  const codeBlocks = document.querySelectorAll("pre code.language-mermaid");
  if (codeBlocks.length === 0) return;

  const diagrams = [];
  for (const code of codeBlocks) {
    const caption = code.parentElement.previousElementSibling?.textContent?.trim();
    const container = document.createElement("div");
    container.className = "mermaid diagram";
    container.setAttribute("role", "img");
    container.setAttribute("aria-label", caption || "System block diagram");
    container.textContent = code.textContent;
    code.parentElement.replaceWith(container);
    diagrams.push(container);
  }

  try {
    const { default: mermaid } = await import(
      "https://cdn.jsdelivr.net/npm/mermaid@11.16.0/dist/mermaid.esm.min.mjs"
    );
    mermaid.initialize({
      startOnLoad: false,
      securityLevel: "strict",
      theme: "base",
      fontFamily: "IBM Plex Sans, system-ui, sans-serif",
      flowchart: {
        curve: "basis",
        htmlLabels: false,
        nodeSpacing: 28,
        rankSpacing: 38,
        useMaxWidth: true,
      },
    });
    await mermaid.run({ nodes: diagrams, suppressErrors: false });
  } catch (error) {
    for (const diagram of diagrams) {
      diagram.classList.add("diagram-error");
      diagram.removeAttribute("role");
      diagram.removeAttribute("aria-label");
    }
    console.error("Unable to render block diagrams", error);
  }
});

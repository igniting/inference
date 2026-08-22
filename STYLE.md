# Manuscript style guide

This guide is based on an editorial review of effective technical-book
techniques, including the supplied reference book. It records general
pedagogical qualities, not language to imitate.

## Reader experience

- Start each chapter with a concrete problem, system, or decision the reader
  can picture.
- Introduce one new abstraction at a time. Explain why it is needed before
  naming its parts.
- Move in this order: situation, intuition, mechanism, trade-off, real
  implementation, exercise.
- Use ordinary sentences and short paragraphs. Define jargon at first use.
- Prefer one sustained example over many disconnected examples.
- Add transitions that explain why the next section follows from the last.
- Use tables and lists only when comparison or enumeration is clearer than
  prose.
- Give every chapter at least two diagrams that explain distinct relationships:
  a data or control path and a decision, state, or feedback loop. Place them
  before the detailed prose they organize.
- Use tables for exact mappings, conditions, or trade-offs. Do not duplicate a
  diagram as a table unless the table adds a decision-relevant dimension.
- Keep implementation details subordinate to the engineering idea.
- End with a practical investigation, not a generic recap list.

## Voice

- Address the reader directly where it helps.
- Be confident without pretending that conditional advice is universal.
- Avoid slogans, stacked noun phrases, repetitive summaries, and excessive
  bold labels.
- Avoid opening several consecutive paragraphs with commands or definitions.
- Vary sentence length, but keep technical claims unambiguous.
- Use analogies sparingly and carry them only as far as they remain accurate.

## Visual system

- Use the single light reading theme. Signal blue (`#2458d3`) is the only
  chromatic accent; use its tints for emphasis, never a competing hue.
- Treat diagrams and tables as technical plates: the same border, radius,
  spacing, typography, and signal-blue hierarchy should connect them.
- Use fixed block proportions and balanced label wrapping. When a graph is too
  wide, change its layout direction so it fits the page without internal
  scrolling or unreadably small type.
- Reserve Literata for sustained reading, Inter for hierarchy and navigation,
  and JetBrains Mono for code, measurements, and structural labels.
- Use color to express emphasis and flow, not to assign undocumented meaning.
  A diagram that needs semantic categories must label them explicitly.
- Preserve comfortable prose width. Diagrams and tables may extend slightly
  beyond it when extra width materially improves comprehension.

## Originality

- Do not copy or closely paraphrase the reference book.
- Do not reproduce its analogies, examples, diagrams, or chapter sequence.
- Learn from high-level qualities such as pacing, clarity, and use of concrete
  examples while maintaining this book's own voice and systems-centered model.

## Chapter review questions

1. Can a reader explain why the chapter matters after the first page?
2. Does every abstraction solve a problem already introduced?
3. Is there a concrete example before the densest technical section?
4. Are citations close to claims without interrupting the explanation?
5. Does the chapter connect naturally to the chapters before and after it?
6. Can the final exercise be completed with information in the chapter?
7. Do the diagrams expose state, movement, or dependency rather than decorate
   the page?

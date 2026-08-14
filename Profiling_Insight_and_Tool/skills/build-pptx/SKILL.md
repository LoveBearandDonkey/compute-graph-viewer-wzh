---
name: build-pptx
description: Create, revise, and validate polished PowerPoint presentations (.pptx) from source material, screenshots, and structured narratives using reusable SVG slide primitives and PowerPoint assembly. Use when Codex needs to turn documents or product evidence into an executive deck, product introduction, proposal, review, training deck, comparison deck, or other presentation; create diagrams and screenshot-led slides; or regenerate an existing code-authored deck consistently.
---

# Build PPTX

Build presentation-ready decks through a deterministic `source -> SVG slides -> PPTX -> rendered PNG review` workflow. Keep the narrative and visuals specific to the user's request while keeping the build machinery reusable.

## Follow the workflow

1. Read repository instructions and all user-named sources. Preserve unrelated worktree changes.
2. Define the audience, decision, central message, and evidence boundary before designing pages.
3. Create a short slide outline. Give each slide one conclusion, not merely a topic.
4. Copy [assets/deck-template.js](assets/deck-template.js) into a task-local build directory and replace all sample content. Do not edit the bundled template in place for a deck.
5. Use [scripts/slide-kit.js](scripts/slide-kit.js) through the template's `buildDeck` interface. Prefer diagrams, comparisons, and real screenshots when they clarify a relationship; avoid decorative clutter.
6. Generate SVG slides with [scripts/render-deck.js](scripts/render-deck.js).
7. Assemble the SVG files into `.pptx` with [scripts/assemble-pptx.ps1](scripts/assemble-pptx.ps1). This step requires Windows with Microsoft PowerPoint installed.
8. Inspect every rendered PNG, not only the source code. Fix overflow, weak hierarchy, unreadable screenshots, visual imbalance, and factual ambiguity; regenerate until clean.
9. Deliver the `.pptx` and state any evidence or editability limitations.

Read [references/deck-design.md](references/deck-design.md) before designing a new deck or materially changing its story. It contains slide-selection guidance, evidence rules, and the visual QA checklist.

## Create a deck project

Keep generated artifacts outside the skill folder:

```text
<task-build>/
|-- deck.js
|-- assets/
|-- slide_svgs/       # generated
`-- render_check/     # generated
```

Render the source:

```powershell
node <skill>/scripts/render-deck.js --source <task-build>/deck.js --out <task-build>/slide_svgs
```

Assemble and render-check it:

```powershell
powershell -ExecutionPolicy Bypass -File <skill>/scripts/assemble-pptx.ps1 `
  -SvgDir <task-build>/slide_svgs `
  -OutputPath <delivery>/presentation.pptx `
  -RenderDir <task-build>/render_check
```

Use `-Force` only when replacing the exact intended output and render directory. The script refuses ambiguous or non-empty targets by default.

## Use the slide primitives

The deck source exports `buildDeck({ createDeck, theme })`. Add slides with `deck.addSlide()` and draw with:

- `slide.background(color)`
- `slide.rect(x, y, w, h, options)`
- `slide.ellipse(x, y, w, h, options)`
- `slide.line(x1, y1, x2, y2, options)`
- `slide.text(text, x, y, w, h, options)`
- `slide.image(path, x, y, w, h, options)`

Coordinates use a 1600 x 900 canvas. `render-deck.js` rejects empty slides and elements outside the canvas. Use `slide.text` with `wrap: true` for body copy; keep important text concise instead of relying on aggressive wrapping.

## Capture web evidence when needed

Use [scripts/capture-webpage.js](scripts/capture-webpage.js) for a deterministic browser screenshot when Microsoft Edge is available:

```powershell
node <skill>/scripts/capture-webpage.js --url http://127.0.0.1:8765/page.html `
  --output <task-build>/assets/page.png --wait 2500
```

Add repeated `--eval "JavaScript expression"` arguments only for verified interactions such as selecting a tab or opening a detail panel. Serve local pages over HTTP when they load relative data or scripts.

## Preserve evidence integrity

- Separate verified facts, sample/demo data, inference, and concept-only proposals.
- Never present a prototype interaction as implemented without inspecting its current HTML and JavaScript behavior.
- Cite or footnote non-obvious claims close to the slide where practical.
- Use real UI screenshots when explaining an existing tool. Crop to the decision-relevant area and accompany each screenshot with the conclusion it supports.
- Do not fabricate quantitative results, customer statements, product states, or UI operations.

## Treat editability honestly

This workflow places each SVG slide into PowerPoint as one full-slide graphic. It produces stable visual fidelity and repeatable generation, but the individual text boxes and shapes are not separately editable in PowerPoint. If element-level editing is required, use a native PowerPoint authoring library or manually rebuild the approved slides and explicitly state the tradeoff.

## Validate before delivery

Treat these as blockers:

- render command fails or reports an empty/out-of-bounds slide;
- PowerPoint assembly produces fewer slides than the SVG manifest;
- any rendered slide contains clipping, overlap, placeholder copy, mojibake, illegible text, or distorted imagery;
- the deck promises an unverified operation or presents sample data as production evidence;
- the final filename is a lock file (`~$...`) or a temporary build artifact.

Run the structural Skill validation after editing the Skill itself:

```powershell
uv run --no-project python <skill-creator>/scripts/quick_validate.py <skill>
```


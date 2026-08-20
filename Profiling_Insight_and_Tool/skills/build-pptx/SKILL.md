---
name: build-pptx
description: Create, revise, and validate polished presentations as either true PowerPoint files (.pptx) or PTO-standard browser-based HTML decks, using the skill's bundled portable PTO visual specification without an external design-system checkout. Online HTML decks natively support coherent light/dark theme switching; true PowerPoint work requires the user to choose light or dark when the theme is not already explicit. Use when Codex needs to turn Markdown, documents, screenshots, or structured narratives into an executive deck, product introduction, proposal, review, training deck, comparison deck, online PPT, or other presentation; create diagrams and screenshot-led slides; or regenerate an existing code-authored deck consistently. If the requested presentation format is ambiguous, ask whether the user wants a real PowerPoint file or an online HTML presentation before building.
---

# Build PPTX or PTO online PPT

Build presentation-ready decks through one of two workflows:

- **True PowerPoint:** `source -> SVG slides -> .pptx -> rendered PNG review`.
- **PTO online PPT:** `Markdown/source -> narrative outline -> PTO HTML slides -> browser/static review -> optional standalone HTML`.

Keep the narrative and visuals specific to the request while keeping the build machinery reusable.

## Decide the deliverable first

Resolve the output mode before editing or generating files:

- Choose **true PowerPoint** when the user explicitly asks for `.pptx`, Microsoft PowerPoint, a downloadable PowerPoint file, or delivery inside PowerPoint.
- Choose **PTO online PPT** when the user explicitly asks for HTML, a webpage deck, browser presentation, online PPT, URL sharing, web interaction, or a standalone HTML presentation.
- If the user says only “PPT”, “deck”, “slides”, “演示文稿”, or otherwise leaves the format unclear, stop and ask one blocking question:

  `你希望交付真正的 PowerPoint 文件（.pptx），还是浏览器打开和分享的在线 PPT（HTML）？`

Do not infer the output mode from the source file extension, the existence of Markdown, or the word “PPT” alone. If the user requests both, create two explicit deliverables from one approved outline; do not claim that HTML is a `.pptx` or that a full-slide SVG PowerPoint is element-editable.

Resolve theme behavior immediately after resolving the output mode:

- For a **true PowerPoint**, if the user has not explicitly selected light or dark and has not explicitly asked to preserve an existing deck's theme, stop and ask one blocking question before designing: `真正的 PowerPoint 需要浅色版还是深色版？` Do not infer a static PowerPoint theme from browser/OS preference, the source document, or a convenient template. If the user explicitly requests both, build two clearly named `.pptx` files from the same approved outline.
- For a **PTO online PPT**, do not block on a theme question. Build both complete light and dark states with an accessible runtime switch. Use the user's explicit choice as the authored default when supplied; resolve the actual initial state in the order documented in `references/design.md` while keeping both modes available.

## Follow the shared narrative workflow

1. Read repository instructions and all user-named sources. Preserve unrelated worktree changes.
2. Read [references/design.md](references/design.md) completely before designing either output mode. Treat it as the authoritative, self-contained PTO presentation visual contract. Do not require or import an external `pto-design-system` folder.
3. Define the audience, decision, central message, evidence boundary, presentation mode, and theme behavior. Complete the true-PowerPoint theme question before visual design when required.
4. Create a short outline. Give every slide one conclusion that the audience can absorb at once, not merely a topic heading.
5. Convert prose into visual relationships where useful: flow, branch, timeline, hierarchy, architecture, comparison, or annotated evidence. Do not decorate a text dump.
6. Preserve verified facts and distinguish examples, inference, concept proposals, and unverified claims.

## Build a true PowerPoint file

1. Confirm a single light or dark delivery theme before visual design. If it remains unclear, ask `真正的 PowerPoint 需要浅色版还是深色版？` and wait; do not silently choose. Then copy [assets/deck-template.js](assets/deck-template.js) into a task-local build directory and replace all sample content. Do not edit the bundled template in place for a deck.
2. Use [scripts/slide-kit.js](scripts/slide-kit.js) through the template's `buildDeck` interface. Prefer diagrams, comparisons, and real screenshots when they clarify a relationship; avoid decorative clutter.
3. Generate SVG slides with [scripts/render-deck.js](scripts/render-deck.js).
4. Assemble the SVG files into `.pptx` with [scripts/assemble-pptx.ps1](scripts/assemble-pptx.ps1). This step requires Windows with Microsoft PowerPoint installed.
5. Inspect every rendered PNG, not only the source code. Fix overflow, weak hierarchy, unreadable screenshots, visual imbalance, and factual ambiguity; regenerate until clean.
6. Deliver the `.pptx` and state any evidence or editability limitations.

Read [references/deck-design.md](references/deck-design.md) for slide selection, evidence rules, and narrative QA. Use [references/design.md](references/design.md) for all colors, typography, spacing, surfaces, diagrams, and component styling; translate its browser and 1600 x 900 PowerPoint values directly into the slide source.

## Build a PTO online HTML PPT from Markdown

Follow this branch only after the user has selected an online/browser presentation.

### 1. Read and structure the source

1. Read the entire Markdown and any linked sources before writing HTML.
2. Separate the document into audience-sized conclusions. Split a dense section across slides; combine short sections only when they support the same conclusion.
3. Put the requested opening journey, summary, or decision context first. Preserve the source's main-business focus and remove repetitive exposition.
4. Build an explicit slide map containing `section`, `title`, `takeaway`, `visual form`, and source coverage. Ensure every important Markdown section is represented exactly once unless intentional repetition is useful.
5. Keep explanatory examples concrete enough to create a mental picture. Use diagrams and annotated examples instead of introducing a second domain knowledge problem.

### 2. Apply the bundled portable PTO presentation design

Use only [references/design.md](references/design.md) as the presentation visual source. It contains the subset distilled from the proven workflow deck: coherent light/dark tokens, aura backgrounds, translucent panes, transparent chrome, 16:9 slide stage, typography, surfaces, cards, tables, process diagrams, theme behavior, and interaction rules.

- Implement the portable shell and semantic tokens in the generated HTML itself or in task-local files delivered with it. Do not link to an external design-system directory.
- Use the `pto-ide-frame`-derived class structure documented in `design.md` for the topbar, workarea, split, Explorer pane, preview pane, and status strip. This is a presentation-specific portable subset, not a dependency on the full IDE product shell.
- Omit `.pto-ide-frame__activity-rail` from online PPTs. The Explorer pane is the slide directory and sits directly in the workarea/split.
- Reuse the documented component recipes before adding page-local layout classes. Add layout-only classes when the story requires them; do not invent a second visual language.
- Implement both bundled light and dark themes. Switch the entire semantic token set through the root `data-theme` value; never combine a light business canvas with dark chrome or vice versa. Do not add floating playback merely because the deck has previous/next navigation.
- Keep prose at the documented body baseline; solve crowding by splitting slides, changing layout, wrapping, or scrolling—not by shrinking body text or scaling the page.

### 3. Build the browser presentation shell

- Keep every slide at a true `16:9` stage ratio. Show one active slide at a time inside the preview pane.
- Represent slides as stable elements such as `<article class="deck-slide" id="..." data-section="..." data-title="...">` so navigation can be generated from the document rather than duplicated manually.
- Give the Explorer pane a vertically scrollable chapter/slide list that uses the available viewport height, including on small screens. Give `.pto-ide-frame__pane.pto-ide-frame__explorer` an explicit `margin-left: 8px`; preserve that outer breathing room when resizing the split.
- Let the workarea use the width freed by the omitted activity rail. Do not replace the rail with a second icon strip, tool palette, or decorative navigation band.
- Support previous/next controls, Arrow/Page keys, Home/End, hash-addressable slide IDs, active-title/page status, and fullscreen when appropriate.
- Put an accessible light/dark theme control in the transparent topbar. Apply the initial theme before first paint, persist an explicit user choice locally when storage is available, update `aria-pressed`/accessible labeling, and dispatch the documented theme-change event so rendered charts or canvases can repaint.
- Support blank-area click paging without interfering with reading or copying. Divide the active slide at its visual midpoint: clicking an eligible blank area on the left goes to the previous slide; clicking an eligible blank area on the right goes to the next slide. Trigger only when the click target is the stage or `.deck-slide.is-active` itself. Do not page when the target is a title, paragraph, card, table, diagram, link, button, or any other slide content descendant, and do not page while `window.getSelection()` contains non-collapsed text.
- Keep the slide stage independent from viewport dimensions: resize the surrounding shell while preserving `16:9`; do not stretch slide content.
- Use inline SVG or an approved PTO visualization pattern for diagrams. Prefer orthogonal flow lines, place short annotations on their lines, and omit a legend when labels already explain the encoding.
- Use diagrams, architecture views, comparisons, and restrained callouts to clarify relationships. Avoid dense paragraphs, decorative gradients, excessive card grids, and five-color process lines.
- Keep external hyperlinks as references. Do not mistake ordinary `<a href>` links for rendering dependencies.

### 4. Keep development and sharing outputs distinct

Maintain a readable development HTML using the portable styles from `references/design.md`, with no external PTO design-system path. If the user asks for offline use, email sharing, or one self-contained file, preserve the development source and create a separate `*-standalone.html`.

When available, invoke `$bundle-single-html` or run its deterministic script:

```powershell
node .codex/skills/bundle-single-html/scripts/bundle-single-html.mjs `
  <deck.html> <deck-standalone.html>
```

The standalone build must inline any remaining task-local CSS, JavaScript, fonts, images, and media while preserving cascade and script order. Report remote runtime dependencies, dynamic requests, module imports, iframes, and server APIs that prevent true offline independence. Packaging must not alter slide content or the bundled PTO presentation visual contract.

### 5. Validate the online PPT

Treat these as blockers:

- blank content pane, broken initialization, or fewer rendered slides than the slide map;
- slide, navigation, and page-count mismatches;
- clipping, overlap, unreadably small text, broken `16:9` ratio, or viewport stretching;
- an Explorer pane that cannot scroll through all chapters, does not use available screen height, or lacks the required `8px` left margin;
- an online PPT that still contains `.pto-ide-frame__activity-rail` or an equivalent IDE icon rail;
- click paging that advances from content clicks or text selection, or does not map eligible blank-left/blank-right clicks to previous/next respectively;
- a missing/inoperable theme switch, a visible wrong-theme flash on initialization, a theme choice that is not persisted when storage is available, or any component/diagram that remains styled for the opposite theme;
- malformed SVG paths, arbitrary diagonal routing where orthogonal routing is intended, detached annotations, or redundant legends;
- external PTO design-system imports, missing local resources, unresolved CSS imports, local script references in a standalone build, mojibake, placeholder copy, or truncation markers;
- invented evidence or presentation claims that exceed the source.

Audit the output against the typography, token, component, and density tables in `references/design.md`. Treat prose below the documented baseline, ordinary UI below 12px, unexplained hard-coded colors, border-heavy containers, and page-level scaling as blockers.

Perform a browser visual review at representative viewport sizes in both light and dark modes unless the user explicitly asks to own or skip screenshot/visual verification. Even when visual review is delegated, still run dependency, tag-balance, slide-count, navigation, text, theme-switch, persistence, and initialization checks.

Deliver the development HTML and, when requested, the standalone HTML. State which portable PTO shell and component recipes were used, how the initial theme is resolved, whether both theme states were validated, whether the file has runtime dependencies, and which interactions were validated.

## Create a true-PowerPoint deck project

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

## Use the true-PowerPoint slide primitives

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

## Treat true-PowerPoint editability honestly

This workflow places each SVG slide into PowerPoint as one full-slide graphic. It produces stable visual fidelity and repeatable generation, but the individual text boxes and shapes are not separately editable in PowerPoint. If element-level editing is required, use a native PowerPoint authoring library or manually rebuild the approved slides and explicitly state the tradeoff.

## Validate a true PowerPoint before delivery

Treat these as blockers:

- a light/dark theme that was inferred for a new static PowerPoint instead of explicitly confirmed when the user's request was unclear;
- render command fails or reports an empty/out-of-bounds slide;
- PowerPoint assembly produces fewer slides than the SVG manifest;
- any rendered slide contains clipping, overlap, placeholder copy, mojibake, illegible text, or distorted imagery;
- the deck promises an unverified operation or presents sample data as production evidence;
- the final filename is a lock file (`~$...`) or a temporary build artifact.

Run the structural Skill validation after editing the Skill itself:

```powershell
uv run --no-project python <skill-creator>/scripts/quick_validate.py <skill>
```

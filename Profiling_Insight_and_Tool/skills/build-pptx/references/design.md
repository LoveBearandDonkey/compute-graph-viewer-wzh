# Portable PTO presentation design

## Contents

1. Authority and scope
2. Visual signature
3. Theme contract
4. Shared semantic tokens
5. Typography and geometry
6. Online HTML shell
7. Presentation components
8. Diagrams and data displays
9. True PowerPoint mapping
10. Validation

## Authority and scope

Use this file for every presentation produced by `build-pptx`, whether the output is a true `.pptx` or an online HTML deck. It is the self-contained presentation subset distilled from the proven workflow deck. Do not import, require, or assume access to an external PTO design-system folder.

Use one visual language across both output modes. The online deck always contains coherent light and dark states and a native switch; the true PowerPoint uses one user-confirmed light or dark state with the same type hierarchy, spacing, surfaces, cards, tables, and diagram rules without browser chrome.

## Visual signature

- Use a cool-neutral light or dark base with four restrained color auras at the canvas edges.
- Use transparent top chrome and soft translucent white panes. Let the background show through; do not add a solid header band.
- Use neutral white/gray surfaces for most content. Reserve saturated colors for small semantic signals, selected states, data branches, and compact callouts.
- Prefer spacing, alignment, fill contrast, and typography over borders. Avoid nested outlines, heavy card grids, decorative side rails, and multiple shadows.
- Use one dark inverse panel to establish focus when needed; do not turn every section into a colored card.
- Use rounded corners consistently: 6px compact, 8–10px standard, 12–14px large. Do not mix arbitrary radii.

## Theme contract

Treat light and dark as two complete semantic themes, not as a background-color toggle.

- Online HTML decks must include both modes and a visible, keyboard-operable theme control in the transparent topbar. Do not ask a blocking theme question for an online deck because the viewer can switch at runtime.
- Set `data-theme="light"` or `data-theme="dark"` on `<html>`. Every shell, slide, card, table, diagram label, SVG stroke/fill, focus state, and status element must consume semantic variables so the whole page switches together.
- Resolve the online initial theme before first paint in this order: valid `?theme=light|dark` URL value; saved `localStorage` value; an explicit authored default from the user's request; `prefers-color-scheme`; then light. Keep both modes available regardless of the initial state.
- Save only an explicit viewer action to `localStorage` under `pto-deck-theme`. Do not overwrite the saved preference merely because the deck opened with a URL or authored default.
- Set `color-scheme` to the active theme, keep the control's `aria-pressed` and accessible label current, and dispatch `pto-theme-change` with `{ detail: { theme } }` after switching. Canvas/WebGL/chart consumers must repaint or call their `resize`/update lifecycle on that event.
- For a true PowerPoint, use exactly one user-confirmed theme. If the request is unclear, ask `真正的 PowerPoint 需要浅色版还是深色版？` before designing. A request to preserve an existing deck's appearance counts as confirmation of its existing theme. If both modes are requested, make two separately named files.
- Never mix light content with dark chrome in one default state. A deliberate inverse-focus card is a component treatment, not a second page theme.

## Shared semantic tokens

Use these exact light values as the portable baseline. For true PowerPoint, use the listed opaque equivalents where alpha/color-mix is unavailable.

| Role | HTML value | PowerPoint equivalent |
|---|---|---|
| Font | `"Segoe UI", "Microsoft YaHei", Arial, sans-serif` | Segoe UI / Microsoft YaHei |
| Mono font | `"Cascadia Code", Consolas, monospace` | Cascadia Code / Consolas |
| Background start / mid / end | `#fdfdff` / `#f6f7fb` / `#eceff6` | same |
| Elevated white | `#ffffff` | `#ffffff` |
| Surface 2 | `#f2f2f2` | `#f2f2f2` |
| Surface 3 | `#e6e6e6` | `#e6e6e6` |
| Primary text | `rgba(0,0,0,.90)` | `#1a1a1a` |
| Secondary text | `rgba(0,0,0,.55)` | `#737373` |
| Muted text | `rgba(0,0,0,.42)` | `#949494` |
| Subtle / strong separator | `rgba(0,0,0,.07)` / `rgba(0,0,0,.18)` | `#ededed` / `#d1d1d1` |
| Primary | `#3577f6` | `#3577f6` |
| Accent | `#7c5ce0` | `#7c5ce0` |
| Success | `#16865c` | `#16865c` |
| Warning | `#c77a05` | `#c77a05` |
| Danger | `#d6455d` | `#d6455d` |
| Selected fill | `rgba(53,119,246,.12)` | `#e7efff` |
| Hover fill | `rgba(24,24,24,.06)` | `#f0f0f0` |
| Small / large shadow | `0 4px 12px rgba(0,0,0,.08)` / `0 18px 42px rgba(0,0,0,.15)` | soft 8% / 15% black shadow |

Use these dark overrides while retaining the same semantic roles:

| Role | Dark HTML value | Dark PowerPoint equivalent |
|---|---|---|
| Background start / mid / end | `#17181d` / `#101114` / `#0c0d10` | same |
| Surface 1 / 2 / 3 | `#161616` / `#1c1c1c` / `#262626` | same |
| Primary / secondary / muted text | `rgba(255,255,255,.90)` / `.60` / `.40` | `#e6e6e6` / `#999999` / `#666666` |
| Subtle / strong separator | `rgba(255,255,255,.06)` / `.16` | `#252525` / `#383838` |
| Primary / accent | `#4369ef` / `#7c8db8` | same |
| Success / warning / danger | `#04d793` / `#ffaa3b` / `#ff4b7b` | same |
| Selected / hover fill | `rgba(67,105,239,.14)` / `rgba(255,255,255,.06)` | `#1c2546` / `#252525` |
| Small / large shadow | `0 4px 12px rgba(0,0,0,.22)` / `0 18px 42px rgba(0,0,0,.38)` | soft 22% / 38% black shadow |

Use the 4px spacing scale: `4, 8, 12, 16, 20, 24, 32, 40`. Most component gaps are 8–16px; most card padding is 16–22px in HTML.

Use this background recipe in HTML and reproduce the same layered composition with full-slide shapes in PowerPoint:

```css
--pto-deck-background:
  radial-gradient(circle at top left, rgb(155 96 170 / .16), transparent 28%),
  radial-gradient(circle at top right, rgb(24 99 220 / .14), transparent 30%),
  radial-gradient(circle at 48% 6%, rgb(20 184 166 / .08), transparent 34%),
  radial-gradient(circle at 18% 86%, rgb(245 158 11 / .06), transparent 34%),
  linear-gradient(180deg, #fdfdff 0%, #f6f7fb 42%, #eceff6 100%);
```

## Typography and geometry

| Role | Online HTML | 1600 x 900 PowerPoint |
|---|---:|---:|
| Slide title | 30px / 1.18 / 700 | 42–46px / 1.15 / semibold-bold |
| Kicker | 12px / 1.4 / 700 | 17–18px / 1.3 / bold |
| Subtitle and prose | 14px / 1.55–1.65 | 21–23px / 1.35–1.5 |
| Card title | 16–18px / 1.35 | 24–28px / 1.25–1.35 |
| Compact table and node | 12px / at least 1.35 | 17–19px / at least 1.3 |
| Micro label and page number | 11px / short text only | 15–16px / short text only |

- Never use prose below 14px in HTML or 20px in a 1600 x 900 PowerPoint.
- Use 11px HTML only for page numbers, eyebrows, short badges, or diagram annotations. Never use it for explanatory paragraphs.
- Keep HTML slide padding near `34px 42px 28px`. Keep PowerPoint safe margins at 64–72px.
- Keep one conclusion per slide and preferably six or fewer major visual objects. Split content instead of shrinking it.

## Online HTML shell

Embed or generate this presentation subset locally. Class names are a stable contract; no external stylesheet is required.

```css
:root {
  color-scheme: light;
  --font-sans: "Segoe UI", "Microsoft YaHei", Arial, sans-serif;
  --font-mono: "Cascadia Code", Consolas, monospace;
  --background: #f5f5f5; --background-elevated: #fff;
  --surface-1: #fff; --surface-2: #f2f2f2; --surface-3: #e6e6e6;
  --foreground: rgba(0,0,0,.90); --foreground-secondary: rgba(0,0,0,.55); --foreground-muted: rgba(0,0,0,.42);
  --border-subtle: rgba(0,0,0,.07); --border-strong: rgba(0,0,0,.18);
  --primary: #3577f6; --accent: #7c5ce0; --success: #16865c; --warning: #c77a05; --danger: #d6455d;
  --state-hover: rgba(24,24,24,.06); --state-selected: rgba(53,119,246,.12);
  --radius-sm: 6px; --radius-md: 10px; --radius-lg: 14px;
  --shadow-sm: 0 4px 12px rgba(0,0,0,.08); --shadow-lg: 0 18px 42px rgba(0,0,0,.15);
  --pane-fill: rgba(255,255,255,.80); --pane-header-fill: rgba(255,255,255,.72);
  --stage-fill: rgb(255 255 255/.44); --slide-fill: rgb(255 255 255/.96);
  --type-body: 400 14px/1.55 var(--font-sans); --type-ui: 400 12px/1.4 var(--font-sans); --type-mono: 400 12px/1.4 var(--font-mono);
  --pto-deck-background: radial-gradient(circle at top left,rgb(155 96 170/.16),transparent 28%),radial-gradient(circle at top right,rgb(24 99 220/.14),transparent 30%),radial-gradient(circle at 48% 6%,rgb(20 184 166/.08),transparent 34%),radial-gradient(circle at 18% 86%,rgb(245 158 11/.06),transparent 34%),linear-gradient(180deg,#fdfdff 0%,#f6f7fb 42%,#eceff6 100%);
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --background: #101114; --background-elevated: #161616;
  --surface-1: #161616; --surface-2: #1c1c1c; --surface-3: #262626;
  --foreground: rgba(255,255,255,.90); --foreground-secondary: rgba(255,255,255,.60); --foreground-muted: rgba(255,255,255,.40);
  --border-subtle: rgba(255,255,255,.06); --border-strong: rgba(255,255,255,.16);
  --primary: #4369ef; --accent: #7c8db8; --success: #04d793; --warning: #ffaa3b; --danger: #ff4b7b;
  --state-hover: rgba(255,255,255,.06); --state-selected: rgba(67,105,239,.14);
  --shadow-sm: 0 4px 12px rgba(0,0,0,.22); --shadow-lg: 0 18px 42px rgba(0,0,0,.38);
  --pane-fill: rgba(22,22,22,.80); --pane-header-fill: rgba(22,22,22,.72);
  --stage-fill: rgb(0 0 0/.18); --slide-fill: rgb(22 22 22/.96);
  --pto-deck-background: radial-gradient(circle at top left,rgb(155 96 170/.12),transparent 28%),radial-gradient(circle at top right,rgb(67 105 239/.12),transparent 30%),radial-gradient(circle at 48% 6%,rgb(20 184 166/.06),transparent 34%),radial-gradient(circle at 18% 86%,rgb(245 158 11/.05),transparent 34%),linear-gradient(180deg,#17181d 0%,#101114 44%,#0c0d10 100%);
}
* { box-sizing: border-box; }
html, body { width: 100%; height: 100%; margin: 0; }
body { overflow: hidden; background: var(--pto-deck-background); color: var(--foreground); font-family: var(--font-sans); }
.pto-ide-frame { width: 100vw; height: 100dvh; display: flex; flex-direction: column; overflow: hidden; background: var(--pto-deck-background); }
.pto-ide-frame__topbar { min-height: 40px; display: flex; align-items: center; justify-content: space-between; padding: 0 14px; background: transparent; }
.pto-ide-frame__body, .pto-ide-frame__workarea { min-width: 0; min-height: 0; flex: 1 1 auto; display: flex; overflow: hidden; }
.pto-ide-frame__workarea { flex-direction: column; }
.pto-ide-frame__split { min-width: 0; min-height: 0; flex: 1 1 auto; display: grid; grid-template-columns: 280px minmax(640px,1fr); gap: 8px; padding-right: 8px; overflow: hidden; }
.pto-ide-frame__explorer { min-height: 0; margin-left: 8px; display: flex; flex-direction: column; overflow: hidden; }
.pto-ide-frame__pane { min-width: 0; min-height: 0; overflow: hidden; border-radius: 10px; background: var(--pane-fill); box-shadow: var(--shadow-sm); backdrop-filter: blur(18px) saturate(1.18); }
.pto-ide-frame__pane-header { min-height: 34px; display: flex; align-items: center; justify-content: space-between; gap: 8px; padding: 0 10px; background: var(--pane-header-fill); }
.pto-ide-frame__pane-title { margin: 0; font-size: 12px; font-weight: 600; }
.pto-ide-frame__pane-meta { color: var(--foreground-muted); font: var(--type-mono); font-size: 11px; }
.pto-ide-frame__pane-body { min-width: 0; min-height: 0; flex: 1 1 auto; overflow: auto; }
.deck-nav { display: flex; flex-direction: column; overflow-y: scroll; scrollbar-gutter: stable; gap: 2px; padding: 6px; }
.deck-nav__item { min-height: 34px; display: grid; grid-template-columns: 26px minmax(0,1fr); align-items: center; gap: 7px; padding: 6px 8px; border: 0; border-radius: 6px; background: transparent; color: var(--foreground-secondary); font: var(--type-ui); text-align: left; }
.deck-nav__item:hover { background: var(--state-hover); color: var(--foreground); }
.deck-nav__item.is-active { background: var(--state-selected); color: var(--foreground); font-weight: 650; }
.deck-stage { display: grid; place-items: center; padding: 12px; overflow: auto; background: var(--stage-fill); }
.deck-slide { position: relative; width: min(100%,calc((100dvh - 126px)*16/9)); min-width: 720px; aspect-ratio: 16/9; display: none; flex-direction: column; overflow: hidden; padding: 34px 42px 28px; border: 1px solid var(--border-subtle); border-radius: 10px; background: var(--slide-fill); box-shadow: var(--shadow-lg); }
.deck-slide.is-active { display: flex; }
.slide-kicker { margin: 0 0 7px; color: var(--primary); font-size: 12px; line-height: 1.4; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; }
.slide-title { margin: 0; color: var(--foreground); font-size: 30px; line-height: 1.18; font-weight: 720; letter-spacing: -.025em; }
.slide-subtitle { max-width: 850px; margin: 8px 0 0; color: var(--foreground-secondary); font: var(--type-body); line-height: 1.65; }
.slide-body { min-height: 0; flex: 1 1 auto; margin-top: 22px; display: flex; flex-direction: column; justify-content: center; }
.slide-footer { margin-top: auto; padding-top: 10px; color: var(--foreground-muted); font-size: 11px; line-height: 1.45; }
.pto-ide-frame__status-strip { min-height: 24px; display: flex; align-items: center; justify-content: space-between; padding: 0 10px; color: var(--foreground-muted); font-size: 11px; background: transparent; }
.deck-theme-toggle { min-height: 30px; display: inline-flex; align-items: center; gap: 7px; padding: 5px 9px; border: 0; border-radius: 6px; background: transparent; color: var(--foreground-secondary); font: var(--type-ui); cursor: pointer; }
.deck-theme-toggle:hover { background: var(--state-hover); color: var(--foreground); }
.deck-theme-toggle:focus-visible { outline: 2px solid var(--primary); outline-offset: 2px; }
```

Place this no-flash initializer in `<head>` before the deck stylesheet. Set `data-default-theme` to the user's explicit online default when one exists; otherwise omit it.

```html
<html lang="zh-CN" data-theme="light">
<script>
(() => {
  const root = document.documentElement;
  const query = new URLSearchParams(location.search).get('theme');
  let saved = null;
  try { saved = localStorage.getItem('pto-deck-theme'); } catch (_) {}
  const authored = root.dataset.defaultTheme;
  const system = matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  const theme = [query, saved, authored, system, 'light'].find((value) => value === 'light' || value === 'dark');
  root.dataset.theme = theme;
  root.style.colorScheme = theme;
})();
</script>
```

Use one `[data-theme-toggle]` button in the transparent topbar:

```html
<button class="deck-theme-toggle" type="button" data-theme-toggle aria-pressed="false" aria-label="切换至深色主题">
  主题：<span data-theme-label>浅色</span>
</button>
```

Initialize it after the DOM exists:

```js
const root = document.documentElement;
const themeToggle = document.querySelector('[data-theme-toggle]');

function applyTheme(theme, persist = false) {
  if (theme !== 'light' && theme !== 'dark') return;
  root.dataset.theme = theme;
  root.style.colorScheme = theme;
  const isDark = theme === 'dark';
  themeToggle?.setAttribute('aria-pressed', String(isDark));
  themeToggle?.setAttribute('aria-label', isDark ? '切换至浅色主题' : '切换至深色主题');
  const label = themeToggle?.querySelector('[data-theme-label]');
  if (label) label.textContent = isDark ? '深色' : '浅色';
  if (persist) try { localStorage.setItem('pto-deck-theme', theme); } catch (_) {}
  window.dispatchEvent(new CustomEvent('pto-theme-change', { detail: { theme } }));
}

applyTheme(root.dataset.theme);
themeToggle?.addEventListener('click', () => applyTheme(root.dataset.theme === 'dark' ? 'light' : 'dark', true));
```

Do not add an activity rail to an online PPT. Keep the Explorer pane at 280px by default, vertically scrollable, with the required 8px left margin. The slide pane takes the remaining width. On small screens, preserve the slide's 16:9 canvas and allow the stage/Explorer to scroll; do not squeeze text or distort the slide.

## Presentation components

- **Neutral card:** surface 2, 12–14px radius, 16–22px padding, no full border and no shadow by default. Use equal heights and aligned title baselines in repeated grids.
- **Selected callout:** selected blue fill with primary text or a single 1px primary inset line. Use for one focal item, not every card.
- **Inverse focus:** primary text color as the background and white as text. Use for a key conclusion, merge result, or current production state.
- **Number disc:** 34px circle in HTML; dark neutral by default. Use one semantic color only when the number encodes a data branch or gate.
- **Quote box:** selected blue fill, 18–20px padding, standard radius, 18–20px HTML emphasis text. Keep it to one or two sentences.
- **Compact table:** surface 2 body, surface 3 header, 9–11px cell padding, subtle horizontal separators only, 12px HTML text. Avoid vertical grid lines unless exact column tracking requires them.
- **Column layouts:** 14–16px gaps; two columns for comparison, three for distinct alternatives, four only for short summaries. Keep major objects to six or fewer.
- **Icons:** use simple Lucide-like line SVGs with `fill:none`, `stroke:currentColor`, and consistent 1.7–1.8px browser stroke. Do not mix icon families.

## Diagrams and data displays

- Use orthogonal horizontal/vertical routes for process and architecture flows. Avoid arbitrary diagonal or curved connectors.
- Use a neutral strong separator color for normal routes. Use dashed neutral lines for feedback. Put concise annotations directly on routes; omit a separate legend when labels already explain them.
- Use semantic colors sparingly: primary for the main path or inference, accent for the second data branch, warning for gates, success for release, and danger for stop/failure. Do not color every connector differently.
- Use 110 x 54px nodes as the HTML baseline, 8–10px inner padding, 6px radius, 12px title, and 11px metadata. Scale proportionally for PowerPoint while keeping label sizes presentation-readable.
- A 3px inset side signal is allowed only when it encodes a real data branch or state; do not use decorative left rails on generic cards.
- Use low-contrast dotted canvas backgrounds only behind real graphs. Keep dots near 1px with about 18–20px spacing.
- Keep flow labels on opaque or nearly opaque neutral surfaces so lines do not reduce readability.
- For journeys, use evenly sized neutral steps and one selected step. For architecture, use stacked neutral layers with one inverse core layer. For decisions, use one gate and two clearly separated outcomes.

## True PowerPoint mapping

- Use a 1600 x 900 canvas and reproduce the user-confirmed light or dark aura background across the full slide.
- Do not draw the online topbar, Explorer, pane headers, or status strip in a true PowerPoint unless the slide is explicitly demonstrating the online product shell.
- Use 64–72px safe margins, 24–32px gaps between major regions, 26–36px card padding, 14–20px radii, and soft restrained shadows.
- Translate HTML semantic tokens to the PowerPoint equivalents in the token table. Keep primary/accent/state meaning identical across modes.
- Use the PowerPoint type sizes in the typography table; do not copy browser pixel sizes directly onto a 1600 x 900 slide.
- Preserve the same component hierarchy: neutral cards, one selected callout or inverse focus area, compact tables, orthogonal diagrams, and limited semantic accents.

## Validation

- Confirm that no output imports or references an external PTO design-system directory.
- Confirm that online output contains both complete token states and that true PowerPoint uses the explicitly confirmed state.
- Confirm that prose and ordinary controls meet the minimum type sizes; list any smaller data-visualization exception and its reason.
- Confirm that repeated cards align, slide margins are consistent, colors retain one meaning, and borders are not the primary visual language.
- Confirm that online decks omit the activity rail, preserve the Explorer's 8px left margin and full-height scrolling, and keep the slide at 16:9.
- Confirm that the online switch works by keyboard and pointer, updates accessible state, persists an explicit viewer choice when storage is available, avoids a first-paint theme flash, and leaves no opposite-theme components behind.
- Confirm blank-area click paging goes previous on the left and next on the right, ignores content descendants, and does nothing while text is selected.
- Confirm every diagram uses orderly routes, readable labels, and no redundant legend.

# Deck design and validation

## Contents

1. Narrative contract
2. Slide selection
3. Visual system
4. Evidence handling
5. Quality assurance

## Narrative contract

Before drawing, write four short statements:

- **Audience:** who must understand or decide.
- **Decision:** what should be approved, changed, or remembered.
- **Central message:** one sentence the audience should repeat afterward.
- **Evidence boundary:** what is verified, inferred, sample data, or concept only.

Build the slide sequence as an argument. A common product-tool sequence is:

`stakes -> user problem -> business model -> pain -> opportunity -> product journey -> workflow -> proof -> value -> next step`

Use only the beats the story needs. A short deck is stronger than a complete inventory.

## Slide selection

Choose the smallest visual that makes the relationship clear:

| Need | Preferred slide |
|---|---|
| One memorable assertion | Statement slide with one supporting visual |
| 2–4 alternatives | Comparison columns or matrix |
| Dependent steps | Left-to-right process or vertical journey |
| Hierarchy or ownership | Tree or layered architecture |
| Before/after effect | Matched panels with identical scale |
| Product operation | Screenshot plus numbered callouts |
| Several exact mappings | Compact table |
| Final recall | Three or four takeaways, not a feature inventory |

Avoid a diagram for a single step. Avoid screenshots that merely prove a page exists.

## Visual system

- Use 16:9 and a consistent safe margin of at least 55 px.
- Use one type family unless a brand standard requires otherwise.
- Keep titles around 30–44 px and body copy around 18–26 px on a 1600 x 900 canvas.
- Use one primary accent and at most two semantic accents per slide.
- Keep repeated cards identical in width, padding, title baseline, and visual weight.
- Prefer 6 or fewer major objects per slide.
- Use whitespace to group related items before adding borders or background panels.
- Keep screenshot labels readable at presentation distance; crop or split the view if they are not.
- Do not stretch raster images. Use `fit: "contain"` when the whole image matters and `fit: "cover"` only when cropping is acceptable.

## Evidence handling

Maintain a small working ledger when the deck describes a real product or business result:

| Claim | Source | Status | Slide | Notes |
|---|---|---|---|---|
| Current UI operation | HTML + handler | verified | 7 | exact visible label |
| Performance result | report | verified | 4 | include unit and baseline |
| Prototype dataset | fixture | sample data | 8 | label on slide |
| Proposed capability | design note | concept only | 5 | do not demo as live |

For quantitative comparisons, include unit, aggregation basis, and baseline. Do not equate correlation with causation. When uncertainty matters, state the test or confidence boundary rather than using an arbitrary noise band.

## Quality assurance

Inspect the generated PNGs at full size and as a contact sheet.

### Content

- Every title states a conclusion or useful question.
- Each slide advances exactly one inference.
- Terms and scenario names remain consistent across slides.
- Numbers retain units, baseline, direction, and evidence status.
- No placeholders, internal notes, or unsupported promises remain.

### Visual

- Nothing crosses the safe margin or overlaps.
- Text is readable without zooming.
- Repeated components align precisely.
- Screenshots are sharp and preserve aspect ratio.
- Contrast is sufficient and colors have consistent meaning.
- The deck still makes sense in grayscale where color is not the only encoding.

### Package

- SVG count, manifest count, PPTX slide count, and rendered PNG count agree.
- The PPTX opens without a repair warning.
- The filename is user-facing and contains no temporary prefix.
- Only the final deck and useful source project are delivered; lock files and transient render directories are excluded.


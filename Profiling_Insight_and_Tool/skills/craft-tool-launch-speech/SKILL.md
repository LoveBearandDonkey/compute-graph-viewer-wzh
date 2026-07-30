---
name: craft-tool-launch-speech
description: Create or update scenario-led product launch speeches for technical tools by grounding every claim in user verbatims, diagnosis cases, and real UI interactions. Use when Codex needs to turn CheckList-style research, scattered pain-point inputs, Markdown case chains, HTML/JavaScript prototypes, or task-comparison pages into a professional Chinese keynote, demo script, experience slogan set, or unified user-pain-to-opportunity-to-design matrix.
---

# Craft Tool Launch Speech

Turn scattered user research, case evidence, and tool prototypes into one stage-ready story. Make the tools solve a user problem together; do not present them as a page tour.

## Choose the mode

- **Create**: Build a speech from current sources and the output template.
- **Update**: Diff changed sources against the accepted speech. Preserve approved rhetoric and edit only affected claims, demo steps, opportunity mappings, and design points.
- **Audit**: Verify an existing speech against current UI operations, source facts, original voices, and required structure.
- **Rehearse**: Convert an accepted speech into a shorter cue sheet while preserving demo actions and fact-safe wording.

## Build the evidence ledger first

Read repository instructions before task files. Locate the user-voice source, canonical case, tool pages, and accepted speech. For this repository, use [references/source-map.md](references/source-map.md).

Create a working ledger before drafting:

| Beat | Claim or number | Source | Tool | Real operation | Voice row | Status |
|---|---|---|---|---|---|---|
| Discovery | What happened | Case | Monitoring | Click marker | Row N | verified |
| Root cause | Why it happened | Case + JS | Relation | Select event | Row N | verified |
| Validation | How repair is proved | Case + compare UI | Compare | Set baseline | Row N | demo data needed |

Use three statuses:

- `verified`: directly supported by a source and a current interaction.
- `demo data needed`: the operation exists, but the case-specific records must be prepared.
- `concept only`: not implemented. Exclude from a live-demo promise or label it explicitly as a roadmap.

Keep the ledger in working notes while drafting. Every case-specific claim that depends on prepared records must remain labeled `demo data needed` in the affected Demo step.

## Follow the workflow

1. **Extract user tension.** Group original voices and scattered pain-point inputs by the user’s business journey: prepare/baseline, monitor/stop loss, understand/triage, trace/root cause, decide/repair, and compare/validate. Retain Row IDs. Assign non-CheckList inputs stable IDs such as `补充原声-YYYYMMDD-01`, with their source and exact wording.
2. **Select one story.** Prefer a case with a visible symptom, a misleading surface signal, a causal pivot, an actionable repair, and a measurable verification result.
3. **Assign product roles.** Give each tool one distinct role in the journey. A useful default is `monitor → explain relationships → compare and prove`.
4. **Verify every operation.** Inspect visible HTML labels and the JavaScript that binds clicks, tabs, deep links, event data, and automatic view changes. Do not infer an operation from a filename or mockup comment.
5. **Design the reveal.** Start with stakes, reveal the symptom, expose an earlier warning, translate the error, follow the misleading branch, pivot to root cause, show propagation, repair by priority, and validate.
6. **Write each beat as a stage unit.** Include `【演讲】`, `【Demo｜工具：…】`, `【覆盖用户原声】`, and `【体验亮点 Slogan】`. Put every verbatim user-voice citation in Markdown blockquote form, for example `> **Row 34**：“……”`.
7. **Build the product matrix as Chapter 1.** Use one table, not separate mapping tables. Each row follows `business stage → pain summary + numbered verbatims → canonical product opportunity → Top highlight → design points → tool/Demo carrier`.
8. **Audit semantics manually.** Compare the speech with the live page for exact labels, default selected states, number units/bases, and preloaded data. The structural script cannot prove these.
9. **Validate structure.** Run `scripts/audit_speech.py` with the complete row IDs derived from the selected case’s “覆盖原声” sections plus any rows explicitly requested by the user. Fix errors and inspect warnings.

Use [assets/speech-template.md](assets/speech-template.md) as the output scaffold. Do not ship placeholder text.

Name every delivered Markdown speech `业务主题-体验改进总结-YYYYMMDD.md`, using the delivery/update date in local time. Make the H1 exactly match the filename stem. For example: `训练监控与异常定位-体验改进总结-20260728.md`.

## Reconcile conflicting facts

Apply this source priority by claim type:

1. **User need**: exact CheckList voice.
2. **Diagnosis and result**: canonical case document.
3. **Live operation and visible label**: current HTML plus loaded JavaScript.
4. **Accepted phrasing**: latest user-approved speech.

When sources disagree:

- Use the visible UI label for what the presenter clicks.
- Use the canonical case for the technical cause and verification result.
- Use the least-specific true wording when scale or threshold differs, such as “single-card normal, multi-card reproducible.”
- Put case-specific preparation next to the affected Demo step; do not add a separate preparation chapter or claim that generic sample data already proves the case.
- Distinguish root-cause repair, auxiliary mitigation, and fallback. Never sell a timeout extension as a root-cause fix.
- Avoid dimensionally invalid cost claims. State time and card-hour assumptions explicitly.
- State the unit and basis of every diagnostic count. If logs show token counts while a chart shows traffic volume, present both as different measurements rather than treating them as conflicting values.
- Inspect the initial DOM/JavaScript state before writing “打开” or “开启.” Say “确认已开启” when the control is already selected by default.

## Write like a product manager on stage

- Lead with user stakes and a question, not architecture.
- Give the audience one new inference per beat.
- Explain technical terms through observable consequences.
- Use numbers only when they advance the diagnosis.
- Let the Slogan name the experience improvement, not the feature.
- Keep Demo instructions imperative and exact: open, click, switch, point, compare, return.
- Make tool transitions causal: “We know X; now the next tool must answer Y.”
- Mark optional or preloaded demo data explicitly.
- Prefer four memorable highlights over a long capability inventory.

## Update an accepted speech safely

When updating:

1. Diff the user-voice rows, case facts, UI labels/handlers, and demo datasets.
2. List affected beats before editing.
3. Preserve the title, main narrative, and approved Slogans unless the changed evidence invalidates them.
4. Update the body, Demo instructions, and the unified product matrix together.
5. Re-run the audit with the same required rows, then add newly requested rows.
6. Report material wording or demo-preparation changes to the user.

## Required output

Produce:

- filename and H1 in the exact `业务主题-体验改进总结-YYYYMMDD` format;
- duration and one main Slogan;
- Chapter 1 as one unified user-voice coverage matrix before tool roles and the speech;
- one `用户痛点主题` cell per business-stage row containing the summary first, followed directly by every source ID and verbatim; do not create a separate `原声输入` column;
- one canonical name for each product opportunity, reused exactly wherever that opportunity appears;
- the same table’s complete mapping from business stage and user pain to product opportunity, Top highlight, design points, and tool/Demo carrier;
- tool-role summary;
- timed or ordered stage narrative;
- exact Demo operation at every tool step;
- original-voice row/excerpt coverage at every beat;
- Markdown blockquotes for every original-voice excerpt cited in the speech body;
- experience Slogan at every major beat;
- Top 3 or Top 4 experience summary.

Do not add a separate README or process diary inside the skill.
Do not add a standalone `现场 Demo 准备与口径检查` chapter.

## Validate

From the skill directory, run:

```powershell
python scripts/audit_speech.py <speech.md> --required-rows 5,12,15 --top-count 4
```

Replace the sample row list with the complete selected-case coverage set. If `python` is unavailable but `uv` exists, run:

```powershell
uv run --no-project python scripts/audit_speech.py <speech.md> --required-rows 5,12,15 --top-count 4
```

If no Python 3 runner exists, manually verify:

1. every Demo block has concrete operation verbs and a following voice-coverage block;
2. every required Row appears in the body and Chapter 1 matrix;
3. Top headings match the requested count;
4. no `{{placeholder}}` remains;
5. every body citation under `【覆盖用户原声】` uses Markdown `>` blockquote syntax;
6. the matrix has no separate `原声输入` column and maps every row through opportunity, Top highlight, design point, and tool carrier;
7. every `demo data needed` claim has an inline preparation instruction.
8. filename and H1 use the same business theme and eight-digit date.

Treat audit errors as blockers. Review warnings and the manual semantic audit because narrative quality, current UI state, and factual causality cannot be fully automated.

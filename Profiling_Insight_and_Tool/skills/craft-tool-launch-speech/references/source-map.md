# Repository source map

Use this reference only for work under `Profiling_Insight_and_Tool`.

## Source priority

| Purpose | Default source | What to extract |
|---|---|---|
| User verbatims and desired design points | `Profiling_Insight_and_Tool/CheckList.md` | Row ID, exact voice, pain, existing Slogan |
| Canonical diagnosis story | `Profiling_Insight_and_Tool/training-run-twin-standalone/定位链-openPangu-2.0-Flash.md` | background, evidence, criterion, cause, repair, verification |
| Monitoring interaction | `Profiling_Insight_and_Tool/training-run-twin-standalone/training-monitoring-v2.html` and loaded files under `js/` | visible labels, tabs, buttons, automatic spotlight steps, numbers |
| Professional performance-analysis interaction | `Profiling_Insight_and_Tool/AI_Profiling_Tool/index_v3.html`, `memory-analysis.js`, `hbm-memory-snapshot/`, and case data under `data/` | overview/memory/timeline/code tabs, memory summary, composition, allocation lifetime, stack/source drill-down, deep-link behavior |
| Profile-comparison interaction | `Profiling_Insight_and_Tool/AI_Profiling_Tool/profileCompare.html` and `swimlane-data.js` | baseline/compare selection, grouped/aggregated views, task/lane difference dashboard, detail drill-down, actual available records |
| Relation-query interaction | `Profiling_Insight_and_Tool/training-run-twin-standalone/config-relation-observer.html` and `js/config-relation-observer.js` | event names, selectable objects, propagation scope, cross-view highlighting |
| Task-comparison interaction | `Profiling_Insight_and_Tool/training-run-twin-standalone/TaskCompare.html` | baseline selection, chart/parameter tabs, difference filters, actual demo records |
| Training conversational-configuration prototype | `Profiling_Insight_and_Tool/training-run-twin-standalone/training-monitoring-v2.html` and `js/training-chat-panel.js` | intelligent-dialog toggle/input, scripted reminder/chart-adjustment shortcuts, real free-text chat path, configuration-preview wording, implementation boundaries |
| Accepted launch speech | Latest `Profiling_Insight_and_Tool/*-体验改进总结-YYYYMMDD.md`; current example: `Profiling_Insight_and_Tool/训练监控与异常定位-体验改进总结-20260728.md` | approved narrative, Slogans, current coverage and wording |

Resolve paths from the repository rather than assuming the current working directory.

## Efficient inspection

1. Read the relevant case section completely, including its “覆盖原声” and repair/verification sections.
2. Search the HTML for visible labels and loaded scripts.
3. Search loaded JavaScript for the case key, event registry, click handler, tab selector, deep-link behavior, and displayed numbers.
4. Read only the relevant comparison-task data and controls.
5. Build the evidence ledger before writing.

Useful search terms include:

```text
问题一|问题二|caseKey|diagnosis|spotlight|定位链|熔断|AMP
Router|softmax|FP8|all-to-all|send|recv|dead expert
运行事件|传播源|victim|baseline|基线|图表对比|参数对比|只看差异
```

## Current case identifiers

For the Router overflow story:

- Visible case label: `问题二`
- Internal monitoring key: `moe-a2a`
- Layer: `38`
- Expert: `193`
- Runtime EP rank: `23`
- Global rank in relation view: `1559`

Treat these as current-source hints, not timeless constants. Re-check them on every update.

For the memory OOM story:

- Visible monitoring case label: `问题一`
- Internal monitoring key / professional-tool issue: `mem-oom`
- Professional-tool deep link: `index_v3.html?issue=mem-oom&tab=memory`
- Incident rank / step: `rank 17` / `step 12003`
- Snapshot context: `rank 17` / `step 12000`
- Hotspot: `PP stage 3` / `layer 38` / `expert_dispatch`

For the MFU decomposition story:

- Canonical story: performance case 2
- Case values: operator achievement ≈ `68%`, compute occupancy ≈ `64.3%`, derived MFU ≈ `43.7%`, measured MFU ≈ `45%`
- Current formula-drawer sample: `64.0% × 66.6% = 42.6%`
- Drawer entry: hover the MFU `?`, then click `查看计算明细`
- Drawer filters: click `64.0%`, `66.6%`, or `42.6%`; switch `表格视图 / 泳道视图`

## Common traps

- Source comments may retain an older problem number while the visible UI uses a newer one.
- The canonical case and spotlight copy may use different multi-card reproduction sizes. Say “multi-card” unless the exact number matters and is aligned.
- Thresholds may differ between narrative drafts and live configuration. Present the live configuration and note that thresholds are baseline-adjustable.
- Generic TaskCompare records may not contain the case-specific before/after runs. Mark them as demo preparation rather than implying they already exist.
- Generic `profileCompare.html` records may not contain a case-specific optimization pair. Its baseline selection and task/lane comparison operations can be verified independently, but case result numbers must remain marked `demo data needed` until matching records are preloaded.
- `profileCompare.html` may show estimated PHS values derived from demo data. Do not present those estimates as measured case-validation results.
- `TaskCompare.html` currently does not load `training-chat-panel.js` or contain the `打开智能对话` control. Treat TaskCompare conversational configuration as `concept only` until it is connected.
- The shared training-chat prototype's `消息设置` and `调整图表` suggestion chips run fixed scripts. Describe them as stage shortcuts only; the intended product interaction is user-entered free text followed by an explicit configuration preview and confirmation.
- The shared notification shortcut does not prove backend notification-rule persistence, and the chart shortcut modifies the monitoring accuracy panel rather than TaskCompare. Do not promise either as a current TaskCompare operation.
- The MFU drawer's hard-coded sample and the canonical case use different Profile data. Explain the method with either set, but never combine their values as one run.
- The MFU card's selectable peak-FLOPS assumption and the drawer's sample peak-FLOPS text may differ. Before a production Demo, synchronize those denominators or state the sample-data boundary explicitly.
- `index_v3.html` can render preloaded/imported report summaries, issue lists, evidence, and report Markdown. Do not claim arbitrary raw Profile data is automatically diagnosed without omissions unless the corresponding analysis pipeline and case report have been verified.
- EP rank and global rank are different identifiers; introduce their mapping once.
- A communication timeout is often the terminal symptom, not the original cause.

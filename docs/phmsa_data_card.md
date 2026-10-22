# Data card: PHMSA hazardous liquid accident reports (as used in AssetMind)

## Source

- **Dataset:** Hazardous Liquid Accident Data, January 2010 to present, from the US Pipeline and Hazardous Materials Safety Administration (PHMSA): [phmsa.dot.gov](https://www.phmsa.dot.gov/data-and-statistics/pipeline/distribution-transmission-gathering-lng-and-liquid-accident-and-incident-data).
- **What it is:** reports that US pipeline operators must file after accidents involving crude oil, refined products and similar liquids. Each report includes the operator's free-text narrative and the official cause.
- **Files:** `accident_hazardous_liquid_jan2010_present.txt` (tab-separated, 6,052 reports × 648 columns) and the blank reporting form PHMSA F7000-1 (PDF), which documents the fields and cause categories.
- **Downloaded:** October 2026 (incidents through September 2026).

## Fields used

| Column | Use |
|---|---|
| `NARRATIVE` | Input text |
| `CAUSE` | Label (official cause category) |
| `CAUSE_DETAILS` | Sub-cause; `UNKNOWN` defines the pending-detection task |
| `IYEAR` | Time-based split |

`ROOT_CAUSE_CATEGORY` / `ROOT_CAUSE_TYPE` are filled in only ~3% of reports and are not used.

## Composition

- 6,021 usable reports (text, label and year present); 99.5% of reports have a narrative.
- Narrative length: median 125 words (75% under ~200); narratives are capped at 4,000 characters in the export, which affects only ~5 reports.
- Text is upper-case engineering prose with abbreviations.

| Cause | Reports |
|---|---|
| Equipment failure | 2,759 |
| Corrosion failure | 1,328 |
| Incorrect operation | 834 |
| Material failure of pipe or weld | 406 |
| Natural force damage | 264 |
| Excavation damage | 199 |
| Other accident cause | 134 |
| Other outside force damage | 128 |

## Splits (by incident year)

| Split | Years | Main track (7 causes) | Pending (UNKNOWN) cases |
|---|---|---|---|
| Train | 2010–2021 | 4,557 | 17 |
| Validation | 2022–2023 | 590 | 0 |
| Test | 2024–2026 | 742 | 33 |

The time-based split mirrors deployment: models are always tested on incidents newer than anything they learned from.

## Tasks

1. **Main: cause classification.** 7 physical causes; "Other accident cause" excluded (in the test years it is almost entirely `UNKNOWN`). Metric: macro-F1.
2. **Side: pending detection.** Does the narrative indicate that the cause is still undetermined (`CAUSE_DETAILS == UNKNOWN`)? Metrics: precision, recall, F1.

## Known quirks and risks

| Issue | Evidence | How AssetMind handles it |
|---|---|---|
| **Reporting lag** | Initial reports are filed within 30 days, often before the investigation ends; supplemental reports later update the cause. Pending language appears in 3.8% of train, 7.4% of validation and 16.5% of test narratives | Undeterminable cases removed from the main task; pending detection treated as its own task |
| **Supplemental narratives** | 327 narratives keep "a supplemental report will be submitted…" but carry a confirmed cause; sampled narratives append the conclusion at the end ("SUPPLEMENTAL & FINAL: the cause was…") | Kept as valid examples; pending ground truth taken from the label, not keywords; text is never truncated from the end |
| **Keyword giveaways** | "corro…" appears in 66% of corrosion narratives vs 3% of others | Shortcut test with giveaway words masked |
| **Ambiguous keywords** | "excavat…" appears in 18% of non-excavation narratives (crews excavate to find and repair leaks) | Reported in error analysis; fine-tuning learns the distinction |
| **Class imbalance** | Equipment failure is 46% of reports | Macro-F1; class-balanced logistic regression |
| **Small classes** | 19–28 test documents for excavation, other outside force, natural force | Report counts; paired tests (McNemar, bootstrap) for model comparisons |
| **Brittle keyword rules** | Pending phrasing varies ("an investigation into the cause of the release is ongoing") | Language-model reading instead of fixed patterns |

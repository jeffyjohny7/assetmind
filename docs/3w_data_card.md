# Data card: Petrobras 3W (as used in AssetMind)

## Source

- **Dataset:** 3W, version 2.0.0 (from `dataset.ini`), maintained by Petrobras at [github.com/petrobras/3W](https://github.com/petrobras/3W).
- **License:** data files CC BY 4.0; code Apache 2.0.
- **Domain:** offshore oil wells; multivariate sensor time series with expert labels for undesirable events.

## Composition

- ~2,228 instances (one Parquet file each): **1,119 real**, ~1,089 simulated, 20 hand-drawn.
- Real instances come from **40 wells** (~28 files per well).
- Folders `0`–`9` hold one event type each; file names encode the source and start time (e.g. `WELL-00001_20170201010207`).

| Label | Event | Transient phase | Real files | Wells |
|---|---|---|---|---|
| 0 | Normal operation | – | 594 | 9 |
| 1 | Abrupt increase of BSW | Yes | 4 | 3 |
| 2 | Spurious closure of DHSV | Yes | 22 | 7 |
| 3 | Severe slugging | No | 32 | 2 |
| 4 | Flow instability | No | 343 | 7 |
| 5 | Rapid productivity loss | Yes | 11 | 3 |
| 6 | Quick restriction in PCK | Yes | 6 | 2 |
| 7 | Scaling in PCK | Yes | 36 | 6 |
| 8 | Hydrate in production line | Yes | 14 | 9 |
| 9 | Hydrate in service line | Yes | 57 (14 with labeled events) | 15 |

Several event types are mostly simulated (e.g. class 5: 11 real vs 439 simulated; class 6: 6 real vs 215 simulated), while normal data is 100% real.

## Structure of an instance

- **Sampling:** 1 Hz, timestamp as the DataFrame index, no gaps found within files.
- **Columns:** 27 sensors + `class` + `state`.
- **Labels are per observation:** normal `0`, transient (developing) `100 + k`, event `k`. Example (class 6): ~4 h normal → ~41 min transient → ~2 h event.
- **Unlabeled first hour:** every one of the 1,119 real files begins with exactly 3,600 unlabeled rows. A test comparing first- vs second-hour variability (median ratio 0.99) found no warm-up effect: it is a dataset convention.
- **Fixed lengths:** normal and slugging files are almost all 4.9 h of labeled data; flow-instability files 1.9 h. Scaling (median 45 h) and hydrate files (median 78 h) are much longer.
- **Class 9 folder:** only 14 of 57 real files contain abnormal labels; the rest contain only normal and unlabeled rows.

## Sensors

| Group | Columns | Units |
|---|---|---|
| Valve states | 9 `ESTADO-*` columns | 0, 0.5, 1 |
| Pressures | P-ANULAR, P-PDG, P-TPT, P-MON-CKP, P-JUS-CKP, … | Pa |
| Temperatures | T-TPT, T-PDG, T-JUS-CKP, T-MON-CKP | °C |
| Flow rates | QGL, QBS | m³/s |
| Choke openings | ABER-CKP, ABER-CKGL | % |

`MON` = upstream, `JUS` = downstream; PDG = downhole gauge; TPT = sensor at the subsea Christmas tree.

## Sensor availability: present is not usable

A sensor is **usable** in a file if its 5-minute windows are valid (not missing, not all zeros, not frozen with min = max) at least half the time.

| Sensor | Present (non-empty) | Usable |
|---|---|---|
| T-TPT | 0.92 | 0.85 |
| P-TPT | 0.86 | 0.77 |
| P-ANULAR | 0.82 | 0.63 |
| P-PDG | 0.74 | **0.14** |
| All four together | 0.71 | **0.11** |
| P-MON-CKP | – | 0.55 of normal files; 1.00 of classes 3, 5, 7, 8, 9 |

- P-PDG (downhole pressure) is mostly zeros or frozen values; usable in only 5% of normal files.
- P-ANULAR often holds values for hours and changes in steps.
- Five sensors are essentially never present in real data: T-MON-CKP, P-MON-SDV-P, P-JUS-BS, QBS, PT-P.

## Known quirks and risks

| Issue | Risk | How AssetMind handles it |
|---|---|---|
| Normal is 100% real; some events mostly simulated | Shortcut learning ("looks simulated = event") | Evaluate on real instances only |
| Sensor availability differs by event type | Model recognizes wells or classes by which sensors work | Validity used for data quality only, never as a model input |
| Files from the same well are similar | Inflated scores if a well is in train and test | Leave-one-well-out evaluation |
| Normal movement differs ~1,400× between wells | A pooled "normal" is too wide | Per-well baselines with a no-future rule |
| Normal files appear curated (fixed 4.9 h, calm) | False-alarm rates optimistic | Report false alarms separately for normal files and pre-event normal periods |
| Files are disjoint snapshots, weeks apart | No continuous history for adaptive baselines | Documented limitation; slow events (scaling) remain hard |
| Few wells per event type | Small-sample noise (~1 file ≈ 2 macro-detection points) | Report counts and wells; check random seeds |

## Processed form used for modeling

- 5-minute sliding windows with a 1-minute step, labeled by their last second; per sensor: mean, std, min, max, change and a validity status; plus 1-hour look-back features (level change, waviness).
- ~28 million labeled rows → **475,682 windows**: normal 239,490; transient 183,383 (94% from scaling and hydrate files); event 52,809.
- Model sensors: P-TPT, T-TPT, P-ANULAR, plus P-MON-CKP in the extended set; P-PDG excluded.

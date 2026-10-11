# Rules: Wrasse Card Metrology

## Runtime contract

- Command: `python3 solution.py <public_dir> <submission_out>` ("Your script receives the public
  dataset directory and exact submission CSV path as two positional arguments.")
- Inputs in `<public_dir>`: photos.parquet, train.csv, test.csv, train_targets.csv,
  train_landmarks.csv, sample_submission.csv
- Output: CSV with exactly `item_id,length_mm,morph,spawning,left_points,right_points`, one row per
  test capture; points are JSON lists of fifteen `[x, y]` in that photograph's pixels.
- Environment: not stated. Assumed offline CPU; code uses only numpy, pandas, scipy, scikit-learn,
  Pillow and pyarrow (needed to read the provided parquet).

## Metric

- length 0.35: credit max(0, 1 - |err|/5 mm), skill vs the median training length.
- morph 0.30: +1 right, 0 wrong, -1 for female<->sneaker; classes weigh equally; skill (c-1/3)/(2/3).
- spawning 0.15: 1 - |err|/2, levels weigh equally; skill (c-2/3)/(1/3).
- landmarks 0.20: mean exp(-d^2/(2 s^2)), s = 0.015 x annotated standard length; skill vs the mean
  training position per side. Only annotated photographs are scored.
- Score = 100 x weighted sum, floored at 0. AI baseline to beat: 46.3451.

## Rules to implementation

| # | Rule, quoted from the spec | Kind | How solution.py meets it |
|---|---|---|---|
| 1 | "Do not use any external copy of these photographs or of crops made from them" | data | reads only the files in `<public_dir>` |
| 2 | "any record of these fish" | data | no external records; targets come only from train_targets.csv |
| 3 | "or any model trained on them" | model | every model is fitted inside solution.py on the provided training data; no pretrained weights, no downloads |
| 4 | "The task is to learn from the training captures provided." | method | scale, length, classes and landmarks are all learned from train captures |
| 5 | Leakage controls: ids are opaque, no date/place | integrity | no logic keyed on ids, row order or file order |

## Contradictions, and the reading chosen

- The spec does not list libraries or hardware. Reading chosen: CPU only, no torch, no network.

## Spec facts checked against the files

- train 1115 captures, test 1147, photos 4683 (4524 used by train+test; the rest are landmark-only
  training photographs).
- train_landmarks.csv: 223 photographs, 64 of them belong to train items.
- Points annotated on ~3% of train captures per side.
- Test holds out other islands and later seasons, so validation can only approximate this split.

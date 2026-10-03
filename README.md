# Week 1 — Data Preprocessing & Pipeline

**Varolline AI Engineer internship · Week 1** — *Clean raw input data, extract
relevant features, and establish evaluation datasets.*

This project builds a reproducible preprocessing pipeline for the Titanic
passenger-survival dataset and publishes stratified **train / validation /
test** splits that downstream weeks train on.

## Repository layout

```
week1-data-pipeline/
├── data/
│   ├── raw/titanic.csv          # vendored raw data (datasciencedojo/datasets)
│   └── processed/
│       ├── train.csv            # 623 rows × 27 features + target
│       ├── val.csv              # 134 rows
│       ├── test.csv             # 134 rows
│       └── feature_metadata.json
├── src/
│   ├── titanic_pipeline.py      # cleaning + feature-engineering transformers
│   └── preprocess.py            # end-to-end pipeline entry point
├── reports/
│   └── data_quality_report.md   # missingness, distributions, cleaning decisions
├── requirements.txt
├── LICENSE (MIT)
└── README.md
```

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/preprocess.py
```

The script writes the three processed CSVs, `feature_metadata.json`, and
regenerates the data-quality report. Everything is deterministic
(`random_state=42`).

## What the pipeline does

| Step | Detail |
| --- | --- |
| Missing `Age` (177) | Imputed with the median of the `(Pclass, Sex)` group, learned on the **train split only** |
| Missing `Embarked` (2) | Imputed with the train mode (`S`) |
| Missing `Cabin` (687, 77%) | Converted to a `Deck` feature (first letter, `U` when unknown) instead of dropping |
| Fare outliers | Clipped at the train 99th percentile (249.01) — extreme fares are real, just heavy-tailed |
| Engineered | `Title` (from Name), `FamilySize`, `IsAlone`, `FarePerPerson` |
| Dropped | `PassengerId`, `Name`, `Ticket`, `Cabin` (identifiers / replaced) |
| Encoded | Numerics standardised; categoricals one-hot (`Sex`, `Embarked`, `Deck`, `Title`) |
| Splits | Stratified 70 / 15 / 15 → 623 / 134 / 134; imputation stats fit on train only (no leakage) |

Final feature matrix: **27 features**. See
[`reports/data_quality_report.md`](reports/data_quality_report.md) for the full
analysis and justification of every cleaning decision.

## Reproducibility

The raw CSV is vendored in `data/raw/`, so the pipeline runs fully offline.

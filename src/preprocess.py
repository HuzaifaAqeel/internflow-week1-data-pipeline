"""Week 1 — Data Preprocessing & Pipeline (Varolline AI Engineer internship).

Loads the raw Titanic CSV, applies the cleaning/feature-engineering
pipeline from ``src/titanic_pipeline.py``, and writes stratified
70/15/15 train/val/test splits plus feature metadata.

Imputation statistics (Age group medians, Embarked mode, Fare clip bound)
are learned on the TRAIN split only and applied to val/test, so no
information leaks across splits.

Usage:
    python src/preprocess.py
"""

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from titanic_pipeline import (
    CATEGORICAL_FEATURES,
    DROP_COLS,
    NUMERIC_FEATURES,
    TARGET_COL,
    build_preprocessor,
    load_raw,
)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "titanic.csv"
OUT = ROOT / "data" / "processed"
REPORT = ROOT / "reports" / "data_quality_report.md"
SEED = 42


def main() -> None:
    df = load_raw(RAW)
    print(f"Loaded {len(df)} rows x {df.shape[1]} cols from {RAW}")

    # --- stratified 70/15/15 split on the RAW data (before any fitting) ---
    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=SEED, stratify=df[TARGET_COL]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=SEED, stratify=temp_df[TARGET_COL]
    )
    print(f"Split sizes -> train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

    # --- fit preprocessing on TRAIN only, apply to val/test ---
    prep = build_preprocessor()
    X_train = pd.DataFrame(
        prep.fit_transform(train_df.drop(columns=[TARGET_COL])),
        columns=prep.named_steps["encode"].get_feature_names_out(),
    )
    X_val = pd.DataFrame(
        prep.transform(val_df.drop(columns=[TARGET_COL])), columns=X_train.columns
    )
    X_test = pd.DataFrame(
        prep.transform(test_df.drop(columns=[TARGET_COL])), columns=X_train.columns
    )
    y_train = train_df[TARGET_COL].astype(int).reset_index(drop=True)
    y_val = val_df[TARGET_COL].astype(int).reset_index(drop=True)
    y_test = test_df[TARGET_COL].astype(int).reset_index(drop=True)

    OUT.mkdir(parents=True, exist_ok=True)
    for name, X, y in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
        frame = pd.concat([X.reset_index(drop=True), y.rename(TARGET_COL)], axis=1)
        frame.to_csv(OUT / f"{name}.csv", index=False)
        print(f"Wrote {OUT / f'{name}.csv'}  ({frame.shape[0]} rows, {frame.shape[1]} cols)")

    # --- feature metadata ---
    engineer = prep.named_steps["engineer"]
    metadata = {
        "source": "data/raw/titanic.csv (vendored, datasciencedojo/datasets)",
        "random_state": SEED,
        "split": {"train": len(train_df), "val": len(val_df), "test": len(test_df),
                  "stratified_on": TARGET_COL},
        "missingness_raw": {c: int(df[c].isna().sum()) for c in df.columns},
        "cleaning_decisions": {
            "Age": "imputed with median of (Pclass, Sex) group learned on train; "
                   "global median fallback for unseen groups",
            "Embarked": "imputed with train mode ('S')",
            "Cabin": "converted to Deck (first letter, 'U' when unknown); "
                     "77% missingness made raw Cabin unusable as-is",
            "Fare_outliers": f"clipped at train 99th percentile "
                             f"({engineer.fare_clip_:.2f})",
            "dropped": DROP_COLS,
        },
        "engineered_features": ["Title", "Deck", "FamilySize", "IsAlone", "FarePerPerson"],
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "output_features": X_train.columns.tolist(),
        "age_group_medians": {f"{k[0]}_{k[1]}": float(v)
                              for k, v in engineer.age_medians_.items()},
        "target_balance": {
            "train": float(y_train.mean()), "val": float(y_val.mean()),
            "test": float(y_test.mean()),
        },
    }
    (OUT / "feature_metadata.json").write_text(json.dumps(metadata, indent=2))
    print("Wrote feature metadata.")

    # --- data quality report ---
    lines = [
        "# Titanic Data Quality Report (Week 1)",
        "",
        f"Raw rows: **{len(df)}**, columns: **{df.shape[1]}**.",
        "",
        "## Missingness (raw)",
        "",
        "| Column | Missing | % |",
        "| --- | --- | --- |",
    ]
    for c in df.columns:
        n = int(df[c].isna().sum())
        lines.append(f"| {c} | {n} | {100 * n / len(df):.1f} |")
    lines += [
        "",
        "## Target balance (Survived)",
        "",
        f"Overall survival rate: **{df[TARGET_COL].mean():.3f}**.",
        f"Train: {y_train.mean():.3f} · Val: {y_val.mean():.3f} · Test: {y_test.mean():.3f} "
        "(stratified — rates match).",
        "",
        "## Cleaning decisions",
        "",
        "- **Age** (177 missing): imputed with the median of each "
        "(Pclass, Sex) group — survival-relevant cohorts differ strongly "
        f"(e.g. Pclass-1 female median {engineer.age_medians_[(1, 'female')]:.0f} vs "
        f"Pclass-3 male median {engineer.age_medians_[(3, 'male')]:.0f}); a global "
        "median would wash that out.",
        "- **Embarked** (2 missing): imputed with the mode ('S').",
        "- **Cabin** (687 missing, 77%): kept as **Deck** (first letter, "
        "'U' when unknown) instead of dropping, since deck level correlates "
        "with class/location on the ship; raw cabin codes are too sparse.",
        "- **Fare outliers**: clipped at the train 99th percentile "
        f"({engineer.fare_clip_:.2f}) instead of dropping rows — the "
        "extreme fares are real (luxury tickets), just heavy-tailed.",
        "- **Dropped**: PassengerId (row id), Name (kept as Title), Ticket "
        "(high-cardinality code), Cabin (replaced by Deck).",
        "",
        "## Feature engineering",
        "",
        "- **Title** extracted from Name (Mr/Mrs/Miss/Master/Rare) — strong "
        "survival proxy via social status/sex/age.",
        "- **FamilySize** = SibSp + Parch + 1; **IsAlone** = FamilySize == 1 — "
        "solo travellers survived less often.",
        "- **FarePerPerson** = Fare / FamilySize — normalises ticket price by "
        "party size.",
        "",
        "## Encoding / scaling",
        "",
        "Numerics standardised (mean 0, std 1); categoricals one-hot encoded "
        "(unknown categories ignored at inference). Final matrix: "
        f"**{X_train.shape[1]} features**.",
        "",
        "## Splits",
        "",
        f"Stratified 70/15/15 -> train {len(train_df)}, val {len(val_df)}, "
        f"test {len(test_df)} (random_state={SEED}). Imputation stats learned "
        "on train only and applied to val/test: no leakage.",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n")
    print(f"Wrote {REPORT}")


if __name__ == "__main__":
    main()

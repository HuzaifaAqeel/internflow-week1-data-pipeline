# Titanic Data Quality Report (Week 1)

Raw rows: **891**, columns: **12**.

## Missingness (raw)

| Column | Missing | % |
| --- | --- | --- |
| PassengerId | 0 | 0.0 |
| Survived | 0 | 0.0 |
| Pclass | 0 | 0.0 |
| Name | 0 | 0.0 |
| Sex | 0 | 0.0 |
| Age | 177 | 19.9 |
| SibSp | 0 | 0.0 |
| Parch | 0 | 0.0 |
| Ticket | 0 | 0.0 |
| Fare | 0 | 0.0 |
| Cabin | 687 | 77.1 |
| Embarked | 2 | 0.2 |

## Target balance (Survived)

Overall survival rate: **0.384**.
Train: 0.384 · Val: 0.388 · Test: 0.381 (stratified — rates match).

## Cleaning decisions

- **Age** (177 missing): imputed with the median of each (Pclass, Sex) group — survival-relevant cohorts differ strongly (e.g. Pclass-1 female median 35 vs Pclass-3 male median 26); a global median would wash that out.
- **Embarked** (2 missing): imputed with the mode ('S').
- **Cabin** (687 missing, 77%): kept as **Deck** (first letter, 'U' when unknown) instead of dropping, since deck level correlates with class/location on the ship; raw cabin codes are too sparse.
- **Fare outliers**: clipped at the train 99th percentile (227.53) instead of dropping rows — the extreme fares are real (luxury tickets), just heavy-tailed.
- **Dropped**: PassengerId (row id), Name (kept as Title), Ticket (high-cardinality code), Cabin (replaced by Deck).

## Feature engineering

- **Title** extracted from Name (Mr/Mrs/Miss/Master/Rare) — strong survival proxy via social status/sex/age.
- **FamilySize** = SibSp + Parch + 1; **IsAlone** = FamilySize == 1 — solo travellers survived less often.
- **FarePerPerson** = Fare / FamilySize — normalises ticket price by party size.

## Encoding / scaling

Numerics standardised (mean 0, std 1); categoricals one-hot encoded (unknown categories ignored at inference). Final matrix: **27 features**.

## Splits

Stratified 70/15/15 -> train 623, val 134, test 134 (random_state=42). Imputation stats learned on train only and applied to val/test: no leakage.

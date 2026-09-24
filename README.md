# Credit Risk Scorecard: Development and Implementation

A credit scorecard built in Python on the **Statlog German Credit** dataset. It covers the standard industry pipeline, from raw data to a points-based scorecard and risk rating classes: preprocessing, binning, Weight of Evidence (WoE), Information Value (IV), logistic regression, PSI stability testing and score scaling.

This repository contains the practical part of my **Bachelor's thesis** at the Bucharest University of Economic Studies (ASE), Faculty of Cybernetics, Statistics and Economic Informatics (2025).

**Thesis title:** *Development and Implementation of a Banking Scorecard Model for Credit Risk Assessment*

---

## Why a scorecard?

Two clients with identical incomes can carry very different risks. A scorecard translates a client's profile into a single, easy-to-read number: each characteristic gets points, and the total gives the credit score. A higher score means a lower probability of default (PD).

Scorecards remain the industry standard because they are **transparent and explainable**. This matters for regulatory frameworks such as Basel II/III/IV and **IFRS 9**, which requires banks to estimate expected credit losses (ECL) in advance.

---

## Dataset

- **Source:** [Statlog (German Credit Data)](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data), UCI Machine Learning Repository, loaded with `ucimlrepo`
- **Size:** 1,000 observations, 20 features, 1 binary target
- **Target:** `clasa`, recoded as `0` = good payer, `1` = bad payer (default)
- **Missing values:** none

All variables and categorical codes (e.g. `A11`, `A12`) were renamed into readable Romanian labels. Some categories were merged where there was no meaningful difference in risk. See the [glossary](#variable-glossary) below.

---

## Methodology

```
Raw data → Renaming & recoding → EDA → Binning (ChiMerge) → WoE transform
        → IV-based feature assessment → Logistic regression (4 variants)
        → Model comparison (Accuracy, AUC, confusion matrix) → PSI stability
        → Score scaling (PDO) → Scorecard → Rating classes → Business scenarios
```

1. **Preprocessing:** renamed attributes, recoded categories, converted the target to 0/1.
2. **Exploratory data analysis:** distributions of good vs. bad clients for every numeric and categorical variable, and outlier detection with `toad.detect`.
3. **Binning:** automatic supervised binning of numeric variables with the Chi-square method (`toad.Combiner`, min. 5% of observations per bin). Manual binning was also tested but performed worse.
4. **WoE transformation:** all bins and categories were converted to Weight of Evidence values, then checked for monotonicity and correlation.
5. **Information Value:** each variable's predictive power was measured and interpreted (< 0.02 very weak, 0.02–0.1 weak, 0.1–0.3 moderate, > 0.3 strong).
6. **Logistic regression:** four model variants were compared:

   | Variant | Features | Train/Test split |
   |---|---|---|
   | VAR1 | all variables | 70 / 30 |
   | VAR2 | IV ≥ 0.02 | 70 / 30 |
   | VAR3 | all variables | 80 / 20 |
   | VAR4 | IV ≥ 0.02 | 80 / 20 |

7. **Population Stability Index (PSI):** checked that variable distributions are stable between the train and test sets.
8. **Scaling and scorecard:** log-odds were converted to points with **PDO = 20** and a base score of **600**. Points per attribute = −WoE × β × Factor.
9. **Rating classes:** test scores were split into 7 equal-frequency rating classes (AA → D), and the observed default rate was measured for each class.

---

## Results

### Most predictive variables (Information Value)

| Variable | IV | Strength |
|---|---|---|
| `cont_curent` (checking account status) | 0.639 | strong |
| `durata_luni` (loan duration) | 0.302 | strong |
| `istoric_rambursari` (repayment history) | 0.293 | moderate |
| `economii` (savings) | 0.194 | moderate |
| `scop` (loan purpose) | 0.169 | moderate |
| `varsta_client` (client age) | 0.124 | moderate |

### Model comparison

| Variant | Accuracy | AUC | Misclassified (FN + FP) |
|---|---|---|---|
| VAR1 – all, 70/30 | 0.80 | 0.818 | 43 + 18 = 61 / 300 |
| VAR2 – IV ≥ 0.02, 70/30 | 0.79 | 0.816 | 44 + 18 = 62 / 300 |
| **VAR3 – all, 80/20** | **0.80** | **0.822** | **27 + 13 = 40 / 200** |
| VAR4 – IV ≥ 0.02, 80/20 | 0.80 | 0.819 | 27 + 13 = 40 / 200 |

**VAR3** was selected as the final model because it had the highest AUC. All PSI values are below 0.1, so the population is stable between train and test.

### Rating classes (test set, 200 clients)

| Rating | Score range | Mean score | Observed PD |
|---|---|---|---|
| AA | 696 – 759 | 722 | 3.4% |
| A | 678 – 695 | 686 | 10.7% |
| BB | 665 – 677 | 671 | 13.8% |
| B | 650 – 665 | 658 | 21.4% |
| CC | 631 – 649 | 641 | 27.6% |
| C | 611 – 631 | 621 | 53.6% |
| D | 572 – 609 | 595 | 79.3% |

Default rates rise monotonically from AA to D, which shows the score separates risk levels well.

### Business interpretation

The rating classes were translated into two credit-decision policies:

- **Scenario I, prudent:** AA is auto-approved; A–CC go to additional review (collateral, co-borrowers, lower amounts); C and D are auto-rejected. This gives stronger risk control but a longer "time to yes".
- **Scenario II, permissive:** AA and A are auto-approved; BB–CC get a conditional review with more flexible requirements; C and D are auto-rejected. This gives faster decisions and more volume, at the cost of higher risk.

---



---

## How to run

```bash
git clone https://github.com/rxxiaa/Credit-Scorecard.git
cd Credit-Scorecard
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook Credit_Scorecard.ipynb
```

The dataset is downloaded automatically from the UCI repository, so an internet connection is required.

---

## Tech stack

Python · pandas · NumPy · scikit-learn · toad · matplotlib · seaborn · plotly · Jupyter

---

## Limitations and future work

- The dataset is small (1,000 clients), dates from 1994 and comes from a single country, so it does not represent today's more diverse, digital banking populations.
- The model is validated on a hold-out test set only. Out-of-time validation and cross-validation would strengthen it.
- Possible extensions include alternative data (digital behaviour, transaction history), comparison with machine learning models (e.g. gradient boosting) using explainability tools, and fairness checks on sensitive attributes.

---

## Variable glossary

The code uses Romanian variable names. The main ones:

| Variable | Meaning |
|---|---|
| `cont_curent` |  account |
| `durata_luni` | loan duration (months) |
| `istoric_rambursari` | credit / repayment history |
| `scop` | loan purpose |
| `valoare_credit` | credit amount |
| `economii` | savings account / bonds |
| `vechime_munca` | employment length |
| `procent_rata_venit` | instalment as % of income |
| `statut_sex` | personal status and sex |
| `alti_debitori` | other debtors / guarantors |
| `ani_la_resedinta` | years at current residence |
| `proprietati` | property |
| `varsta_client` | age |
| `rate_alte_institutii` | other instalment plans |
| `situatie_locuinta` | housing |
| `nr_credite_active_banca` | number of existing credits at this bank |
| `tip_ocupatie` | job type |
| `nr_pers_intretinere` | number of dependents |
| `telefon` | has telephone |
| `alta_nationalitate` | foreign worker |
| `clasa` | target: 0 = good, 1 = bad |

---




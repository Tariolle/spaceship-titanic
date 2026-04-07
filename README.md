# Spaceship Titanic

Classical ML approach to the [Kaggle Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic/) competition.

Goal: practice and sharpen non-deep-learning ML skills by iteratively building a decent classifier.

**Result: 80.03% public LB** with a stacking ensemble (XGBoost + LightGBM + ExtraTrees).

## Approach

1. **Feature engineering** -- cabin parsing (deck/number/side), family size from last names, spending aggregates (total, ratios, log transforms), age groups, and a zero-spending flag
2. **Imputation** -- KNN imputation with domain knowledge (CryoSleep passengers have zero spending)
3. **Modeling** -- Optuna-tuned XGBoost + LightGBM + ExtraTrees in a `StackingClassifier` with logistic regression as meta-learner

## Progression

| Submission | CV | Public LB |
|---|---|---|
| XGB+LGBM ensemble | 79.6% | 79.7% |
| KNN imputation + CryoSleep domain knowledge | 79.8% | 79.7% |
| Tuned ExtraTrees | 80.3% | 79.9% |
| Stacking (XGB+LGBM+ET) | 80.3% | 80.0% |

## Setup

```bash
conda create -n spaceship python=3.11 -y
conda activate spaceship
pip install -r requirements.txt
```

## Usage

Download the data from Kaggle and run the notebook:

```bash
kaggle competitions download -c spaceship-titanic -p data/
unzip data/spaceship-titanic.zip -d data/
jupyter notebook spaceship_titanic.ipynb
```

# Stock Return Classifier

Can tomorrow's **direction** of a stock (up / down) be predicted from past price data?
This project compares **Random Forest**, **XGBoost** and an **LSTM** on ~7 years of
**AAPL** daily data, using **walk-forward validation** to avoid look-ahead bias.

> **Honest takeaway:** with these features, next-day direction is essentially **not
> predictable** — all three models land around 48–52% accuracy, i.e. no reliable
> edge over a naive baseline. The value of this project is the **rigorous, honest
> methodology**, not a magic number.

---

## How it works (in plain words)

The question: *looking at how a stock moved over the last few days, can we guess
whether it will go up or down tomorrow?*

1. **Features** — numbers that summarize recent behavior: returns over the last
   1 / 5 / 10 days, how far the price is from its moving average, and how volatile
   it has been.
2. **Target** — for each day we record whether the **next** day closed positive (`1`)
   or negative (`0`).
3. **Models** — Random Forest, XGBoost and an LSTM look for a relationship between
   the features and the next-day direction.
4. **Walk-forward validation** — we always train on the *past* and test on the
   *future*, never the other way around, so the model never "sees" information it
   wouldn't have had in real time (this is what *no look-ahead bias* means).
5. **Baseline** — we compare every model to the dumbest possible rule: "always
   predict up" (53.5%). If a model can't beat that, it has found no real signal.

## Data

- **Source:** Yahoo Finance via `yfinance`
- **Asset:** AAPL, daily, 2018-01-01 → 2025-01-01 (~1,760 rows)
- **Target:** `1` if the next day's return is positive, else `0`
  (`(return.shift(-1) > 0)`) — classes are roughly balanced (≈54% up / 46% down)

## Features

All computed using **only past data** (no future information):

| Feature | Meaning |
|---|---|
| `return_1`, `return_5`, `return_10` | returns over the last 1 / 5 / 10 days |
| `price_vs_ma10`, `price_vs_ma50` | price relative to its 10- / 50-day moving average |
| `volatility_10` | std. dev. of daily returns over the last 10 days |

## Method

- **Validation:** `TimeSeriesSplit(n_splits=5)` — always train on the past, test on
  the following slice, then roll forward. This prevents the future from leaking into
  the training set.
- **Models:** Random Forest (100 trees), XGBoost (100 trees), LSTM (PyTorch,
  30-day sequences, 2 layers, hidden size 64).

## Results (5-fold walk-forward, accuracy)

Using **relative** moving-average features (see below):

| Model | Mean accuracy |
|---|---:|
| **Always "up" (baseline)** | **0.535** |
| LSTM | 0.526 |
| XGBoost | 0.525 |
| Random Forest | 0.500 |

**Effect of fixing the features.** Switching the moving averages from absolute prices
to relative ratios (`Close / ma - 1`) made them comparable across price levels and
improved every model:

| Model | Absolute features | Relative features |
|---|---:|---:|
| XGBoost | 0.483 | **0.525** |
| Random Forest | 0.488 | **0.500** |
| LSTM | 0.519 | **0.526** |

**Reading:** the relative features lift the models close to the 53.5% baseline, but
none beats it. For a near-random target like next-day direction this is the expected,
and honest, result: there is no exploitable edge — though the feature fix shows the
pipeline now behaves correctly.

### Experiment: 5-day horizon

Predicting direction 5 days ahead (instead of 1) **did not help** — it made things worse:

| Model | Mean accuracy (5-day) |
|---|---:|
| **Always "up" (baseline)** | **0.588** |
| LSTM | 0.516 |
| Random Forest | 0.446 |
| XGBoost | 0.445 |

Over 5 days the market goes up far more often, so the "always up" baseline jumps to
58.8% and becomes very hard to beat. The momentum-based models end up well below it —
they fail to capture even the simple upward drift. A longer horizon is not a free win:
it raises the bar more than it raises the models.

## How to run

```bash
pip install -r requirements.txt
jupyter notebook notebooks/01_data_exploration.ipynb
```

## Limitations

- **Single asset (AAPL):** results may not generalize; a multi-ticker test is needed.
- **No transaction costs / strategy backtest yet:** accuracy ≠ profit.
- **LSTM** is trained without feature scaling and for few epochs, so its small edge
  is likely noise rather than a real signal.

---

*Author: Marco Mazzocco — [mazzocco51.github.io](https://mazzocco51.github.io)*

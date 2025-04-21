# ML‑for‑Big‑Pharma‑Stock

> Machine Learning models used to predict future stock trends. Options on a **single company**, as well as **multiple companies**.

---

## Table of Contents

1. [Project Motivation](#project-motivation)
2. [Key Features](#key-features)
3. [Data Sources](#data-sources)
4. [Tech Stack](#tech-stack)
5. [Installation](#installation)
7. [Project Structure](#project-structure)
8. [Example Results](#example-results)

---

## Project Motivation
The Pharmaceutical industry is characetrized by long R&D periods, and can be influenced greatly by public sentiment. The goal of **ML‑for‑Big‑Pharma‑Stock** is to use traditional technical indicators (SMA, RSI, etc) alongside Sentiment Analysis to predict market trends using a variety of ML models.

Specific objectives:

- **Feature engineering**: convert classic indicators (SMA, RSI, MACD, Bollinger Bands…) into a single numerical values (either the datapoint itself, or rolling average) over a window.
- **Multi‑ticker vs. single‑ticker**: Compare results when training data includes multiple companies, vs when training data has just one company.
- **Model zoo**: Logistic Regression, Random Forest, and custom PyTorch Neural Networks for both classification & regression.
- **Transparent evaluation**: plots, residual analyses, correlation heatmaps, and metric comparisons stored alongside code.

## Key Features

| ✔                             | Description                                                                                                    |
| ----------------------------- | -------------------------------------------------------------------------------------------------------------- |
| 🗄 **Automated data pull**    | Uses **Yahoo Finance** via `yfinance` to download OHLCV data for chosen date windows.                          |
| ⚙️ **Indicator pipeline**      | Functions to compute SMA, RSI, MACD (+signal), Bollinger Bands, OBV, ATR & ADX [see `Inputs_No_Sentiment.py`]. |
| 🤖 **Model training scripts** | *Classification* & *Regression* folders for Logistic Regression, Random Forest, and a PyTorch MLP.             |
| 📊 **Evaluation visuals**     | `/Pictures` contains loss curves, scatter plots, residual plots & correlation heatmaps.                        |
| 🏗 **Modular layout**         | Separate folders for **One Company** vs **Multiple Companies** |

## Data Sources

- **Market data**: Yahoo Finance (free API through `yfinance`).
- **Sentiment Analysis**: Reddit posts via personal API link. Sentiment Analysis done with `VADER`

## Tech Stack

- **Python 3.11**
- `pandas`, `numpy`, `yfinance`, `matplotlib`, `seaborn`
- `scikit‑learn`
- `torch` (PyTorch)

```text
pandas>=2.2
numpy>=1.26
yfinance>=0.2
matplotlib>=3.9
seaborn>=0.13
scikit-learn>=1.5
torch>=2.3
```

## Installation

```bash
# 1) Clone the repo
$ git clone https://github.com/dylanjcurran/ML-for-Big-Pharma-Stock.git
$ cd ML-for-Big-Pharma-Stock

# 2) (Recommended) create & activate a virtualenv
$ python -m venv .venv
$ source .venv/bin/activate           # Windows: .venv\Scripts\activate

# 3) Install dependencies
$ pip install -r requirements.txt      # or pip install -e .
```

*Results (MSE, MAE, R² or F1, ROC‑AUC) print to the console and plots are written to the local directory.*

> **Tip**: to change the ticker list or date windows, edit the `companies` list or `START_DATE` / `END_DATE` constants at the top of each script.

## Project Structure

```text
ML-for-Big-Pharma-Stock/
├─ Multiple Companies/
│  └─ Regression Net/
│     ├─ Code/                # Python scripts & indicator helpers
│     ├─ Pictures/            # Auto‑generated evaluation plots
│     └─ New_Training.csv     # Joined dataset used by Neural_Network.py
├─ One Company/
│  ├─ Classification Net/
│  ├─ Regression Net/
│  ├─ Random Forest/
│  └─ Logistic Regression/
└─ CX Project Signup.pdf       # Project Guide (reference)
```

## Example Results

| Loss Curve             | Correlation Heatmap       | Residual Plot   |
| ---------------------- | ------------------------- | --------------- |
| `Loss_Over_Epochs.png` | `Correlation_Heatmap.png` | `Residuals.png` |

# ML‑for‑Big‑Pharma‑Stock

> Machine Learning models used to predict future Pharma stock trends. Options on a **single company**, as well as **multiple companies**.

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

| Feature                       | Description                                                                                                    |
| ----------------------------- | -------------------------------------------------------------------------------------------------------------- |
|**Automated data pull**        | Uses **Yahoo Finance** via `yfinance` to download OHLCV data for chosen date windows.                          |
|**Indicator pipeline**         | Functions to compute SMA, RSI, MACD (+signal), Bollinger Bands, OBV, ATR & ADX [see `Inputs_No_Sentiment.py`]. |
|**Model training scripts**     | *Classification* & *Regression* folders for Logistic Regression, Random Forest, and a PyTorch MLP.             |
|**Evaluation visuals**         | `/Pictures` contains loss curves, scatter plots, residual plots & correlation heatmaps.                        |
|**Modular layout**             | Separate folders for **One Company** vs **Multiple Companies**                                                 |

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

##Execution

- Use `Make_Train_and_Test` files to create training data in .csv format. Data files used in project has names along the lines of `Training.csv`.
- Scripts like `Random_Forest.py` use the prior created .csv file to train a model with the specified title.
- Files like `Inputs.py`, `Outputs.py`, `One_Window.py` are auxillary scripts used by the `Make_Train_and_Test` scripts. You can ignore them.

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

| Loss Curve | Correlation Heatmap | Residual Plot |
|------------|---------------------|----------------|
| ![Loss Curve](https://github.com/user-attachments/assets/97f7e8bc-b4cb-45b1-933f-b4cdce695804) | ![Correlation Heatmap](https://github.com/user-attachments/assets/ef68557c-3172-4311-a58b-7330a364b619) | ![Residual Plot](https://github.com/user-attachments/assets/d7de11e8-7969-460c-a844-ee8f5a5c6e18) |

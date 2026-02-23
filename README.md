# Sea Ice Extent Prediction — Machine Learning

A machine learning project that models and predicts Arctic sea ice extent (SIE) using climate data, including CO₂ concentrations and land-ocean surface temperature anomalies. The project also estimates the first year the Arctic could become effectively "ice-free" during summer.

---

## Overview

Arctic sea ice extent is one of the most closely watched indicators of climate change. This project uses historical climate data — CO₂ concentrations and surface temperature anomalies — as features to train machine learning models that can:

1. **Predict future Arctic sea ice extent (SIE)** across different time horizons.
2. **Estimate the first "ice-free" year** — defined as the first year in which September Arctic sea ice extent falls below 1 million km², widely used as the threshold for an effectively ice-free Arctic.

---

## Data Sources

All datasets are stored in the `data/` directory.

| File | Description | Frequency |
|---|---|---|
| `c02_annmean_gl.csv` | Global marine surface CO₂ concentration | Annual |
| `c02_mm_gl.csv` | Global marine surface CO₂ concentration | Monthly |
| `N_seaice_extent_dailyv4.0.csv` | Northern Hemisphere sea ice extent | Daily |
| `NH.Ts+dSST.csv` | Northern Hemisphere land-ocean surface temperature anomalies | Monthly |

---

## Project Structure

```
SeaIce-Extent-Prediction-ML/
├─ data/
│  ├─ c02_annmean_gl.csv           # Global marine surface CO₂ (annual)
│  ├─ c02_mm_gl.csv                # Global marine surface CO₂ (monthly)
│  ├─ N_seaice_extent_dailyv4.0.csv  # Northern Hemisphere sea ice extent (daily)
│  └─ NH.Ts+dSST.csv               # NH land-ocean surface temperature anomalies
├─ notebooks/
│  ├─ data_analysis.ipynb                       # Exploratory data analysis
│  ├─ first_ice_free_year_prediction.ipynb      # Predicting the first ice-free year
│  └─ SIE_prediction.ipynb                      # Predicting SIE using multiple models
├─ results/
│  ├─ first_ice_free_year.csv                   # Predicted first ice-free year output
│  └─ SIE_prediction.csv                        # SIE model prediction output
├─ source/
│  ├─ models.py                                 # ML model definitions and training
│  ├─ preprocessing.py                          # Data loading and feature engineering
│  ├─ saving_results.py                         # Utilities for exporting results
│  └─ visualisation.py                          # Plotting and visualisation helpers
└─ README.md
```

---

## Notebooks

### `data_analysis.ipynb`
Exploratory data analysis (EDA) of all four datasets. Covers data cleaning, handling missing values, trend visualisation, and correlation analysis between CO₂ levels, temperature anomalies, and sea ice extent.

### `first_ice_free_year_prediction.ipynb`
Uses a linear regression model trained on historical September SIE values to extrapolate the trajectory of Arctic sea ice and predict the first year it is likely to fall below the 1 million km² "ice-free" threshold.

### `SIE_prediction.ipynb`
Trains and compares multiple machine learning models to predict Arctic sea ice extent. Features include CO₂ concentrations and surface temperature anomalies. Model performance is evaluated and results are saved to `results/`.

---

## Source Modules

| Module | Purpose |
|---|---|
| `preprocessing.py` | Loading raw CSVs, cleaning, merging datasets, and engineering features for model input |
| `models.py` | Defining, training, and evaluating ML models |
| `visualisation.py` | Reusable plotting functions for trends, predictions, and model performance |
| `saving_results.py` | Exporting prediction outputs to CSV in the `results/` directory |

---

## Results

Outputs from the models are saved in `results/`:

- **`first_ice_free_year.csv`** — The predicted first year the Arctic sea ice extent drops below 1 million km² in September, based on linear extrapolation of historical trends.
- **`SIE_prediction.csv`** — Predicted sea ice extent values from the trained ML models, alongside actual historical values for comparison.

---

## Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/SeaIce-Extent-Prediction-ML.git
   cd SeaIce-Extent-Prediction-ML
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the notebooks** in order:
   - Start with `data_analysis.ipynb` to understand the data.
   - Then explore `first_ice_free_year_prediction.ipynb` and `SIE_prediction.ipynb` for the modelling work.

---

## Dependencies

The project uses standard Python data science libraries:

- `pandas` — data manipulation
- `numpy` — numerical computing
- `scikit-learn` — machine learning models and evaluation
- `matplotlib` / `seaborn` — data visualisation
- `jupyter` — running notebooks

---

## Notes

- This project was completed as part of **MSc Machine Learning in Science** coursework.





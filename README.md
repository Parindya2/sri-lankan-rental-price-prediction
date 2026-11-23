# Sri Lankan House Rental Price Prediction – Ensemble Learning

Predicts the monthly rent (LKR) of houses and apartments in Sri Lanka using ensemble learning.

- **Data:** 90 rental listings collected manually from ikman.lk (Sri Lankan context)
- **Models:** Decision Tree, Bagging, Random Forest, XGBoost, Stacking
- **Metrics:** MAE, MSE, RMSE, R²
- **Dashboard:** Streamlit app with model comparison, Random Forest vs XGBoost, Bagging vs Stacking, and rent prediction

## Project structure

```
ML_Assignment/
├── data/
│   ├── rental_properties_raw.csv     collected listings (input)
│   └── rental_properties.csv         cleaned data (created by the notebook)
├── notebooks/
│   └── rental_price_prediction.ipynb cleaning, EDA, models, comparison
├── models/                           saved models and result tables (created by the notebook)
├── dashboard/
│   └── app.py                        Streamlit dashboard
├── requirements.txt
└── README.md
```

## 1. Requirements

- Python 3.10 or newer 
- VS Code with the Python and Jupyter extensions (or any Jupyter environment)

## 2. Setup

Open a terminal in the `ML_Assignment` folder.

**Create the virtual environment**
```
python -m venv venv
```

**Activate it**

| System | Command |
|---|---|
| Windows (PowerShell) | `venv\Scripts\activate` |
| Windows (CMD) | `venv\Scripts\activate.bat` |
| Mac / Linux | `source venv/bin/activate` |

`(venv)` should now appear at the start of the terminal line.
If PowerShell blocks the script, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once and try again.

**Install the libraries**
```
pip install -r requirements.txt
```

**Check the installation**
```
python -c "import pandas, sklearn, xgboost, streamlit, joblib; print('All libraries OK')"
```
Expected output: `All libraries OK`

## 3. Run the notebook 

1. Open `notebooks/rental_price_prediction.ipynb` in VS Code.
2. Click **Select Kernel** (top right) and choose the Python from the `venv` folder.
   If it asks to install `ipykernel`, click Install.
3. Click **Run All**. It takes about 1–2 minutes.
4. Every cell should show a number such as `[12]` and no red error.

The notebook creates these files:
- `data/rental_properties.csv`
- `models/*.joblib` (5 trained models)
- `models/cv_results.csv`, `holdout_results.csv`, `feature_importance.csv`, `test_predictions.csv`, `info.json`

The dashboard needs these files, so run the notebook before the dashboard.

## 4. Run the dashboard

In the terminal, from the `ML_Assignment` folder (not inside `dashboard/`):
```
streamlit run dashboard/app.py
```
The browser opens `http://localhost:8501`. If it does not, open that address manually.
If Streamlit asks for an email, press Enter. Stop the dashboard with `Ctrl + C`.

If `streamlit` is not recognised, use `python -m streamlit run dashboard/app.py`.

Approximate cross-validation results (small differences are normal between library versions):

| Model | R² | RMSE (LKR) |
|---|---|---|
| XGBoost | 0.48 | 82,700 |
| Stacking | 0.45 | 84,100 |
| Random Forest | 0.43 | 87,400 |
| Bagging | 0.43 | 86,900 |
| Decision Tree | 0.20 | 101,300 |

## 5. Limitations

- Small dataset (90 listings).
- Getting prices from ikman.lk, not final rents.

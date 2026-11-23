# Ensemble Learning Dashboard - Sri Lankan rental price prediction
# Run from the project root:  streamlit run dashboard/app.py

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "rental_properties.csv"
MODEL_DIR = ROOT / "models"

st.set_page_config(page_title="Ensemble Learning Dashboard", layout="wide")

# stop early if the notebook has not been run yet
if not (MODEL_DIR / "cv_results.csv").exists():
    st.error("Results not found. Run the notebook from top to bottom first.")
    st.stop()

data = pd.read_csv(DATA_FILE)
cv_results = pd.read_csv(MODEL_DIR / "cv_results.csv")
holdout = pd.read_csv(MODEL_DIR / "holdout_results.csv")
test_pred = pd.read_csv(MODEL_DIR / "test_predictions.csv")
importance = pd.read_csv(MODEL_DIR / "feature_importance.csv")
info = json.loads((MODEL_DIR / "info.json").read_text())

st.sidebar.title("Ensemble Learning Dashboard")
page = st.sidebar.radio("Select page", [
    "Dataset",
    "Model Comparison",
    "Random Forest vs XGBoost",
    "Bagging vs Stacking",
    "Predict Rent",
])


def show_table(df):
    st.dataframe(
        df.style.format({"MAE": "{:,.0f}", "MSE": "{:,.0f}", "RMSE": "{:,.0f}", "R2": "{:.3f}",
                         "R2_std": "{:.3f}"}),
        hide_index=True, width="stretch",
    )


if page == "Dataset":
    st.title("Sri Lankan Rental Price Prediction")
    st.write("Monthly rent of houses and apartments, collected manually from ikman.lk.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Listings", info["rows"])
    c2.metric("Average rent (LKR)", f"{data['Monthly_rent'].mean():,.0f}")
    c3.metric("Median rent (LKR)", f"{data['Monthly_rent'].median():,.0f}")

    left, right = st.columns(2)
    with left:
        st.subheader("Average rent by area")
        st.bar_chart(data.groupby("Area_Tier")["Monthly_rent"].mean())
    with right:
        st.subheader("Size vs rent")
        st.scatter_chart(data, x="Size_sqft", y="Monthly_rent", color="Type")

    st.subheader("Data")
    st.dataframe(data, width="stretch")

elif page == "Model Comparison":
    st.title("Model Comparison")
    st.write("Decision Tree, Bagging, Random Forest, XGBoost and Stacking.")

    st.subheader("Cross-validation results (main result)")
    st.caption("5-fold cross-validation repeated 3 times. Lower MAE, MSE, RMSE is better. Higher R2 is better.")
    show_table(cv_results)

    metric = st.selectbox("Metric to plot", ["RMSE", "MAE", "R2"])
    st.bar_chart(cv_results.set_index("Model")[metric])

    st.subheader(f"Test set results (train {info['train_rows']} rows / test {info['test_rows']} rows)")
    show_table(holdout)

elif page == "Random Forest vs XGBoost":
    st.title("Random Forest vs XGBoost")

    both = cv_results[cv_results["Model"].isin(["Random Forest", "XGBoost"])]
    show_table(both)
    st.bar_chart(both.set_index("Model")[["MAE", "RMSE"]])

    st.subheader("Actual vs predicted rent (test set)")
    left, right = st.columns(2)
    with left:
        st.write("Random Forest")
        st.scatter_chart(test_pred, x="Actual", y="Random Forest")
    with right:
        st.write("XGBoost")
        st.scatter_chart(test_pred, x="Actual", y="XGBoost")

    st.subheader("Top features")
    left, right = st.columns(2)
    for col, name in zip([left, right], ["Random Forest", "XGBoost"]):
        top = importance[importance["Model"] == name].nlargest(10, "Importance")
        col.write(name)
        col.bar_chart(top.set_index("Feature")["Importance"])

elif page == "Bagging vs Stacking":
    st.title("Bagging vs Stacking")
    st.write("Bagging averages many trees. Stacking uses a Ridge model to combine Bagging, Random Forest and XGBoost.")

    both = cv_results[cv_results["Model"].isin(["Bagging", "Stacking"])]
    show_table(both)
    st.bar_chart(both.set_index("Model")[["MAE", "RMSE"]])

    st.subheader("Actual vs predicted rent (test set)")
    left, right = st.columns(2)
    with left:
        st.write("Bagging")
        st.scatter_chart(test_pred, x="Actual", y="Bagging")
    with right:
        st.write("Stacking")
        st.scatter_chart(test_pred, x="Actual", y="Stacking")

elif page == "Predict Rent":
    st.title("Predict Monthly Rent")
    st.write("Enter the property details. Each model gives its own prediction.")

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        prop_type = c1.selectbox("Type", sorted(data["Type"].unique()))
        area = c1.selectbox("Area", sorted(data["Area_Tier"].unique()))
        furnished = c1.selectbox("Furnished", ["Yes", "Semi", "No", "Unknown"])
        bedrooms = c2.number_input("Bedrooms", min_value=1, max_value=10, value=3)
        bathrooms = c2.number_input("Bathrooms", min_value=1, max_value=10, value=2)
        size = c2.number_input("Size (sqft)", min_value=200, max_value=10000, value=1500, step=50)
        parking = c3.selectbox("Parking", ["Yes", "No"])
        garden = c3.selectbox("Garden", ["Yes", "No"], index=1)
        has_ac = c3.checkbox("Air conditioning")
        has_servant = c3.checkbox("Servant room")
        has_pool_gym = c3.checkbox("Pool or gym")
        submitted = st.form_submit_button("Predict")

    if submitted:
        # same column names as used in the notebook
        new_house = pd.DataFrame([{
            "Bedrooms": bedrooms, "Bathrooms": bathrooms, "Size_sqft": size,
            "Has_AC": int(has_ac), "Has_Servant_Room": int(has_servant),
            "Has_Pool_Gym": int(has_pool_gym), "Type": prop_type, "Area_Tier": area,
            "Furnished": furnished, "Parking": parking, "Garden": garden,
        }])

        rows = []
        for name in ["Decision Tree", "Bagging", "Random Forest", "XGBoost", "Stacking"]:
            model = joblib.load(MODEL_DIR / (name.lower().replace(" ", "_") + ".joblib"))
            rows.append({"Model": name, "Predicted rent (LKR)": float(model.predict(new_house)[0])})
        result = pd.DataFrame(rows)

        ensemble = result[result["Model"] != "Decision Tree"]["Predicted rent (LKR)"]
        c1, c2 = st.columns(2)
        c1.metric("Average of ensemble models", f"Rs. {ensemble.mean():,.0f}")
        c2.metric("Range", f"Rs. {ensemble.min():,.0f} - Rs. {ensemble.max():,.0f}")

        st.bar_chart(result.set_index("Model"))
        st.dataframe(result.style.format({"Predicted rent (LKR)": "{:,.0f}"}),
                     hide_index=True, width="stretch")
        st.caption("A rough guide only. The models learned from about 90 asking prices.")

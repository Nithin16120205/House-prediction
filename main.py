import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# ---------------- CONFIG ----------------
MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"
DATA_FILE = "housing.csv"
INPUT_FILE = "input.csv"
OUTPUT_FILE = "output.csv"

# ---------------- PIPELINE ----------------
def build_pipeline(num_attribs, cat_attribs):
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs)
    ])

    return full_pipeline

# ---------------- TRAINING ----------------
def train_model():
    print("🚀 Training model...")

    housing = pd.read_csv(DATA_FILE)

    # Create income category
    housing['income_cat'] = pd.cut(
        housing['median_income'],
        bins=[0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5]
    )

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    for train_index, test_index in split.split(housing, housing["income_cat"]):
        train_set = housing.loc[train_index].drop("income_cat", axis=1)
        test_set = housing.loc[test_index].drop("income_cat", axis=1)

    # Save test set
    test_set.to_csv(INPUT_FILE, index=False)

    # Separate labels
    housing_labels = train_set['median_house_value'].copy()
    housing_features = train_set.drop('median_house_value', axis=1)

    num_attribs = housing_features.drop("ocean_proximity", axis=1).columns.tolist()
    cat_attribs = ['ocean_proximity']

    pipeline = build_pipeline(num_attribs, cat_attribs)

    housing_prepared = pipeline.fit_transform(housing_features)

    model = RandomForestRegressor(random_state=42)
    model.fit(housing_prepared, housing_labels)

    # Evaluate
    predictions = model.predict(housing_prepared)
    rmse = np.sqrt(mean_squared_error(housing_labels, predictions))
    print(f"📊 Training RMSE: {rmse:.2f}")

    # Save files
    joblib.dump(model, MODEL_FILE)
    joblib.dump(pipeline, PIPELINE_FILE)

    print("✅ Model & Pipeline saved successfully!")

# ---------------- INFERENCE ----------------
def run_inference():
    print("⚡ Running inference...")

    if not os.path.exists(MODEL_FILE) or not os.path.exists(PIPELINE_FILE):
        print("❌ Model or pipeline not found. Train first.")
        return

    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)

    if not os.path.exists(INPUT_FILE):
        print("❌ input.csv not found!")
        return

    input_data = pd.read_csv(INPUT_FILE)

    # Drop label if exists
    if 'median_house_value' in input_data.columns:
        input_data = input_data.drop('median_house_value', axis=1)

    transformed_input = pipeline.transform(input_data)
    predictions = model.predict(transformed_input)

    input_data['Predicted_Price'] = predictions
    input_data.to_csv(OUTPUT_FILE, index=False)

    print(f"✅ Inference complete! Saved to {OUTPUT_FILE}")

# ---------------- MAIN ----------------
if __name__ == "__main__":
    print("📁 Checking files...")

    # Force training if either file missing
    if not os.path.exists(MODEL_FILE) or not os.path.exists(PIPELINE_FILE):
        train_model()
    else:
        run_inference()
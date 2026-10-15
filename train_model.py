
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

# 1. Load the new dataset
df = pd.read_csv("loan_prediction_train.csv")

# 2. Prepare features and target
X = df.drop(columns=["Loan_Status", "Loan_ID"])
y = df["Loan_Status"].map({"N": 0, "Y": 1})

if y.isna().any():
    raise ValueError("Unexpected values found in Loan_Status.")

# 3. Identify feature types
numeric_features = X.select_dtypes(include="number").columns.tolist()
categorical_features = X.select_dtypes(exclude="number").columns.tolist()

# 4. Build preprocessing pipelines
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
])

# 5. Build the model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
    )),
])

# 6. Split data for validation
X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# 7. Train and evaluate
print("Training Random Forest...")
model.fit(X_train, y_train)

predictions = model.predict(X_valid)
probabilities = model.predict_proba(X_valid)[:, 1]

print("\nValidation accuracy:", round(accuracy_score(y_valid, predictions), 4))
print("Balanced accuracy:", round(balanced_accuracy_score(y_valid, predictions), 4))
print("ROC-AUC:", round(roc_auc_score(y_valid, probabilities), 4))

print("\nClassification report:")
print(classification_report(
    y_valid,
    predictions,
    labels=[0, 1],
    target_names=["Rejected (N)", "Approved (Y)"],
    zero_division=0,
))

print("Confusion matrix (actual rows, predicted columns):")
print(confusion_matrix(y_valid, predictions, labels=[0, 1]))

# 8. Retrain on all labelled data for the final app
print("\nTraining final model on all labelled rows...")
model.fit(X, y)

# 9. Save the complete pipeline
joblib.dump(model, "loan_approval_model.joblib")
print("\nSaved model: loan_approval_model.joblib")

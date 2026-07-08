import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src.preprocessing import add_outcome_label

def build_ml_dataset(df, target_col):
    """Prepare features and target."""
    df = add_outcome_label(df, target_col).dropna(subset=["Outcome_Binary"])

    drop_cols = [target_col, "Outcome_Binary", "Outcome_Label", "player_id", "goalkeeper_id", "goalkeer_id"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    y = df["Outcome_Binary"].astype(int)

    dropped_cols = []
    for col in X.select_dtypes(include=["object", "category"]).columns:
        if X[col].nunique(dropna=True) > 60:
            dropped_cols.append(col)

    X = X.drop(columns=dropped_cols, errors="ignore")
    return X, y, dropped_cols

def train_models(df, target_col, test_size=0.25, random_state=42):
    """Train and compare ML models."""
    X, y, dropped_cols = build_ml_dataset(df, target_col)

    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ], remainder="drop")

    stratify_y = y if y.nunique() == 2 and y.value_counts().min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=250, random_state=random_state, class_weight="balanced"),
        "Gradient Boosting": GradientBoostingClassifier(random_state=random_state)
    }

    results = []
    trained_models = {}

    for name, model in models.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        try:
            y_proba = pipeline.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_proba)
        except Exception:
            y_proba = None
            roc_auc = np.nan

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1-score": f1_score(y_test, y_pred, zero_division=0),
            "ROC-AUC": roc_auc
        })

        trained_models[name] = {
            "pipeline": pipeline,
            "y_test": y_test,
            "y_pred": y_pred,
            "y_proba": y_proba,
            "X_train": X_train,
            "X_test": X_test,
            "feature_columns": X.columns.tolist(),
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "dropped_cols": dropped_cols
        }

    return pd.DataFrame(results), trained_models

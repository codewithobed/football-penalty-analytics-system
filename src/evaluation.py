import pandas as pd

def get_feature_importance(trained_item):
    """Return top global feature importance values."""
    pipeline = trained_item["pipeline"]
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]

    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        return None

    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        values = abs(model.coef_[0])
    else:
        return None

    importance = pd.DataFrame({"Feature": feature_names, "Importance": values})
    return importance.sort_values("Importance", ascending=False).head(20)

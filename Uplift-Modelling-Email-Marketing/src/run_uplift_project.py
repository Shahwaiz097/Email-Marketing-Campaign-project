from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "hillstrom_email_campaign.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
MODEL_DIR = PROJECT_ROOT / "models"

for folder in [PROCESSED_DIR, FIG_DIR, TABLE_DIR, MODEL_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )
    return df


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    df = clean_columns(df)

    df["segment"] = df["segment"].astype(str).str.strip()
    df["treatment"] = np.where(df["segment"].str.lower().eq("no e-mail"), 0, 1)
    df["outcome"] = df["conversion"].astype(int)
    df["customer_id"] = np.arange(1, len(df) + 1)

    return df


def save_summary_tables(df: pd.DataFrame) -> tuple[float, float, float, float]:
    treatment_counts = df["segment"].value_counts().reset_index()
    treatment_counts.columns = ["segment", "count"]
    treatment_counts.to_csv(TABLE_DIR / "treatment_counts.csv", index=False)

    outcome_summary = (
        df.groupby("segment")
        .agg(
            customers=("customer_id", "count"),
            visit_rate=("visit", "mean"),
            conversion_rate=("conversion", "mean"),
            avg_spend=("spend", "mean"),
        )
        .reset_index()
    )
    outcome_summary.to_csv(TABLE_DIR / "outcome_summary_by_segment.csv", index=False)

    treated = df[df["treatment"] == 1]
    control = df[df["treatment"] == 0]

    treated_conversion = treated["outcome"].mean()
    control_conversion = control["outcome"].mean()
    uplift = treated_conversion - control_conversion
    relative_lift = uplift / control_conversion if control_conversion != 0 else np.nan

    simple_uplift_summary = pd.DataFrame(
        {
            "metric": [
                "Treated Conversion Rate",
                "Control Conversion Rate",
                "Uplift",
                "Relative Lift",
            ],
            "value": [treated_conversion, control_conversion, uplift, relative_lift],
        }
    )
    simple_uplift_summary.to_csv(TABLE_DIR / "simple_uplift_summary.csv", index=False)

    return treated_conversion, control_conversion, uplift, relative_lift


def create_charts(df: pd.DataFrame) -> None:
    outcome_summary = pd.read_csv(TABLE_DIR / "outcome_summary_by_segment.csv")

    plt.figure(figsize=(8, 5))
    plt.bar(outcome_summary["segment"], outcome_summary["conversion_rate"])
    plt.title("Conversion Rate by Campaign Segment")
    plt.xlabel("Campaign Segment")
    plt.ylabel("Conversion Rate")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "conversion_rate_by_segment.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(outcome_summary["segment"], outcome_summary["avg_spend"])
    plt.title("Average Spend by Campaign Segment")
    plt.xlabel("Campaign Segment")
    plt.ylabel("Average Spend")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "average_spend_by_segment.png", dpi=150)
    plt.close()


def train_t_learner(df: pd.DataFrame):
    features = [
        "recency",
        "history_segment",
        "history",
        "mens",
        "womens",
        "zip_code",
        "newbie",
        "channel",
    ]

    X = df[features].copy()
    y = df["outcome"].copy()
    treatment = df["treatment"].copy()

    X_train, X_test, y_train, y_test, t_train, t_test = train_test_split(
        X,
        y,
        treatment,
        test_size=0.30,
        random_state=42,
        stratify=treatment,
    )

    categorical_cols = X_train.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    numeric_cols = [col for col in X_train.columns if col not in categorical_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols),
        ]
    )

    treated_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=8,
                    min_samples_leaf=50,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    control_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=8,
                    min_samples_leaf=50,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    treated_model.fit(X_train[t_train == 1], y_train[t_train == 1])
    control_model.fit(X_train[t_train == 0], y_train[t_train == 0])

    treated_pred = treated_model.predict_proba(X_test[t_test == 1])[:, 1]
    control_pred = control_model.predict_proba(X_test[t_test == 0])[:, 1]

    treated_auc = roc_auc_score(y_test[t_test == 1], treated_pred)
    control_auc = roc_auc_score(y_test[t_test == 0], control_pred)

    model_metrics = pd.DataFrame(
        {
            "metric": ["treated_model_auc", "control_model_auc"],
            "value": [treated_auc, control_auc],
        }
    )
    model_metrics.to_csv(TABLE_DIR / "model_metrics.csv", index=False)

    joblib.dump(treated_model, MODEL_DIR / "treated_model.pkl")
    joblib.dump(control_model, MODEL_DIR / "control_model.pkl")

    return treated_model, control_model, treated_auc, control_auc, features


def score_customers(df: pd.DataFrame, treated_model, control_model, features: list[str]) -> pd.DataFrame:
    X = df[features].copy()

    p_treated = treated_model.predict_proba(X)[:, 1]
    p_control = control_model.predict_proba(X)[:, 1]

    scored = df.copy()
    scored["p_conversion_if_treated"] = p_treated
    scored["p_conversion_if_control"] = p_control
    scored["predicted_uplift"] = p_treated - p_control
    scored = scored.sort_values("predicted_uplift", ascending=False)

    scored.to_csv(PROCESSED_DIR / "customers_scored_by_predicted_uplift.csv", index=False)
    return scored


def create_uplift_deciles(scored: pd.DataFrame) -> pd.DataFrame:
    scored = scored.copy()
    scored["decile"] = (
        pd.qcut(
            scored["predicted_uplift"].rank(method="first", ascending=False),
            q=10,
            labels=False,
        )
        + 1
    )

    decile_summary = (
        scored.groupby("decile")
        .apply(
            lambda x: pd.Series(
                {
                    "customers": len(x),
                    "treated_customers": (x["treatment"] == 1).sum(),
                    "control_customers": (x["treatment"] == 0).sum(),
                    "treated_conversion_rate": x.loc[x["treatment"] == 1, "outcome"].mean(),
                    "control_conversion_rate": x.loc[x["treatment"] == 0, "outcome"].mean(),
                    "actual_uplift": x.loc[x["treatment"] == 1, "outcome"].mean()
                    - x.loc[x["treatment"] == 0, "outcome"].mean(),
                    "avg_predicted_uplift": x["predicted_uplift"].mean(),
                }
            )
        )
        .reset_index()
    )

    decile_summary.to_csv(TABLE_DIR / "uplift_by_decile.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(decile_summary["decile"], decile_summary["actual_uplift"], marker="o")
    plt.title("Actual Uplift by Predicted Uplift Decile")
    plt.xlabel("Decile: 1 = Highest Predicted Uplift")
    plt.ylabel("Actual Uplift")
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "actual_uplift_by_decile.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(decile_summary["decile"], decile_summary["avg_predicted_uplift"], marker="o")
    plt.title("Average Predicted Uplift by Decile")
    plt.xlabel("Decile: 1 = Highest Predicted Uplift")
    plt.ylabel("Average Predicted Uplift")
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "predicted_uplift_by_decile.png", dpi=150)
    plt.close()

    return decile_summary


def save_top_customers(scored: pd.DataFrame, top_fraction: float = 0.20) -> pd.DataFrame:
    n_top = int(len(scored) * top_fraction)
    top_customers = scored.head(n_top).copy()

    selected_cols = [
        "customer_id",
        "segment",
        "recency",
        "history_segment",
        "history",
        "mens",
        "womens",
        "zip_code",
        "newbie",
        "channel",
        "p_conversion_if_treated",
        "p_conversion_if_control",
        "predicted_uplift",
    ]

    top_customers[selected_cols].to_csv(
        TABLE_DIR / f"top_{int(top_fraction * 100)}pct_customers_to_target.csv",
        index=False,
    )

    return top_customers


def save_business_summary(
    df: pd.DataFrame,
    top_customers: pd.DataFrame,
    treated_conversion: float,
    control_conversion: float,
    uplift: float,
    relative_lift: float,
    treated_auc: float,
    control_auc: float,
) -> None:
    top_actual_conversion = top_customers["outcome"].mean()
    top_avg_predicted_uplift = top_customers["predicted_uplift"].mean()

    summary = f"""# Business Summary

## Objective

Identify customers who are most likely to respond because of the email campaign.

## Main Results

- Total customers: {len(df):,}
- Customers recommended for targeting: {len(top_customers):,}
- Treated conversion rate: {treated_conversion:.4f}
- Control conversion rate: {control_conversion:.4f}
- Absolute uplift: {uplift:.4f}
- Relative lift: {relative_lift:.2%}
- Treated model AUC: {treated_auc:.4f}
- Control model AUC: {control_auc:.4f}
- Top 20% actual conversion rate: {top_actual_conversion:.4f}
- Top 20% average predicted uplift: {top_avg_predicted_uplift:.4f}

## Recommendation

Target the top 20% of customers ranked by predicted uplift instead of sending emails to every customer. This focuses the campaign on customers most likely to change behaviour because of the email.
"""

    (PROJECT_ROOT / "outputs" / "business_summary.md").write_text(summary, encoding="utf-8")


def main():
    df = load_data()
    treated_conversion, control_conversion, uplift, relative_lift = save_summary_tables(df)
    create_charts(df)
    treated_model, control_model, treated_auc, control_auc, features = train_t_learner(df)
    scored = score_customers(df, treated_model, control_model, features)
    create_uplift_deciles(scored)
    top_customers = save_top_customers(scored)
    save_business_summary(
        df,
        top_customers,
        treated_conversion,
        control_conversion,
        uplift,
        relative_lift,
        treated_auc,
        control_auc,
    )

    print("Uplift modelling project completed successfully.")
    print(f"Outputs saved in: {PROJECT_ROOT / 'outputs'}")


if __name__ == "__main__":
    main()

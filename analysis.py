"""Churn EDA + RandomForest predictor.

1. Loads the synthetic telecom data
2. Pandas EDA: churn rate overall and by key segments
3. One-hot preprocessing -> train/test split (stratified)
4. RandomForest classifier with a held-out test evaluation
5. Feature importance -> charts + findings.json (input for ai_insights.py)
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split

sns.set_theme(style="whitegrid")

df = pd.read_csv("data/churn_data.csv")
df["churned"] = (df["churn"] == "Yes").astype(int)

# ---------------- EDA ------------------------------------------------------
overall_churn = df["churned"].mean()

by_contract = df.groupby("contract")["churned"].agg(["mean", "count"]).round(3)
by_internet = df.groupby("internet_service")["churned"].agg(["mean", "count"]).round(3)
by_payment = df.groupby("payment_method")["churned"].mean().round(3)

# tenure buckets: brand-new vs mid vs loyal customers
df["tenure_bucket"] = pd.cut(df["tenure_months"],
                             bins=[0, 6, 12, 24, 72],
                             labels=["0-6 mo", "7-12 mo", "1-2 yr", "2+ yr"])
by_tenure = df.groupby("tenure_bucket", observed=True)["churned"].mean().round(3)

print(f"Overall churn rate: {overall_churn:.1%}")
print("\nChurn by contract:\n", by_contract)
print("\nChurn by internet service:\n", by_internet)
print("\nChurn by tenure bucket:\n", by_tenure)

# ---------------- preprocessing --------------------------------------------
X = pd.get_dummies(df.drop(columns=["customer_id", "churn", "churned", "tenure_bucket"]),
                    drop_first=True)
y = df["churned"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nFeatures after one-hot encoding: {X.shape[1]}  "
      f"| train {X_train.shape[0]:,} / test {X_test.shape[0]:,}")

# ---------------- model ------------------------------------------------------
model = RandomForestClassifier(
    n_estimators=200, max_depth=None, min_samples_leaf=2,
    class_weight="balanced", random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

proba = model.predict_proba(X_test)[:, 1]
pred = (proba >= 0.5).astype(int)

metrics = {
    "accuracy": round(accuracy_score(y_test, pred), 4),
    "precision": round(precision_score(y_test, pred), 4),
    "recall": round(recall_score(y_test, pred), 4),
    "f1": round(f1_score(y_test, pred), 4),
    "roc_auc": round(roc_auc_score(y_test, proba), 4),
}
print("\n" + classification_report(y_test, pred, target_names=["stayed", "churned"]))

# ---------------- feature importance ----------------------------------------
importance = (pd.Series(model.feature_importances_, index=X.columns)
              .sort_values(ascending=False))
top_features = importance.head(15).round(4).to_dict()
print("\nTop 15 features:\n", importance.head(15).round(4))

# ---------------- charts ------------------------------------------------------
# 1) churn rate by contract and internet service
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
by_contract["mean"].mul(100).sort_values().plot.barh(ax=axes[0], color="#e74c3c")
axes[0].set_title("Churn rate by contract type")
axes[0].set_xlabel("Churn rate (%)")
by_internet["mean"].mul(100).sort_values().plot.barh(ax=axes[1], color="#e67e22")
axes[1].set_title("Churn rate by internet service")
axes[1].set_xlabel("Churn rate (%)")
fig.tight_layout()
fig.savefig("charts/churn_rates.png", dpi=130)
plt.close(fig)

# 2) top feature importances
fig, ax = plt.subplots(figsize=(9, 6))
importance.head(15).iloc[::-1].plot.barh(ax=ax, color="#2980b9")
ax.set_title("Top 15 churn predictors (RandomForest feature importance)")
ax.set_xlabel("Importance (mean decrease in impurity)")
fig.tight_layout()
fig.savefig("charts/feature_importance.png", dpi=130)
plt.close(fig)

# 3) confusion matrix
cm = confusion_matrix(y_test, pred)
fig, ax = plt.subplots(figsize=(4.5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["stayed", "churned"], yticklabels=["stayed", "churned"])
ax.set_title("Confusion matrix (test set)")
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
fig.tight_layout()
fig.savefig("charts/confusion_matrix.png", dpi=130)
plt.close(fig)
print("\nCharts saved to charts/")

# ---------------- findings ----------------------------------------------------
high_risk = df.copy()
high_risk["risk"] = model.predict_proba(pd.get_dummies(
    high_risk.drop(columns=["customer_id", "churn", "churned", "tenure_bucket"]),
    drop_first=True).reindex(columns=X.columns, fill_value=0))[:, 1]
n_high_risk = int((high_risk["risk"] >= 0.5).sum())

findings = {
    "n_customers": len(df),
    "overall_churn_rate_pct": round(overall_churn * 100, 2),
    "churn_by_contract_pct": (by_contract["mean"] * 100).round(2).to_dict(),
    "churn_by_internet_pct": (by_internet["mean"] * 100).round(2).to_dict(),
    "churn_by_tenure_pct": {str(k): round(v * 100, 2) for k, v in by_tenure.items()},
    "churn_by_payment_pct": (by_payment * 100).round(2).to_dict(),
    "avg_tenure_stayed": round(df.loc[df.churn == "No", "tenure_months"].mean(), 1),
    "avg_tenure_churned": round(df.loc[df.churn == "Yes", "tenure_months"].mean(), 1),
    "avg_bill_stayed": round(df.loc[df.churn == "No", "monthly_charges"].mean(), 2),
    "avg_bill_churned": round(df.loc[df.churn == "Yes", "monthly_charges"].mean(), 2),
    "pct_churners_no_tech_support": round(
        (df.loc[df.churn == "Yes", "tech_support"] == "No").mean() * 100, 1),
    "pct_stayers_no_tech_support": round(
        (df.loc[df.churn == "No", "tech_support"] == "No").mean() * 100, 1),
    "model": "RandomForestClassifier (200 trees, balanced class weights)",
    "metrics": metrics,
    "top_features": top_features,
    "n_high_risk_customers": n_high_risk,
    "confusion_matrix": cm.tolist(),
}
with open("findings.json", "w") as f:
    json.dump(findings, f, indent=2)
print("findings.json saved")

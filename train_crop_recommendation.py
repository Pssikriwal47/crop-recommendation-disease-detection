import pandas as pd, numpy as np, joblib, json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

plt.rcParams["figure.dpi"] = 150
FIG = "/home/claude/crop_project/figures"

df = pd.read_csv("/home/claude/crop_project/data/Data.csv")
X = df.drop(columns=["label"])
le = LabelEncoder()
y = le.fit_transform(df["label"])
classes = le.classes_

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ---- Stage 1: AutoAI-style candidate pipeline search ----
candidates = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=42),
}

results = []
for name, model in candidates.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    results.append({"Pipeline": name, "Holdout Accuracy": acc})
    print(f"{name}: {acc:.4f}")

leaderboard = pd.DataFrame(results).sort_values("Holdout Accuracy", ascending=False).reset_index(drop=True)
leaderboard.to_csv("/home/claude/crop_project/models/leaderboard.csv", index=False)
print(leaderboard)

best_name = leaderboard.iloc[0]["Pipeline"]

# ---- Stage 2: Hyperparameter tuning of the winning pipeline (Random Forest expected) ----
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
}
base_rf = RandomForestClassifier(random_state=42)
grid = GridSearchCV(base_rf, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
grid.fit(X_train, y_train)
best_model = grid.best_estimator_
best_params = grid.best_params_
print("Best params:", best_params)

pred = best_model.predict(X_test)
final_acc = accuracy_score(y_test, pred)
print("Final tuned accuracy:", final_acc)

report = classification_report(y_test, pred, target_names=classes, output_dict=True)
with open("/home/claude/crop_project/models/classification_report.json", "w") as f:
    json.dump(report, f, indent=2)

joblib.dump({"model": best_model, "label_encoder": le, "features": list(X.columns)},
            "/home/claude/crop_project/models/crop_recommendation_model.pkl")

with open("/home/claude/crop_project/models/summary.json", "w") as f:
    json.dump({
        "best_pipeline_before_tuning": best_name,
        "final_model": "Random Forest (tuned)",
        "best_params": best_params,
        "final_accuracy": final_acc,
        "leaderboard": leaderboard.to_dict(orient="records"),
    }, f, indent=2)

# ================= FIGURES =================

# Fig 1: AutoAI-style leaderboard (Training Screenshot)
fig, ax = plt.subplots(figsize=(9, 5))
colors = ["#2E7D32" if p == best_name else "#8BA888" for p in leaderboard["Pipeline"]]
bars = ax.barh(leaderboard["Pipeline"], leaderboard["Holdout Accuracy"], color=colors)
ax.set_xlabel("Holdout Accuracy")
ax.set_xlim(0, 1.05)
ax.set_title("AutoAI-style Pipeline Leaderboard — Crop Recommendation", fontsize=13, fontweight="bold")
for bar, val in zip(bars, leaderboard["Holdout Accuracy"]):
    ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, f"{val:.3f}", va="center", fontsize=10)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(f"{FIG}/training_leaderboard.png", facecolor="white")
plt.close()

# Fig 2: Best model confusion matrix (Best Model Screenshot)
cm = confusion_matrix(y_test, pred)
fig, ax = plt.subplots(figsize=(9, 7.5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=classes, yticklabels=classes, ax=ax, cbar=False)
ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")
ax.set_title(f"Best Model: Random Forest (Tuned) — Test Accuracy {final_acc*100:.1f}%", fontsize=13, fontweight="bold")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{FIG}/best_model_confusion_matrix.png", facecolor="white")
plt.close()

# Fig 3: Feature importance (Testing screenshot 1)
importances = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(importances.index, importances.values, color="#2E7D32")
ax.set_title("Feature Importance — Random Forest (Tuned)", fontsize=13, fontweight="bold")
ax.set_xlabel("Relative Importance")
plt.tight_layout()
plt.savefig(f"{FIG}/feature_importance.png", facecolor="white")
plt.close()

# Fig 4: Per-class precision/recall/F1 (Testing screenshot 2)
report_df = pd.DataFrame(report).T.iloc[:-3][["precision", "recall", "f1-score"]]
fig, ax = plt.subplots(figsize=(10, 6))
report_df.plot(kind="bar", ax=ax, color=["#2E7D32", "#66BB6A", "#A5D6A7"])
ax.set_title("Per-Crop Precision / Recall / F1-score (Test Set)", fontsize=13, fontweight="bold")
ax.set_ylabel("Score")
ax.set_ylim(0, 1.15)
ax.legend(loc="lower right", ncol=3)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(f"{FIG}/per_class_metrics.png", facecolor="white")
plt.close()

print("Done. Figures saved to", FIG)

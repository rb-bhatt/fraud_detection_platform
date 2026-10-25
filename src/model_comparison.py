import os
import json
import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,average_precision_score, precision_score,recall_score,f1_score
from sklearn.model_selection import train_test_split

df_clean = pd.read_csv("data/processed/creditcard_clean.csv")
x_data = df_clean.drop(columns="Class")
y_data = df_clean["Class"]
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42, stratify=y_data)

x_fit, x_val, y_fit, y_val = train_test_split(x_train,y_train,test_size=0.2,random_state=42,stratify=y_train)
scale_pos_weight = (y_fit == 0).sum() / (y_fit == 1).sum()
models={
    "Logistic Regression": LogisticRegression(class_weight="balanced",max_iter=2000,random_state=42),
    "Decision Tree": DecisionTreeClassifier(class_weight="balanced",random_state=42),
    "Random Forest": RandomForestClassifier(class_weight="balanced",n_estimators=100,random_state=42),
    "XGBoost": XGBClassifier(n_estimators=100,random_state=42,eval_metric="logloss",scale_pos_weight=scale_pos_weight,n_jobs=-1),
}
comparison_results = []
validation_probabilities = {}
comparison_threshold = 0.60
for name,model in models.items():
    print(f"Training {name}...")
    model.fit(x_fit, y_fit)
    model_filename = name.lower().replace(" ", "_") + "_model.pkl"
    with open(f"models/{model_filename}", "wb") as model_file:
        pickle.dump(model, model_file)
    val_prob=model.predict_proba(x_val)[:,1]
    validation_probabilities[name] = val_prob
    val_pred = (val_prob >= comparison_threshold).astype(int)
    print(f"=== {name}: Validation Results ===")
    confusion_metrics=confusion_matrix(y_val,val_pred)
    classification__report=classification_report(y_val,val_pred)
    accuracy=accuracy_score(y_val,val_pred)
    precision=precision_score(y_val,val_pred, zero_division=0)
    recall=recall_score(y_val,val_pred, zero_division=0)
    F1_Score=f1_score(y_val,val_pred, zero_division=0)
    PR_AUC=average_precision_score(y_val,val_prob)
    comparison_results.append({
        "Model": name,
        "Precision": precision,
        "Recall": recall,
        "F1": F1_Score,
        "PR-AUC": PR_AUC,
        "Accuracy": accuracy
    })
comparison_df = pd.DataFrame(comparison_results)

print("MODEL COMPARISON (Threshold = 0.60)")
print(comparison_df.round(4).sort_values("PR-AUC", ascending=False).to_string(index=False))

val_prob = validation_probabilities["XGBoost"]
thresholds=[0.1,0.2,0.3,0.4,0.5,0.6, 0.7, 0.8, 0.9]
threshold_result=[]
val_prob = validation_probabilities["XGBoost"]
for threshold in thresholds:
        predictions = (val_prob >= threshold).astype(int)
        threshold_result.append({
            "Threshold": threshold,
            "Precision": precision_score(y_val, predictions, zero_division=0),
            "Recall": recall_score(y_val, predictions, zero_division=0),
            "F1": f1_score(y_val, predictions, zero_division=0)
            })
threshold_df = pd.DataFrame(threshold_result)
print("XGBoost: THRESHOLD COMPARISON")
print(threshold_df.round(4).to_string(index=False))
    
model = models["XGBoost"]
val_prob = validation_probabilities["XGBoost"]
selected_threshold = 0.60
val_pred = (val_prob >= selected_threshold).astype(int)
print(f"XGBoost: DETAILED VALIDATION RESULTS (Threshold = {selected_threshold:.2f})")
print("\nConfusion Matrix:")
print(confusion_matrix(y_val, val_pred))
print("\nClassification Report:")
print(classification_report(
        y_val,
        val_pred,
        zero_division=0
))
print("Accuracy:", accuracy_score(y_val, val_pred))
print("Precision:", precision_score(
        y_val, val_pred, zero_division=0
))
print("Recall:", recall_score(
        y_val, val_pred, zero_division=0
))
print("F1 Score:", f1_score(
        y_val, val_pred, zero_division=0
))
print("PR-AUC:", average_precision_score(y_val, val_prob))

comparison_df = comparison_df.sort_values("PR-AUC", ascending=False)
os.makedirs("reports", exist_ok=True)
comparison_df.to_csv("reports/model_comparison.csv", index=False)
threshold_df.to_csv("reports/xgboost_threshold_comparison.csv", index=False)
print("\nResults saved to:")
print("- reports/model_comparison.csv")
print("- reports/xgboost_threshold_comparison.csv")

model="XGBoost"
selected_threshold = 0.6
test_prob= models["XGBoost"].predict_proba(x_test)[:,1]
test_pred=(test_prob >= selected_threshold).astype(int)
print(f"XGBoost: TEST RESULTS (Threshold = 0.60)")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, test_pred))
print("\nPrecision:")
print(precision_score(y_test, test_pred, zero_division=0))
print("\nRecall:")
print(recall_score(y_test, test_pred, zero_division=0))
print("\nF1 Score:")
print(f1_score(y_test, test_pred, zero_division=0))
print("\nPR-AUC:")
print(average_precision_score(y_test, test_prob))
print("\nAccuracy:")
print(accuracy_score(y_test, test_pred))
XGBoost_test={
     "model":"XGBoost",
     "threshold":selected_threshold,
     "precision":float(precision_score(y_test, test_pred, zero_division=0)),
     "recall": float(recall_score(y_test, test_pred, zero_division=0)),
     "f1_score": float(f1_score(y_test, test_pred, zero_division=0)),
     "pr_auc": float(average_precision_score(y_test, test_prob)),
     "accuracy": float(accuracy_score(y_test, test_pred))
}
os.makedirs("reports", exist_ok=True)
with open("reports/final_test_metrics.json", "w") as f:
    json.dump(XGBoost_test, f)
print("\nFinal test metrics saved to reports/final_test_metrics.json")
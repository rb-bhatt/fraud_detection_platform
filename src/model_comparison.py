import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,average_precision_score, precision_score,recall_score,f1_score
from sklearn.model_selection import train_test_split
from analyze_dataset import x_train, x_test, y_train, y_test
x_fit, x_val, y_fit, y_val = train_test_split(x_train,y_train,test_size=0.2,random_state=42,stratify=y_train)
scale_pos_weight = (y_fit == 0).sum() / (y_fit == 1).sum()
models={
    "Logistic Regression": LogisticRegression(class_weight="balanced",max_iter=2000,random_state=42),
    "Decision Tree": DecisionTreeClassifier(class_weight="balanced",random_state=42),
    "Random Forest": RandomForestClassifier(class_weight="balanced",n_estimators=100,random_state=42),
    "XGBoost": XGBClassifier(class_weight="balanced",n_estimators=100,random_state=42,eval_metric="logloss",scale_pos_weight=scale_pos_weight,n_jobs=-1),
}
comparison_results = []
validation_probabilities = {}
threshold = 0.50
for name,model in models.items():
    print(f"Training {name}...")
    model.fit(x_fit, y_fit)
    val_prob=model.predict_proba(x_val)[:,1]
    validation_probabilities[name] = val_prob
    val_pred = (val_prob >= threshold).astype(int)
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
        "Classification Report": classification__report,
        "Confusion Metrics": confusion_metrics.tolist(),
        "Precision": precision,
        "Recall": recall,
        "F1": F1_Score,
        "PR-AUC": PR_AUC,
        "Accuracy": accuracy
    })
    comparison_df = pd.DataFrame(comparison_results)

    print("MODEL COMPARISON (Threshold = 0.50)")
    print(comparison_df.round(4).sort_values("PR-AUC", ascending=False).to_string(index=False))

    val_prob = validation_probabilities["Logistic Regression"]
    thresholds=range(0.1,0.9,0.1)
    threshold_result=[]
    for threshold in thresholds:
        predictions = (val_prob >= threshold).astype(int)
        threshold_result.append({
            "Threshold": threshold,
            "Precision": precision_score(y_val, predictions, zero_division=0),
            "Recall": recall_score(y_val, predictions, zero_division=0),
            "F1": f1_score(y_val, predictions, zero_division=0)
            })
    threshold_df = pd.DataFrame(threshold_result)
    print("LOGISTIC REGRESSION: THRESHOLD COMPARISON")
    print(threshold_df.round(4).to_string(index=False))
    
    model = models["Logistic Regression"]
    val_prob = validation_probabilities["Logistic Regression"]
    val_pred = (val_prob >= 0.50).astype(int)
    print("LOGISTIC REGRESSION: DETAILED VALIDATION RESULTS")
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

    comparison_summary = comparison_df[
        ["Model", "Precision", "Recall", "F1", "PR-AUC", "Accuracy"]
    ]
    comparison_summary.to_csv("reports/model_comparison.csv", index=False)
    threshold_df.to_csv("reports/threshold_comparison.csv", index=False)
    print("\nResults saved to:")
    print("- reports/model_comparison.csv")
    print("- reports/threshold_comparison.csv")
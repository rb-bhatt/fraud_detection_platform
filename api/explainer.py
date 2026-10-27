import shap
import pickle
import pandas as pd
with open("models/xgboost_model.pkl","rb") as open_file:
    model=pickle.load(open_file)
explainer=shap.TreeExplainer(model)
def explain_transaction(transaction_df:pd.DataFrame)->list[dict]:
    if not isinstance(transaction_df,pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    if transaction_df.empty:
        raise ValueError("Input DataFrame is empty.")

    shap_values=explainer.shap_values(transaction_df)
    if isinstance(shap_values,list):
        shap_values=shap_values[0]

    values=shap_values[0]
    if getattr(values,"ndim",1)==2:
        predicted_class = int(model.predict(transaction_df)[0])
        values = values[:, predicted_class]
    ranked_indices = sorted(range(len(values)),key=lambda i: abs(float(values[i])),reverse=True,)[:3]

    return [
        {
            "feature": str(transaction_df.columns[i]),
            "contribution": float(values[i]),
        }
        for i in ranked_indices
    ]
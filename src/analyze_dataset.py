import pandas as pd
import json
import pickle
df=pd.read_csv("data/raw/creditcard.csv")
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.metrics import average_precision_score,precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler

print(df.shape)
print(df.columns.tolist())
print(df.head())
print(df.dtypes)
print(df["Class"].value_counts())
print(df["Class"].mean()*100)
print("Missing Value:",df.isnull().sum().sum())
print("Duplicate Value:",df.duplicated().sum())
print(df.groupby("Class")["Amount"].describe())
print(df["Class"].value_counts(normalize=True)*100)
class_distribution=df["Class"].value_counts(normalize=True)*100
class_distribution=class_distribution.reset_index()
class_distribution.columns=["Class","Percentage"]
class_distribution.to_csv("reports/class_distribution.csv",index=False)

def data_quailty_report(df:pd.DataFrame)->dict:
    report = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "missing_values": df.isnull().sum().sum(),
        "duplicate_values": df.duplicated().sum()
    }
    print("Shape:", report["shape"])
    print("Columns:", report["columns"])
    print("Missing Values:", report["missing_values"])
    print("Duplicate Values:", report["duplicate_values"])
    return report

def class_distribution_report(df:pd.DataFrame):
    count=df["Class"].value_counts()
    percentage=df["Class"].value_counts(normalize=True)*100
    class_distribution=percentage.reset_index()
    class_distribution.columns=["Class","Percentage"]
    class_distribution["Count"]=count.values
    return class_distribution

def amount_summary_by_class(df:pd.DataFrame):
    summary=df.groupby("Class")["Amount"].agg(["count","mean","median","min","max"])
    return summary
print(amount_summary_by_class(df))

def built_analysis_report(df:pd.DataFrame):
    report={
        "rows":df.shape[0],
        "columns":df.shape[1],
        "missing values":int(df.isnull().sum().sum()),
        "duplicate values":int(df.duplicated().sum()),
        "fraud count":df["Class"].value_counts().to_dict(),
        "fraud percentage":df["Class"].value_counts(normalize=True).mul(100).to_dict()}
    return report
print(built_analysis_report(df))

with open("reports/analysis_report.json","w") as open_file:
    json.dump(built_analysis_report(df),open_file,indent=4)
    print("Analysis report successfully written to analysis_report.json")

duplicates= df.duplicated().sum()
clean_df=df.drop_duplicates()
print("Original Rows:",df.shape[0])
print("Duplicated Rows:",clean_df.shape[0])
print("No of rows removed:",duplicates)

with open("data/processed/creditcard_clean.csv","w") as open_file:
    clean_df.to_csv(open_file,index=False)
    print("Cleaned data successfully written to creditcard_clean.csv")

print(df["Class"].value_counts())
print(clean_df["Class"].value_counts())
fraud_rows=df["Class"].value_counts().get(1,0)
if fraud_rows>0:
    with open("data/processed/creditcard_clean.csv","w") as open_file:
        clean_df.to_csv(open_file,index=False)
    print("Fraud rows still exist:", fraud_rows)
    print("Cleaned data successfully written to creditcard_clean.csv")
else:
    print("No fraud rows exist in the dataset.")

df=pd.read_csv("data/processed/creditcard_clean.csv")
x =df.drop(columns="Class")
y=df["Class"]
print(x.shape)
print(y.shape)
print(y.value_counts())

feature_names = x.columns.tolist()
with open("models/feature_names.json", "w") as open_file:
    json.dump(feature_names, open_file, indent=4)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)
print(x_train.shape[0],x_test.shape[0],y_train.shape[0],y_test.shape[0])
print(y_train.value_counts(),y_test.value_counts())

model=LogisticRegression(class_weight="balanced",max_iter=2000,random_state=42,solver='liblinear')
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print(y_pred)

print("Confusion Metrics:",confusion_matrix(y_test,y_pred))
print("Classification Report:",classification_report(y_test,y_pred))
print('Accuracy:',accuracy_score(y_test,y_pred))

model_probabilty=model.predict_proba(x_test)[:,1]
print("PR-AUC:", average_precision_score(y_test,model_probabilty))

scaler=StandardScaler()
x_train_scaler=scaler.fit_transform(x_train)
x_test_scaler=scaler.transform(x_test)
model_scaled=LogisticRegression(class_weight="balanced",max_iter=2000,random_state=42)
model_scaled.fit(x_train_scaler,y_train)
y_pred_new=model_scaled.predict(x_test_scaler)
print("New Classification Report",classification_report(y_test,y_pred_new))
model_probabilty_new=model_scaled.predict_proba(x_test_scaler)[:,1]
print("New PR-AUC:", average_precision_score(y_test,model_probabilty_new))

with open("models/scale.pkl","wb")as open_file:
    pickle.dump(scaler,open_file)

with open("models/logistic_model.pkl","wb")as open_file:
    pickle.dump(model_scaled,open_file)

precision,recall,f1,_=precision_recall_fscore_support(y_test,y_pred_new,average="binary",zero_division=0)
model_dict={
    "precision_score":float(precision),
    "recall_score":float(recall),
    "f1_score":float(f1),
    "PR-AUC":float(average_precision_score(y_test,model_probabilty_new)),
    "accuracy_score":float(accuracy_score(y_test,y_pred_new)),
}
with open("reports/baseline_metrics.json","w") as open_file:
    json.dump(model_dict,open_file,indent=4)
print("Baseline metrics successfully written to baseline_metrics.json")
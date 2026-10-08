import pandas as pd
df=pd.read_csv("data/raw/creditcard.csv")

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

print(data_quailty_report(df))
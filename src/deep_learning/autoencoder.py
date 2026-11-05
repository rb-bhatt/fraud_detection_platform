import pandas as pd
import numpy as np
import torch
from torch import nn
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve,confusion_matrix,classification_report
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

df = pd.read_csv("data/raw/creditcard.csv")

print(df.shape)
print(df.info())
print(df.head())
print(df.columns)
print(df["Class"].value_counts())

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

normal_df= df[df["Class"]==0]
X_normal= normal_df.drop(["Class"],axis=1)

X_train, X_val_normal= train_test_split(X_normal,test_size=0.2,random_state=42)
print("Training shape:", X_train.shape)
print("Validation shape:", X_val_normal.shape)

scaler= StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_val_normal_scaled= scaler.transform(X_val_normal)
print("Training mean:", np.mean(X_train_scaled))
print("Training std:", np.std(X_train_scaled))
print("Validation mean:", np.mean(X_val_normal_scaled))
print("Validation std:", np.std(X_val_normal_scaled))
print("Validation shape:", X_val_normal_scaled.shape)

X_train_tensor=torch.tensor(X_train_scaled,dtype=torch.float32)
X_val_tensor=torch.tensor(X_val_normal_scaled,dtype=torch.float32)
print(X_train_tensor.shape)
print(X_train_tensor.dtype)

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encode= nn.Sequential(nn.Linear(30,16),nn.ReLU(),nn.Linear(16,8) )
        self.decoder=nn.Sequential(nn.Linear(8,16),nn.ReLU(),nn.Linear(16,30))

    def forward(self,x):
        encoder=self.encode(x)
        decoder=self.decoder(encoder)
        return decoder

model= Autoencoder()
output = model(X_train_tensor[:5])
print("Input shape:", X_train_tensor[:5].shape)
print("Output shape:", output.shape)

criterion = nn.MSELoss()
loss=criterion(output,X_train_tensor[:5])
print("Loss:", loss.item())

optimizer=torch.optim.Adam(model.parameters(),lr=0.001)

train_dataset=TensorDataset(X_train_tensor)
train_loader=DataLoader(train_dataset,batch_size=64,shuffle=True)

validate_dataset=TensorDataset(X_val_tensor)
validate_loader=DataLoader(validate_dataset,batch_size=64,shuffle=False)

epochs=20
for epoch in range(epochs):
    model.train()
    running_loss=0.0
    for batch in train_loader:
        transactions=batch[0]
        reconstructed = model(transactions)
        loss=criterion(reconstructed,transactions)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * transactions.size(0)
    epoch_loss = running_loss / len(train_dataset)
    
    model.eval()
    val_loss=0.0
    with torch.no_grad():    
        for batch in validate_loader:
            transactions=batch[0]
            reconstructed=model(transactions)
            loss=criterion(reconstructed,transactions)
            val_loss += loss.item() * transactions.size(0)
        val_loss=val_loss / len(validate_dataset)
    print(
        f"Epoch [{epoch+1}/{epochs}], "
        f"Train Loss: {epoch_loss:.6f} "
        f"Validation Loss: {val_loss:.6f}"
    )

def calculate_reconstruction_error(model,data):
    model.eval()
    with torch.no_grad():
        reconstructed=model(data)
        errors=torch.mean((data - reconstructed)**2,dim=1)
    return errors

normal_errors=calculate_reconstruction_error(model,X_val_tensor)

plt.figure(figsize=(10,6))
plt.hist(normal_errors.numpy(),bins=50)
plt.xlabel("Reconstruction Error")
plt.ylabel("Number of Transactions")
plt.title("Reconstruction Error Distribution - Normal Transactions")
plt.show()

train_df,temp_df= train_test_split(df,test_size=0.3,random_state=42,stratify=df["Class"])
val_df,test_df=train_test_split(temp_df,test_size=0.5,random_state=42,stratify=temp_df["Class"])

normal_train_df = train_df[train_df["Class"] == 0]
X_train = normal_train_df.drop(columns=["Class"])

X_val = val_df.drop(columns=["Class"])
y_val = val_df["Class"]
X_val_normal = val_df[val_df["Class"] == 0].drop(columns=["Class"])

X_test = test_df.drop(columns=["Class"])
y_test = test_df["Class"]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_val_normal_scaled = scaler.transform(X_val_normal)
X_test_scaled = scaler.transform(X_test)

X_train_tensor=torch.tensor(X_train_scaled, dtype=torch.float32)
X_val_tensor=torch.tensor(X_val_scaled,dtype=torch.float32)
X_val_normal_tensor=torch.tensor(X_val_normal_scaled,dtype=torch.float32)
X_test_tensor=torch.tensor(X_test_scaled,dtype=torch.float32)
print("Training class distribution:")
print(train_df["Class"].value_counts())
print("\nValidation class distribution:")
print(val_df["Class"].value_counts())
print("\nTesting class distribution:")
print(test_df["Class"].value_counts())
print("\nShapes:")
print("Training:", X_train_tensor.shape)
print("Validation:", X_val_tensor.shape)
print("Testing:", X_test_tensor.shape)

train_dataset=TensorDataset(X_train_tensor)
train_loader=DataLoader(train_dataset,batch_size=64,shuffle=True)
validate_dataset=TensorDataset(X_val_normal_tensor)
validate_loader=DataLoader(validate_dataset,batch_size=64,shuffle=False)

model= Autoencoder()
criterion=nn.MSELoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)

epochs=20
for epoch in range(epochs):
    model.train()
    running_loss=0.0
    for batch in train_loader:
        transactions=batch[0]
        reconstructed=model(transactions)
        loss=criterion(reconstructed,transactions)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * transactions.size(0)
    epoch_loss = running_loss / len(train_dataset)
    model.eval()
    val_loss=0.0
    with torch.no_grad():
        for batch in validate_loader:
            transactions=batch[0]
            reconstructed=model(transactions)
            loss=criterion(reconstructed,transactions)
            val_loss += loss.item() * transactions.size(0)
    val_loss=val_loss / len(validate_dataset)
    print(
        f"Epoch [{epoch+1}/{epochs}], "
        f"Train Loss: {epoch_loss:.6f}, "
        f"Validation Loss: {val_loss:.6f}"
    )

def calculate_reconstruction_error(model, data):
    model.eval()
    with torch.no_grad():
        reconstructed = model(data)
        errors = torch.mean((data - reconstructed) ** 2,dim=1)
    return errors

normal_val_errors = calculate_reconstruction_error(model,X_val_normal_tensor)

all_val_errors = calculate_reconstruction_error(model,X_val_tensor)

fraud_val_errors = all_val_errors[torch.tensor(y_val.to_numpy() == 1)]
normal_val_errors = all_val_errors[torch.tensor(y_val.to_numpy() == 0)]

print("Normal transactions")
print("Mean:", normal_val_errors.mean().item())
print("Median:", normal_val_errors.median().item())

print("\nFraudulent transactions")
print("Mean:", fraud_val_errors.mean().item())
print("Median:", fraud_val_errors.median().item())

val_errors= all_val_errors.numpy()
precision, recall, thresholds = precision_recall_curve(y_val, val_errors)
f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)

best_idx= np.argmax(f1_scores)
best_threshold= thresholds[best_idx]
best_f1= f1_scores[best_idx]
print(f"Best threshold: {best_threshold:.6f}")
print(f"Best validation F1: {best_f1:.4f}")
print(f"Validation precision: {precision[best_idx]:.4f}")
print(f"Validation recall: {recall[best_idx]:.4f}")

test_errors=calculate_reconstruction_error(model,X_test_tensor)
y_pred_test = (test_errors.numpy() > best_threshold).astype(int)
print("Confusion Matrix:",confusion_matrix(y_test,y_pred_test))
print("Classification Report:",classification_report(y_test,y_pred_test,digits=4,zero_division=0))

torch.save(model.state_dict(),"models/deep_learning/autoencoder.pth")
import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os

# Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
dy_csv_path = os.path.join(script_dir, "student_performance_dataset.csv")
dy_saved_model_path = os.path.join(script_dir, "lstm_model.pth")

# load csv
df = pd.read_csv(dy_csv_path)
df['Final_Result'] = df['Final_Result'].map({'Pass': 1, 'Fail': 0})

categorical_cols = ['Gender', 'Internet_Access', 'Previous_Grade', 'Extra_Curricular']
numeric_cols = ['Age', 'Study_Hours_per_Week', 'Attendance_Rate']
seq_len = 5

# encoding categorical features
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# normalize numeric features
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Split inputs and targets
X = df.drop(['Student_ID', 'Final_Result', 'Family_Income'], axis=1).values.astype(np.float32)
y = df['Final_Result'].values.astype(np.int64)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Dataset
class SeqStudentDataset(Dataset):
    def __init__(self, X, y, seq_len):
        self.X = X
        self.y = y
        self.seq_len = seq_len
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        x_seq = np.tile(self.X[idx], (self.seq_len, 1))
        return torch.tensor(x_seq), torch.tensor(self.y[idx])

train_ds = SeqStudentDataset(X_train, y_train, seq_len)
test_ds = SeqStudentDataset(X_test, y_test, seq_len)
train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=8)

# model
class LSTMmodel_class(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, output_dim=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h.squeeze(0))

input_dim = X.shape[1]
model = LSTMmodel_class(input_dim)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# training
for epoch in range(15):
    model.train()
    total_loss, correct = 0, 0
    for xb, yb in train_loader:
        pred = model(xb)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (pred.argmax(1) == yb).sum().item()
    acc = 100 * correct / len(train_ds)
    print(f"Epoch {epoch+1} - Loss: {total_loss:.4f} - Accuracy: {acc:.2f}%")

# evaluation
model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for xb, yb in test_loader:
        pred = model(xb)
        y_true.extend(yb.numpy())
        y_pred.extend(torch.argmax(pred, dim=1).numpy())

# metrics
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)

print("\nLSTM Classification Evaluation Results:")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1 Score : {f1:.4f}")
print("\nConfusion Matrix:")
print(cm)

# confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Fail", "Pass"])
disp.plot(cmap=plt.cm.Blues)
plt.title("LSTM Confusion Matrix")
plt.grid(False)
plt.tight_layout()
plt.savefig(r"figures/lstm_confusion_matrix.png", dpi=300)
plt.show()
'''
LSTM - for Academic outcome prediction
'''
import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import Dataset, DataLoader
import os

# paths

script_dir = os.path.dirname(os.path.abspath(__file__))
dynamic_csv_path = os.path.join(script_dir, "student_performance_dataset.csv")
dynamic_saved_model_path = os.path.join(script_dir, "lstm_model.pth")
# Load and prepare data

df = pd.read_csv(dynamic_csv_path)
print(dynamic_csv_path)
df['Final_Result'] = df['Final_Result'].map({'Pass': 1, 'Fail': 0})

categorical_cols = ['Gender', 'Internet_Access', 'Previous_Grade', 'Extra_Curricular']
numeric_cols = ['Age', 'Study_Hours_per_Week', 'Attendance_Rate']

# encode categorical features
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# scaling numerical features
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# drop unwanted columns from dataframe 
X_df = df.drop(['Student_ID', 'Final_Result', 'Family_Income'], axis=1)
y_data = df['Final_Result'].values.astype(np.int64)

# convert dataframe to numpy
X_data = X_df.values.astype(np.float32)

# Create sequences by repeating each student features 5 times
seq_len = 5

class SeqStudentDataset(Dataset):
    def __init__(self, X, y, seq_len):
        self.X = X
        self.y = y
        self.seq_len = seq_len
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):

        # repeating  same features seq_len (5) times
        x_seq = np.tile(self.X[idx], (self.seq_len, 1))
        return torch.tensor(x_seq), torch.tensor(self.y[idx])

train_dataset = SeqStudentDataset(X_data, y_data, seq_len=seq_len)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# LSTM model
class LSTMmodel_class(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, output_dim=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        _, (h, _) = self.lstm(x)  # h shape: (1, batch, hidden_dim)
        out = self.fc(h.squeeze(0))
        return out

input_dim = X_data.shape[1]
model = LSTMmodel_class(input_dim=input_dim)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
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
        correct += (pred.argmax(dim=1) == yb).sum().item()
    acc = 100 * correct / len(train_dataset)
    print(f"Epoch {epoch+1} - Loss: {total_loss:.4f} - Accuracy: {acc:.2f}%")

# save model
torch.save(model.state_dict(), dynamic_saved_model_path)
# torch.save(model.state_dict(), r"lstm_model.pth")
print("Model saved to lstm_model.pth")

# prediction function for sequences
def get_user_input():
    print("\nEnter student details to predict Pass/Fail:")
    gender = input("Gender (Male/Female): ").strip().title()
    age = int(float(input("Age: ")))
    hours = int(input("Study Hours per Week: "))
    attendance = int(input("Attendance Rate (0-100): "))
    internet = input("Internet Access (Yes/No): ").strip().title()
    grade = input("Previous Grade (A/B/C/D/F): ").strip().title()
    extra_cur = input("Extra Curricular (Yes/No): ").strip().title()

    user_df = pd.DataFrame([{
        'Gender': gender,
        'Age': age,
        'Study_Hours_per_Week': hours,
        'Attendance_Rate': attendance,
        'Internet_Access': internet,
        'Previous_Grade': grade,
        'Extra_Curricular': extra_cur
    }])

    for col in categorical_cols:
        if user_df.at[0, col] not in label_encoders[col].classes_:
            raise ValueError(f"Input '{user_df.at[0, col]}' not in training categories for '{col}'")
        user_df[col] = label_encoders[col].transform(user_df[col])
    user_df[numeric_cols] = scaler.transform(user_df[numeric_cols])

    # Repeat features seq_len times for LSTM input
    x_seq = np.tile(user_df.values.astype(np.float32), (seq_len, 1))
    return torch.tensor(x_seq).unsqueeze(0)  # add batch dimension

# start here

model.eval()
with torch.no_grad():
    user_input = get_user_input()
    output = model(user_input)
    pred = torch.argmax(output, dim=1).item()
    print("\n Prediction: Student will", "PASS " if pred == 1 else "FAIL ")
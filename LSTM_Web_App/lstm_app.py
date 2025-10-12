from flask import render_template, request, Blueprint
import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

# Define blueprint
lstm_bp = Blueprint("lstm",
                    __name__,
                    template_folder="templates",
                    static_folder = "static",
                    static_url_path = "/static"
)

# Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
dy_csv_path = os.path.join(script_dir, "student_performance_dataset.csv")
dy_saved_model_path = os.path.join(script_dir, "lstm_model.pth")

# Load CSV
df = pd.read_csv(dy_csv_path)
df['Final_Result'] = df['Final_Result'].map({'Pass': 1, 'Fail': 0})

categorical_cols = ['Gender', 'Internet_Access', 'Previous_Grade', 'Extra_Curricular']
numeric_cols = ['Age', 'Study_Hours_per_Week', 'Attendance_Rate']

# Encoders & scaler
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Model
class LSTMmodel_class(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, output_dim=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        out = self.fc(h.squeeze(0))
        return out

# Load trained model
input_dim = df.drop(['Student_ID', 'Final_Result', 'Family_Income'], axis=1).shape[1]
model = LSTMmodel_class(input_dim=input_dim)
model.load_state_dict(torch.load(dy_saved_model_path))
model.eval()
seq_len = 5

# Blueprint route
@lstm_bp.route("/lstm_index", methods=["GET", "POST"])
def lstm_index():
    prediction = None
    if request.method == "POST":
        try:
            gender = request.form.get('gender')
            age = int(float(request.form.get('age')))
            hours = int(request.form.get('hours'))
            attendance = int(request.form.get('attendance'))
            net = request.form.get('net')
            grade = request.form.get('grade')
            extra = request.form.get('extra')

            user_data = {
                'Gender': gender,
                'Age': age,
                'Study_Hours_per_Week': hours,
                'Attendance_Rate': attendance,
                'Internet_Access': net,
                'Previous_Grade': grade,
                'Extra_Curricular': extra
            }
            user_df = pd.DataFrame([user_data])
            display_data = {
                "Gender": gender,
                "Age": age,
                "Study Hours": hours,
                "Attendance": attendance,
                "Internet": net,
                "Grade": grade,
                "Extra Curricular": extra
            }

            for col in categorical_cols:
                user_df[col] = label_encoders[col].transform(user_df[col])
            user_df[numeric_cols] = scaler.transform(user_df[numeric_cols])

            x_seq = np.tile(user_df.values.astype(np.float32), (seq_len, 1))
            x_tensor = torch.tensor(x_seq).unsqueeze(0)

            with torch.no_grad():
                output = model(x_tensor)
                pred = torch.argmax(output, dim=1).item()
                prediction = "PASS" if pred == 1 else "FAIL"

        except Exception as e:
            prediction = f"Error: {str(e)}"

        return render_template("result.html", data= display_data, prediction=prediction)

    return render_template("lstm_index.html", prediction=None)

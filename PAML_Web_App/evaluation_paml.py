import torch
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from PAML_model import Recommender, extract_features
from sklearn.metrics import root_mean_squared_error
import os

#  paths
script_dir = os.path.dirname(os.path.abspath(__file__))
dynamic_csv_path = os.path.join(script_dir, "paml_study_data_2.csv")
dynamic_saved_model_path = os.path.join(script_dir, "paml_trained_model.pth")

# load csv
df = pd.read_csv(dynamic_csv_path)

# meta-tasks from students (student_id wise)
tasks = []
for student_id, group in df.groupby("student_id"):
    if len(group) < 3:
        continue  # skip students with too little data
    features = [extract_features(row) for _, row in group.iterrows()]
    labels = [float(row["study_hours"]) for _, row in group.iterrows()]
    x = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)
    tasks.append((x, y))

print(f"Total student tasks: {len(tasks)}")

# split into train and test tasks
train_tasks, test_tasks = train_test_split(tasks, test_size=0.2, random_state=42)
print(f"Train tasks: {len(train_tasks)} | Test tasks: {len(test_tasks)}")

# load trained PAML model

model = Recommender()
# model.load_state_dict(torch.load(r"PAML_Web_App\paml_trained_model.pth"))
# model.eval()

ckpt = torch.load(dynamic_saved_model_path, map_location="cpu")

loss_history  = ckpt["loss_history"]
state_dict    = ckpt["model_state"]

model.load_state_dict(state_dict)
model.eval()

# evaluate on test tasks using few shot adaptation
def evaluate_model(meta_model, test_tasks):
    all_preds = []
    all_labels = []

    for x_all, y_all in test_tasks:
        # normalize features
        x_all = (x_all - x_all.mean(0)) / (x_all.std(0) + 1e-8)

        if len(x_all) < 3:
            continue  # just to be extra safe

        support_x, support_y = x_all[:1], y_all[:1]
        query_x, query_y = x_all[1:], y_all[1:]

        # Fast adaptation
        fast_model = Recommender()
        fast_model.load_state_dict(meta_model.state_dict())
        optimizer = torch.optim.Adam(fast_model.parameters(), lr=0.005)
        loss_fn = torch.nn.MSELoss()

        for z in range(3):
            pred = fast_model(support_x).view(-1)
            loss = loss_fn(pred, support_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Predict on query
        fast_model.eval()
        with torch.no_grad():
            pred_query = fast_model(query_x).view(-1).numpy()
            true_query = query_y.view(-1).numpy()
            all_preds.extend(pred_query)
            all_labels.extend(true_query)

    mae = mean_absolute_error(all_labels, all_preds)
    rmse = root_mean_squared_error(all_labels, all_preds)

    print("\n Evaluation Results on Test Students:")
    print(f" MAE (Mean Absolute Error): {mae:.4f}")
    print(f" RMSE (Root Mean Squared Error): {rmse:.4f}")


evaluate_model(model, test_tasks)
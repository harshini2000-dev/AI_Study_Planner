# PAML model
'''
PAML model - student performance and study hours predictor works for both new and old students.
Based on past performance, time spent, confidence level, struggle level, sentiment analysis (nltk),
exam date the study hours are predicted.

Based on predictions & factors, we create a personalised study plans - studyplans_generation.py
'''

import math
from PAML_Web_App.nltk_sentiment import extract_negative_keywords

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import random
from datetime import date,datetime, timedelta
import os
import re


script_dir = os.path.dirname(os.path.abspath(__file__))
dy_csv_train_path = os.path.join(script_dir, "paml_study_data_2.csv")
dy_csv_pred_path = os.path.join(script_dir, "paml_predictions_log.csv")
dy_saved_model_path = os.path.join(script_dir, "paml_trained_model.pth")

class Recommender(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(7, 1)  # 7 features
    def forward(self, x):
        out = self.fc(x).squeeze()
        return F.relu(out) + 0.5 

    # def __init__(self):
    #     super().__init__()
    #     self.fc1 = nn.Linear(7, 16) # 7 features
    #     self.fc2 = nn.Linear(16, 1)

    # def forward(self, x):
    #     x = F.relu(self.fc1(x))
    #     out = self.fc2(x).squeeze()
    #     return F.relu(out) + 0.75 

#  helper functions 
default_days = 30
def days_until_exam(exam_date_input):
    if not exam_date_input or pd.isna(exam_date_input):
        return default_days

    # If it’s already a datetime object, use it directly
    if isinstance(exam_date_input, (datetime, date)):
        exam_date = exam_date_input
    else:
        try:
            exam_date = datetime.strptime(exam_date_input, "%Y-%m-%d")
        except Exception:
            return default_days

    today = datetime.today()
    return max(min((exam_date - today).days, 30), 0)

def extract_features(row):
    def safe_float(val, default=0.0):
        try:
            return float(val)
        except:
            return default

    confidence_score = safe_float(row.get('confidence_rating'), 2.5) / 5.0
    sentiment_score = safe_float(row.get('sentiment_score'), 0.0)
    topic_difficulty = safe_float(row.get('topic_difficulty'), 5) / 10.0
    time_spent = min(safe_float(row.get('time_spent_minutes'), 15), 180) / 180.0
    is_struggling = safe_float(row.get('is_struggling'), 0.5)
    days_left = days_until_exam(row.get('exam_date', '')) / 30.0
    exam_known_flag = 0 if not row.get('exam_date') else 1

    exam_result_flag = 0.5
    if 'last_exam_result' in row:
        result = str(row['last_exam_result']).lower()
        exam_result_flag = 1 if result == 'passed' else 0 if result == 'failed' else 0.5

    return [confidence_score, sentiment_score, topic_difficulty,
            time_spent, days_left, exam_known_flag, exam_result_flag]

# load csv
def load_tasks(csv_path):
    import pandas as pd
    df = pd.read_csv(csv_path)
    df['user_input'] = df['user_input'].str.replace(',', ' ').str.replace('/', '').str.replace("'", '').str.replace('"','')
    #df['exam_date'] = pd.to_datetime(df['exam_date']).dt.strftime('%Y-%m-%d')

    tasks = []
    for student_id, group in df.groupby('student_id'):

        features = [extract_features(row) for n, row in group.iterrows()]
        labels = [max(0.1, float(row['study_hours'])) for n, row in group.iterrows()]

        # Use 3 as minimum (2 support + 1 query)
        if len(features) >= 3:
            features_tensor = torch.tensor(features, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.float32)
            tasks.append((features_tensor, labels_tensor))
    
    print(f" Prepared {len(tasks)} meta-learning tasks.")
    return tasks

# Meta Training Loop
def train_paml(tasks, inner_steps=5, outer_lr=0.01):
    meta_model = Recommender()
    outer_optimizer = torch.optim.Adam(meta_model.parameters(), lr=outer_lr)
    loss_fn = nn.MSELoss()

    loss_history = []     

    for epoch in range(300):
        meta_loss = 0.0
        task_count = 0  # avoid division by zero

        for _ in range(4):
            x_all, y_all = random.choice(tasks)

            # Normalize inputs 
            x_all = (x_all - x_all.mean(dim=0)) / (x_all.std(dim=0) + 1e-8)  # add epsilon to prevent /0

            # split support/query
            idx = list(range(len(x_all)))
            random.shuffle(idx)
            if len(idx) < 4:
                continue
            support_idx, query_idx = idx[:1], idx[1:]
            x_support, y_support = x_all[support_idx], y_all[support_idx]
            x_query, y_query = x_all[query_idx], y_all[query_idx]

            # clone model
            fast_model = Recommender()
            fast_model.load_state_dict(meta_model.state_dict())
            inner_optimizer = torch.optim.Adam(fast_model.parameters(), lr=0.001)  # Safer optimizer

            for _ in range(inner_steps):
                pred = fast_model(x_support).view(-1)

                #Check for NaN in support prediction
                if torch.isnan(pred).any():
                    print(" NaN in support prediction — skipping task")
                    continue

                loss = loss_fn(pred, y_support.view(-1))

                if torch.isnan(loss):
                    print(" NaN in support loss — skipping task")
                    continue

                inner_optimizer.zero_grad()
                loss.backward()
                inner_optimizer.step()

            # Query prediction
            query_pred = fast_model(x_query).view(-1)
            if torch.isnan(query_pred).any():
                print("NaN in query prediction - skipping task")
                continue

            query_loss = loss_fn(query_pred, y_query.view(-1))
            
            if torch.isnan(query_loss):
                print("NaN in query loss - skipping task")
                continue

            outer_optimizer.zero_grad()
            query_loss.backward()

            for meta_param, fast_param in zip(meta_model.parameters(), fast_model.parameters()):
                if meta_param.grad is None:
                    meta_param.grad = fast_param.grad
                else:
                    meta_param.grad += fast_param.grad

            outer_optimizer.step()
            meta_loss += query_loss.item()
            task_count += 1

        avg_loss = (meta_loss / task_count) if task_count else float("nan")
        loss_history.append(avg_loss)
        # Only divide if at least one valid task was used
        if epoch % 50 == 0:
            if task_count > 0:
                print(f"[Epoch {epoch}] Avg Meta Loss: {meta_loss / task_count:.4f}")
            else:
                print(f"[Epoch {epoch}] No valid tasks. Meta loss: NaN")    

    #torch.save(meta_model.state_dict(), r"PAML_Web_App\paml_trained_model.pth")

    torch.save(
        {
        "model_state": meta_model.state_dict(),
        "loss_history": loss_history },
        dy_saved_model_path
        )

    print("model saved")
    return meta_model

# predict and adapt
def adapt_and_predict(meta_model, student_id, student_row, full_csv_path):
    df = pd.read_csv(full_csv_path)
    student_rows = df[df['student_id'] == student_id].dropna()

    if student_rows.shape[0] >= 2: # old students
        print(f"\n Found past data for {student_id}. Performing fast adaptation...")
        support_x = torch.tensor([extract_features(r) for _, r in student_rows.iterrows()], dtype=torch.float32)
        support_y = torch.tensor([float(r['study_hours']) for _, r in student_rows.iterrows()], dtype=torch.float32)
    #optimizer = torch.optim.SGD(fast_model.parameters(), lr=0.05)

        fast_model = Recommender()
        fast_model.load_state_dict(meta_model.state_dict())
        
        optimizer = torch.optim.Adam(fast_model.parameters(), lr=0.005)
        loss_fn = nn.MSELoss()
        for z in range(3):
            pred = fast_model(support_x).view(-1)
            loss = loss_fn(pred, support_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        fast_model.eval()
        with torch.no_grad():
            x = torch.tensor([extract_features(student_row)], dtype=torch.float32)
            return fast_model(x).item()
    else:
        msg = ""
        if student_rows.shape[0] == 0:
            msg = "New student detected : "
        else:
            msg = "New student with just 1 entry of past data found : "
        print(f"\n {msg} {student_id}. Using base model prediction.")
        return predict_study_hours(meta_model, student_row)

# base Prediction for new students with 0 or 1 past data entries
def predict_study_hours(model, student_row):
    model.eval()
    with torch.no_grad():
        x = torch.tensor([extract_features(student_row)], dtype=torch.float32)
        return model(x).item()

# logging csv functions
def log_prediction_to_csv(student_id, student_row, predicted_hours, filename=dy_csv_pred_path):
    ordered_fields = [
        'student_id', 'user_input', 'confidence_score', 'sentiment_score',
        'struggle_keywords', 'topic', 'topic_difficulty', 'time_spent_minutes',
        'is_struggling', 'days_left', 'predicted_study_hours', 'last_exam_result'
    ]

    row_dict = {**student_row}
    row_dict['student_id'] = student_id
    row_dict['predicted_study_hours'] = round(predicted_hours, 2)
    row_dict['days_left'] = days_until_exam(student_row.get('exam_date', '')) # store normalized values for better training


    for field in ordered_fields:
        row_dict.setdefault(field, '')
    df = pd.DataFrame([{k: row_dict[k] for k in ordered_fields}])
    if os.path.exists(filename):
        df.to_csv(filename, mode='a', header=False, index=False)
    else:
        df.to_csv(filename, index=False)


def append_to_training_csv(student_id, student_row, predicted_hours, csv_path=dy_csv_train_path):
    ordered_fields = [
        'student_id', 'user_input', 'confidence_score', 'sentiment_score',
        'struggle_keywords', 'topic', 'topic_difficulty', 'time_spent_minutes',
        'is_struggling', 'days_left', 'study_hours', 'last_exam_result'
    ]
    
    # start with a copy to avoid modifying original
    row_dict = student_row.copy()  
    
    # add or overwrite required fields
    row_dict['student_id'] = student_id
    row_dict['study_hours'] = round(predicted_hours, 2)
    row_dict['days_left'] = days_until_exam(student_row.get('exam_date', '')) # store normalized values for better training
 
    # fill missing fields with empty string for consistent columns
    for field in ordered_fields:
        if field not in row_dict:
            row_dict[field] = ''
    
    # create dataframe with correct column order
    df = pd.DataFrame([{k: row_dict[k] for k in ordered_fields}])
    
    # append to csv and write header only if file does not exist
    df.to_csv(csv_path, mode='a', header=not os.path.exists(csv_path), index=False)

def get_student_inputs():
    print("\n Please answer the following questions: \n")

    while True:
        student_id = input(" Enter Student ID: (e.g., S0001):").strip()
        if re.fullmatch(r"S\d+", student_id):
            break
        else:
            print(" Invalid Student ID. It must start with 'S' followed by digits. Please try again.")

    topic = input(" What topic are you currently studying ? (e.g., algebra, probability): ").strip().lower() or "unknown"
    
    user_input = input(" How do you feel about this topic ? Describe any challenges, strengths or concerns in few words:\n> ").strip().replace(',', ' ').replace('/', '').replace("'", '').replace('"','')

    # student confidence rating (0–5 scaled to 0–1 later)
    try:
        student_conf = float(input(" On a scale of 0 to 5, rate your confidence level (0 = low, 5 = high) : ").strip())
        confidence_score = max(0.0, min(1.0, student_conf / 5.0))
    except:
        confidence_score = 0.5  # setting medium confidence as default
    
    # cal sentiment analysis on student statements
    sentiment_results = extract_negative_keywords(user_input)
    sentiment_score = sentiment_results["overall_sentiment"]
    struggle_keywords = len(sentiment_results["negative_keywords"])
    print(sentiment_score)
    print(struggle_keywords)

    # topic_difficulty level
    try:
        topic_difficulty = int(input(" On a scale of 1 to 10, how difficult is this topic for you ? (1 = easy, 10 = very hard): ").strip())
        topic_difficulty = max(1, min(10, topic_difficulty))
    except:
        topic_difficulty = 5 # medium diff as default
    
    try:
        time_spent = int(input(" How many minutes spent on this topic per session ? (e.g., 30): ").strip())
        time_spent = max(0, time_spent)
    except:
        time_spent = 15
        print("Default time set as 15 mins")

    # exam date
    exam_date_input = input(" Enter your exam date (YYYY-MM-DD) or press Enter if not known: ").strip()
    try:
        exam_date = datetime.strptime(exam_date_input, "%Y-%m-%d").strftime("%Y-%m-%d")
    except:
        exam_date = ""

    last_exam_result = input(" What was your last exam result ? (passed / failed): ").strip().lower()
    if last_exam_result not in ["passed", "failed"]:
        last_exam_result = "unknown"
        print("last exam result set as unknown")
    
    is_struggling = 1 if struggle_keywords or confidence_score < 0.3 or sentiment_score < -0.3 else 0

    student_data =  {
        "student_id": student_id,
        "user_input": user_input,
        "confidence_score": confidence_score,
        "sentiment_score": sentiment_score,
        "struggle_keywords": struggle_keywords,
        "topic": topic,
        "topic_difficulty": topic_difficulty,
        "time_spent_minutes": time_spent,
        "is_struggling": is_struggling,
        "exam_date": exam_date,
        "last_exam_result": last_exam_result
    }

    return student_data


# start here
if __name__ == "__main__":
    csv_path = dy_csv_train_path
    tasks = load_tasks(csv_path)
    model = train_paml(tasks)

    student_data = get_student_inputs()

    student_id = student_data["student_id"]
    prediction = adapt_and_predict(model, student_id, student_data, csv_path)
    # prediction = float(math.ceil(prediction))
    print(f"\n PAML predicted study hours per day: {prediction:.2f}")

    if student_data['exam_date'] == '':
        future = str(date.today() + timedelta(days = default_days))
        student_data['exam_date'] = datetime.strptime(future, "%Y-%m-%d")


    log_prediction_to_csv(student_id, student_data, prediction)
    append_to_training_csv(student_id, student_data, prediction, csv_path)

    

# new_student = {
    #     student_id = "S002",
    #     "user_input": "I’m confused about calculus",
    #     "confidence_score": 0.2,
    #     "sentiment_score": -0.6,
    #     "struggle_keywords": 1,
    #     "topic": "calculus",
    #     "topic_difficulty": 7,
    #     "time_spent_minutes": 10,
    #     "is_struggling": 1,
    #     "exam_date": "2025-07-20",
    #     "last_exam_result": "failed"
    # }
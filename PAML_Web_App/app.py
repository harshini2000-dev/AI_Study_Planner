from flask import Flask, render_template, request, Blueprint
import torch
from PAML_Web_App import PAML_model, studyplans_generation as sg, nltk_sentiment as ns
# from PAML_Web_App.nltk_sentiment import extract_negative_keywords
import os
#
# app = Flask(__name__)

# blueprint
paml_bp = Blueprint("paml",
                    __name__,
                    template_folder="templates",
                    static_folder = "static",
                    static_url_path = "/static"
)


# Load trained model

script_dir = os.path.dirname(os.path.abspath(__file__))
dy_csv_path = os.path.join(script_dir, "paml_study_data_2.csv")
dy_saved_model_path = os.path.join(script_dir, "paml_trained_model.pth")


model_data = torch.load(dy_saved_model_path, map_location=torch.device('cpu'))
model = PAML_model.Recommender()
model.load_state_dict(model_data['model_state'])
model.eval()

csv_path = dy_csv_path

@paml_bp.route("/paml_index", methods=["GET", "POST"])
def paml_index():
    predicted_hours = None

    if request.method == "POST":
        student_data = {
            "student_id": request.form["student_id"].strip(),
            "user_input": request.form["user_input"].strip().lower(),
            "confidence_score": float(request.form["confidence_score"]) / 5.0,
            "topic": request.form["topic"].strip().lower(),
            "topic_difficulty": int(request.form["topic_difficulty"]),
            "time_spent_minutes": int(request.form["time_spent_minutes"]),
            "exam_date": str(request.form["exam_date"]),
            "last_exam_result": request.form["last_exam_result"],
            "days_left": PAML_model.days_until_exam(str(request.form["exam_date"]))
        }

        sentiment = ns.extract_negative_keywords(student_data["user_input"])
        student_data["sentiment_score"] = sentiment["overall_sentiment"]
        student_data["struggle_keywords"] = len(sentiment["negative_keywords"])
        student_data["is_struggling"] = 1 if student_data["struggle_keywords"] or student_data["confidence_score"] < 0.3 or student_data["sentiment_score"] < -0.3 else 0

        predicted_hours = PAML_model.adapt_and_predict(model, student_data["student_id"], student_data, csv_path)

        PAML_model.log_prediction_to_csv(student_data["student_id"], student_data, predicted_hours)
        PAML_model.append_to_training_csv(student_data["student_id"], student_data, predicted_hours)

        display_data1 = {
            "User Comments" : student_data["user_input"],
            "Topic" : student_data["topic"],
            "Topic difficulty level": "Easy" if student_data["topic_difficulty"] < 5 else ("Hard" if student_data["topic_difficulty"] > 5 else "Medium"),
            "Time spent (mins)" : student_data["time_spent_minutes"],
            "Exam date" : student_data["exam_date"],
            "Days left for exam" : student_data["days_left"],
            "Is struggling" : "Yes" if student_data["is_struggling"]  == 1 else "No",
            "Last Exam result": str(student_data["last_exam_result"]).capitalize()
        }

        return render_template("result_2.html", prediction=predicted_hours, student_id=student_data["student_id"], display_data = display_data1)
    
    return render_template("form.html")
        #return render_template("form.html", prediction=predicted_hours)


@paml_bp.route("/create_plan", methods=["POST"])
def create_plan():

    student_data = {
        "topic": request.form["topic"],
        "days_left": request.form["days_left"]
    }

    predicted_hours = float(request.form.get("prediction"))

    content = sg.create_study_plan(round(predicted_hours,1), student_data)
    msg = sg.download_sp(content)
    return render_template("study_plan.html", plan_text= content, message = msg)
    #return render_template("study_plan.html", plan=study_plan, student_id=student_data["student_id"], data = student_data)

# if __name__ == "__main__":
#     app.run(debug=True)

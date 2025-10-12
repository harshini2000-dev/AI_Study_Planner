from flask import Flask, render_template
from LSTM_Web_App.lstm_app import  lstm_bp
from PAML_Web_App.app import paml_bp

app = Flask(__name__)
app.register_blueprint(lstm_bp, url_prefix="/lstm")
app.register_blueprint(paml_bp, url_prefix="/paml")

@app.route("/")
def home():
    return render_template("home.html")

if __name__ == "__main__":
    app.run(debug=True)

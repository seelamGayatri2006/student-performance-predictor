from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "student_model.pkl")

def load_model():
    data = joblib.load(MODEL_PATH)
    return data["model"], data["scaler"], data["features"]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        model, scaler, features = load_model()

        values = [
            float(data["study_hours"]),
            float(data["attendance"]),
            float(data["previous_marks"]),
            float(data["sleep_hours"]),
            float(data["internet_hours"]),
            int(data["family_support"]),
            int(data["extra_activities"]),
        ]

        row = np.array([values])
        row_scaled = scaler.transform(row)
        score = float(np.clip(model.predict(row_scaled)[0], 0, 100))
        score = round(score, 1)

        if score >= 75:
            grade = "A"
            grade_label = "Excellent"
            color = "#16A77E"
        elif score >= 60:
            grade = "B"
            grade_label = "Good"
            color = "#2D6BE4"
        elif score >= 45:
            grade = "C"
            grade_label = "Average"
            color = "#E4A82D"
        else:
            grade = "D"
            grade_label = "Needs Improvement"
            color = "#E44B2D"

        tips = []
        if float(data["study_hours"]) < 4:
            tips.append("Try to study at least 4–5 hours daily to improve your score.")
        if float(data["attendance"]) < 75:
            tips.append("Attendance below 75% significantly impacts performance.")
        if float(data["internet_hours"]) > 5:
            tips.append("Reducing internet/social media time can boost focus.")
        if float(data["sleep_hours"]) < 6:
            tips.append("Getting 7–8 hours of sleep improves memory and concentration.")
        if int(data["extra_activities"]) == 0:
            tips.append("Joining extra activities can improve discipline and score.")

        return jsonify({
            "score": score,
            "grade": grade,
            "grade_label": grade_label,
            "color": color,
            "tips": tips,
            "success": True
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, port=5000)

# Student Performance Predictor — Web App

A machine learning web app to predict student exam scores based on study habits.

## 🚀 Features
- Predicts score from 7 behavioral features
- Grade classification (A–D) with improvement tips
- Interactive sliders on Flask web interface
- 4 dynamic charts

## 🛠 Tech Stack
Python, Scikit-learn, Flask, HTML5, CSS3, JavaScript

## 📊 Model Results
| Model | R² Score | RMSE |
|---|---|---|
| Linear Regression | 0.8660 | 5.14 |
| Random Forest | 0.7392 | 7.17 |
| Decision Tree | 0.4396 | 10.51 |

## How to Run

### Step 1 — Install dependencies
Open terminal in this folder and run:
```
pip install -r requirements.txt
```

### Step 2 — Start the server
```
python app.py
```

### Step 3 — Open in browser
Open your browser and go to:
```
http://localhost:5000
```

That's it! Fill in the sliders and click "Predict My Score".

## Files
- `app.py` — Flask backend with ML prediction logic
- `templates/index.html` — The website (frontend)
- `models/student_model.pkl` — Trained ML model
- `requirements.txt` — Python dependencies

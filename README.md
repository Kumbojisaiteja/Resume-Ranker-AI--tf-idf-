## Resume Ranker (TF-IDF + ML)

A Machine Learning-based web application that analyzes resumes against job descriptions and predicts how well a candidate fits the role.



## 🌐Live Demo
https://resume-ranker-ai-tf-idf.onrender.com

---

## 📌 Features

- Upload multiple resumes (PDF, DOCX, TXT)
- Compare resumes with job descriptions
- TF-IDF based similarity scoring
- Machine Learning prediction (XGBoost)
- Skill extraction & matching
- Missing skill suggestions
- Job role recommendations

---

## 🧠 How It Works

1. Text is extracted from resume & job description
2. Data is cleaned using NLP preprocessing
3. TF-IDF vectors are generated
4. Features include:
   - Cosine similarity
   - Skill match count
   - Resume length
   - Keyword overlap
5. Model predicts fit:
   - Good Fit
   - Potential Fit
   - No Fit

---

##  Tech Stack

- Python
- Flask
- Scikit-learn
- XGBoost
- TF-IDF (NLP)
- HTML, CSS

---

## 📁 Project Structure
RESUME-RANKER-TF-IDF/
│
├── app.py
├── requirements.txt
├── Procfile
│
├── model/
│ ├── best_model-tf-idf.pkl
│ ├── best_scaler-tf-idf.pkl
│ └── best_tfidf-tf-idf.pkl
│
├── templates/
│ ├── index.html
│ ├── prediction.html
│ └── result.html
│
├── static/
│ ├── style.css
│ └── images/
│
└── dataset/

##  Run Locally

pip install -r requirements.txt
python app.py



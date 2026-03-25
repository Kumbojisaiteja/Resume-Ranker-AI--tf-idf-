# IMPORTS
from flask import Flask, render_template, request
import pickle
import numpy as np
import re
import PyPDF2
from docx import Document
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.secret_key = "secret123"

# LOAD MODELS (UPDATED NAMES)
model = pickle.load(open('model/best_model-tf-idf.pkl', 'rb'))
scaler = pickle.load(open('model/best_scaler-tf-idf.pkl', 'rb'))
tfidf = pickle.load(open('model/best_tfidf-tf-idf.pkl', 'rb'))

# SKILLS
skills = [
    'python','sql','java','javascript','aws','docker','kubernetes',
    'machine learning','deep learning','nlp','react','angular',
    'spark','pytorch','tensorflow','git','linux','hadoop','azure',
    'html','css','django','flask','spring','api','excel','tableau','power bi'
]

# JOB ROLE MAP
JOB_ROLES = {
    "Data Scientist": ["python","machine learning","deep learning","nlp","statistics"],
    "ML Engineer": ["python","tensorflow","pytorch","machine learning","docker","aws"],
    "Backend Developer": ["java","python","sql","spring","django","api"],
    "Frontend Developer": ["javascript","react","angular","html","css"],
    "Data Analyst": ["sql","excel","python","tableau","power bi"],
    "Cloud Engineer": ["aws","azure","docker","kubernetes","linux"]
}

# CLEAN TEXT
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

# PDF
def extract_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

# DOCX
def extract_docx(file):
    doc = Document(file)
    text = ""
    for para in doc.paragraphs:
        text += para.text + " "
    return text

# SKILL EXTRACTION
def extract_skills(text):
    return [s for s in skills if s in text]

# JOB RECOMMENDATION
def recommend_jobs(resume_skills):
    recommendations = []

    for role, role_skills in JOB_ROLES.items():
        match_count = len(set(resume_skills) & set(role_skills))
        total = len(role_skills)
        score = int((match_count / total) * 100)

        if score > 30:
            recommendations.append((role, score))

    recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)
    return recommendations[:3]

# PREDICTION (UPDATED — NO SBERT)
def predict(resume, jd):

    # CLEAN
    res_clean = clean_text(resume)
    jd_clean = clean_text(jd)

    # TF-IDF VECTORS
    res_vec = tfidf.transform([res_clean]).toarray()
    jd_vec = tfidf.transform([jd_clean]).toarray()

    # SIMILARITY
    tfidf_score = cosine_similarity(res_vec, jd_vec)[0][0]

    # SKILLS
    res_skills = extract_skills(res_clean)
    jd_skills = extract_skills(jd_clean)

    skill_match = len(set(res_skills) & set(jd_skills))

    # EXTRA FEATURES
    res_len = len(res_clean.split())
    length_ratio = res_len / (len(jd_clean.split()) + 1)

    jd_skill_count = len(jd_skills)
    res_skill_count = len(res_skills)

    # FINAL FEATURE VECTOR (MATCH TRAINING)
    features = np.hstack((
        jd_vec,
        res_vec,
        [[
            tfidf_score,
            res_len,
            length_ratio,
            skill_match,
            jd_skill_count,
            res_skill_count
        ]]
    ))

    # SCALE
    features = scaler.transform(features)

    # PREDICT
    prob = model.predict_proba(features)[0][1]
    score = int(prob * 100)

    # LABEL
    if score > 75:
        label = "Good Fit"
    elif score > 50:
        label = "Potential Fit"
    else:
        label = "No Fit"

    # EXPLANATION
    missing_skills = list(set(jd_skills) - set(res_skills))
    explanation = "Missing skills: " + ", ".join(missing_skills[:5])

    return score, label, res_skills, jd_skills, explanation, missing_skills

# ROUTES

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/prediction')
def prediction():
    return render_template('prediction.html')

@app.route('/predict', methods=['POST'])
def predict_page():

    jd = request.form['jd']
    files = request.files.getlist('resumes')

    results = []

    for file in files:
        filename = file.filename.lower()

        # FILE HANDLING
        if filename.endswith('.pdf'):
            resume = extract_pdf(file)
        elif filename.endswith('.docx'):
            resume = extract_docx(file)
        elif filename.endswith('.txt'):
            resume = file.read().decode('utf-8', errors='ignore')
        else:
            resume = ""

        score, label, res_skills, jd_skills, explanation, missing = predict(resume, jd)

        jobs = recommend_jobs(res_skills)
        suggestions = [f"Add skill: {m}" for m in missing[:3]]

        results.append({
            "name": file.filename,
            "score": score,
            "label": label,
            "res_skills": res_skills,
            "jd_skills": jd_skills,
            "explanation": explanation,
            "jobs": jobs,
            "suggestions": suggestions
        })

    results = sorted(results, key=lambda x: x['score'], reverse=True)

    return render_template('result.html', results=results)

# RUN
if __name__ == '__main__':
    app.run(debug=True)
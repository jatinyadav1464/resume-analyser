from flask import Flask, render_template, request, redirect, flash
import os
import fitz  # PyMuPDF
import spacy
import re
import phonenumbers
import nltk
from collections import Counter
from nltk.corpus import stopwords
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.secret_key = "secret_key"  # Needed for flash messages

# Database Configuration (SQLite)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///contacts.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Create Database Model
class Contact(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), nullable=False)
    phone = db.Column(db.String(20), nullable=False)
    message = db.Column(db.Text, nullable=False)

# Ensure database tables exist
with app.app_context():
    db.create_all()

# Ensure necessary directories exist
UPLOAD_FOLDER = 'uploads'
RESUME_FOLDER = 'resumes'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESUME_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Load stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Predefined skill set
predefined_skills = {
    "Python", "JavaScript", "ReactJS", "Machine Learning", "Deep Learning",
    "HTML", "CSS", "SQL", "Angular", "Java", "C++", "Docker", "AWS",
    "Flask", "Django", "Data Science", "AI", "Bootstrap"
}

# Job Recommendations based on skills
job_roles = {
    "Python": ["Python Developer", "Data Scientist", "AI Engineer"],
    "JavaScript": ["Frontend Developer", "Full Stack Developer"],
    "Machine Learning": ["ML Engineer", "Data Scientist"],
    "ReactJS": ["React Developer", "Frontend Engineer"],
    "AWS": ["Cloud Engineer", "DevOps Engineer"],
    "SQL": ["Database Administrator", "Data Analyst"],
    "AI": ["AI Engineer", "Machine Learning Engineer"]
}

# Courses & Certifications based on missing skills
course_recommendations = {
    "Python": "Python for Everybody - Coursera",
    "Machine Learning": "Machine Learning by Andrew Ng - Coursera",
    "Deep Learning": "Deep Learning Specialization - Coursera",
    "ReactJS": "React - The Complete Guide - Udemy",
    "JavaScript": "JavaScript: Understanding the Weird Parts - Udemy",
    "SQL": "The Complete SQL Bootcamp - Udemy",
    "AWS": "AWS Certified Solutions Architect - Coursera",
    "Docker": "Docker Mastery - Udemy",
    "Data Science": "Data Science and Machine Learning Bootcamp - Udemy",
    "AI": "AI For Everyone - Coursera"
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        phone = request.form['phone']
        message = request.form['message']

        # Save to database
        new_contact = Contact(name=name, email=email, phone=phone, message=message)
        db.session.add(new_contact)
        db.session.commit()

        flash("Your message has been sent successfully!", "success")
        return redirect('/contact')

    return render_template('contact.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)
    
    text = extract_text_from_pdf(file_path)
    name = extract_name(text)
    email = extract_email(text)
    phone = extract_phone(text)
    skills = extract_skills(text)

    # Calculate resume score (out of 100)
    score = min(len(skills), 10) * 10  # Max score: 100

    # Get job recommendations
    job_recommendations = recommend_jobs(skills)

    # Get additional suggested skills
    suggested_skills = suggest_additional_skills(skills, job_recommendations)

    # Recommend courses based on missing skills
    course_suggestions = recommend_courses(suggested_skills)

    return render_template('result.html', 
                           name=name, email=email, phone=phone, 
                           skills=", ".join(skills), score=f"{score}/100",
                           job_recommendations=job_recommendations, suggested_skills=suggested_skills,
                           course_suggestions=course_suggestions)

@app.route('/make_resume')
def make_resume():
    return render_template('make_resume.html')

@app.route('/generate_resume', methods=['POST'])
def generate_resume():
    name = request.form['name']
    email = request.form['email']
    phone = request.form['phone']
    skills = request.form['skills']
    experience = request.form['experience']
    education = request.form['education']

    resume_content = f"""
    Name: {name}
    Email: {email}
    Phone: {phone}

    Skills:
    {skills}

    Experience:
    {experience}

    Education:
    {education}
    """

    # Save resume to a text file
    resume_path = os.path.join(RESUME_FOLDER, f"{name.replace(' ', '_')}_resume.txt")
    with open(resume_path, "w") as file:
        file.write(resume_content)

    flash("Resume created successfully!", "success")
    return redirect('/make_resume')

# Utility functions
def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF and clean it."""
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text("text") + " "
        
        text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)
        text = re.sub(r'(?<=[a-zA-Z])(?=[0-9])', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
    except Exception as e:
        text = f"Error extracting text: {e}"
    return text

def extract_name(text):
    """Extract the most likely name from the text."""
    doc = nlp(text)
    names = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    return Counter(names).most_common(1)[0][0] if names else "Not Found"

def extract_email(text):
    """Extract email from text."""
    match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
    return match.group(0) if match else "Not Found"

def extract_phone(text):
    """Extract phone number from text."""
    matches = phonenumbers.PhoneNumberMatcher(text, "IN")
    for match in matches:
        return phonenumbers.format_number(match.number, phonenumbers.PhoneNumberFormat.INTERNATIONAL)
    return "Not Found"

def extract_skills(text):
    """Extract relevant skills from predefined skill set."""
    words = re.findall(r'\b\w+\b', text)
    words = [word.lower() for word in words if word.lower() not in stop_words]
    found_skills = {skill for skill in predefined_skills if skill.lower() in words}
    return found_skills if found_skills else {"Not Found"}

def recommend_jobs(skills):
    """Recommend jobs based on existing skills."""
    recommended_jobs = set()
    for skill in skills:
        recommended_jobs.update(job_roles.get(skill, []))
    return list(recommended_jobs) if recommended_jobs else ["No job recommendations available"]

def suggest_additional_skills(skills, job_recommendations):
    """Suggest skills based on job recommendations."""
    suggested_skills = set()
    for job in job_recommendations:
        for skill, jobs in job_roles.items():
            if job in jobs and skill not in skills:
                suggested_skills.add(skill)
    return list(suggested_skills) if suggested_skills else ["No additional skills suggested"]

def recommend_courses(suggested_skills):
    """Recommend courses based on missing skills."""
    return [course_recommendations[skill] for skill in suggested_skills if skill in course_recommendations]

if __name__ == '__main__':
    app.run(debug=True)
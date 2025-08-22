from flask import Flask, jsonify, request
from flask_cors import CORS
import pdfplumber
import os
import google.generativeai as genai
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
import json
import re

load_dotenv()

app = Flask(__name__)
CORS(app)

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = {'pdf'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return jsonify({"message": "Flask + ATSGemini API running"})

@app.route('/analyze-job', methods=['POST'])
def analyze_job_resume():
    job_description = request.form.get('job_description', '')

    if not job_description:
        return jsonify({"error": "Job description is required."}), 400

    if 'resume_file' not in request.files:
        return jsonify({"error": "No resume file part in the request"}), 400

    file = request.files['resume_file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filepath = None
    try:
        if not (file and allowed_file(file.filename)):
            return jsonify({"error": "Allowed file types are pdf"}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        with pdfplumber.open(filepath) as pdf:
            resume_text = "\n".join(page.extract_text() or "" for page in pdf.pages)

        resume_words = resume_text.split()
        resume_word_count = len(resume_words)

        model = genai.GenerativeModel('gemini-1.5-flash')
        chat = model.start_chat(history=[])

        prompt_content = f"""
        You are an advanced Applicant Tracking System (ATS) that evaluates resumes against job descriptions.

Analyze the resume below in the exact way an ATS would, focusing on keyword matching, skill relevance, and role alignment. 
Respond ONLY with a valid JSON object. DO NOT include any text or markdown formatting.

        ---
        Job Description:
        {job_description}

        ---
        Resume:
        {resume_text}

        ---
        Your JSON response MUST contain:
        - "overall_match_score": integer (0-100)
        - "key_strengths": list of 3-5 strings
        - "areas_for_improvement": list of 2-3 strings
        - "actionable_feedback": 1-2 sentences
        - "extracted_key_skills": list of 5-10 skills
        """

        response = chat.send_message(prompt_content)
        gemini_raw_output = response.text

        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', gemini_raw_output, re.DOTALL)
        if json_match:
            gemini_analysis_str = json_match.group(1)
        else:
            gemini_analysis_str = gemini_raw_output

        try:
            gemini_analysis = json.loads(gemini_analysis_str)
        except json.JSONDecodeError as e:
            return jsonify({
                "error": "AI returned invalid JSON",
                "details": str(e),
                "raw_output": gemini_raw_output
            }), 500

        return jsonify({
            "resume_extracted_text": resume_text,
            "resume_word_count": resume_word_count,
            "job_description_received": job_description,
            "gemini_analysis": gemini_analysis
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Failed to process file or call Gemini API: {str(e)}"}), 500
    finally:
        if filepath and os.path.exists(filepath):
            os.remove(filepath)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import os
import sys
import json
import requests
import logging
import traceback
from dotenv import load_dotenv
from ai_engine import AIEngine

# Load environment variables
load_dotenv()

# Configure logging
log_level = logging.DEBUG if os.getenv('DEBUG', 'False').lower() == 'true' else logging.INFO
logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Log deployment environment information
logger.info(f"Starting application in {os.getenv('FLASK_ENV', 'development')} mode")
logger.info(f"Python version: {sys.version}")
logger.info(f"Running on Render: {os.getenv('RENDER', 'false')}")

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'dev-secret-key')

# Configure error handling
@app.errorhandler(500)
def server_error(e):
    logger.error(f"Server error: {str(e)}\n{traceback.format_exc()}")
    return render_template('error.html', error="Internal server error. Please try again later."), 500

@app.errorhandler(404)
def not_found(e):
    return render_template('error.html', error="Page not found."), 404

# Initialize AI Engine
ai_engine = AIEngine()

# Log AI Engine status
if ai_engine.use_hf:
    model_preference = os.getenv('LLM_MODEL_PREFERENCE', 'deepseek')
    logger.info(f"Enhanced AI reports enabled with Hugging Face integration using {model_preference} model")
else:
    logger.info("Using standard report generation (Hugging Face not configured)")

# Load questions from JSON file
def load_questions():
    try:
        with open('questions.json', 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        # Default questions if file not found
        return {
            "personal": [
                {"id": "p1", "text": "你的年龄是？", "type": "select", "options": ["14-15岁", "16-17岁", "18-19岁"]},
                {"id": "p2", "text": "你目前就读的年级是？", "type": "select", "options": ["高一", "高二", "高三"]},
                {"id": "p3", "text": "你的性别是？", "type": "select", "options": ["男", "女", "不愿透露"]}
            ],
            "academic": [
                {"id": "a1", "text": "你目前最感兴趣的学科是什么？", "type": "text"},
                {"id": "a2", "text": "你认为你的学习风格是什么？", "type": "multiselect", "options": ["视觉学习者", "听觉学习者", "动手实践者", "阅读/写作学习者"]},
                {"id": "a3", "text": "你在学习过程中遇到的最大挑战是什么？", "type": "text"},
                {"id": "a4", "text": "你最擅长的学科是什么？", "type": "text"},
                {"id": "a5", "text": "你最不擅长的学科是什么？", "type": "text"}
            ],
            "career": [
                {"id": "c1", "text": "你未来想从事的职业方向是什么？", "type": "text"},
                {"id": "c2", "text": "你对哪些行业最感兴趣？", "type": "text"},
                {"id": "c3", "text": "你希望在大学主修什么专业？", "type": "text"},
                {"id": "c4", "text": "你更倾向于申请哪个国家的大学？", "type": "multiselect", "options": ["中国", "美国", "英国", "加拿大", "澳大利亚", "香港", "新加坡", "日本", "其他"]}
            ],
            "personality": [
                {"id": "ps1", "text": "在团队合作中，你通常扮演什么角色？", "type": "multiselect", "options": ["领导者", "执行者", "策划者", "创新者", "协调者"]},
                {"id": "ps2", "text": "你更喜欢独立工作还是团队合作？", "type": "select", "options": ["独立工作", "团队合作", "视情况而定"]},
                {"id": "ps3", "text": "你在压力下的应对方式是什么？", "type": "text"},
                {"id": "ps4", "text": "你认为自己的创造力如何？", "type": "select", "options": ["非常高", "较高", "一般", "较低"]}
            ],
            "extracurricular": [
                {"id": "e1", "text": "你参与的课外活动有哪些？", "type": "text"},
                {"id": "e2", "text": "你最喜欢的课外活动是什么？", "type": "text"},
                {"id": "e3", "text": "你有什么特殊的才能或爱好？", "type": "text"}
            ],
            "development": [
                {"id": "d1", "text": "你希望在哪些方面得到进一步发展？", "type": "text"},
                {"id": "d2", "text": "你认为自己的优势是什么？", "type": "text"},
                {"id": "d3", "text": "你认为自己需要改进的地方是什么？", "type": "text"},
                {"id": "d4", "text": "你对AI技术有多少了解？", "type": "select", "options": ["非常了解", "有一定了解", "了解不多", "完全不了解"]}
            ]
        }

# Routes
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/assessment')
def assessment():
    questions = load_questions()
    return render_template("assessment.html", questions=questions)

@app.route('/submit', methods=['POST'])
def submit():
    if request.method == 'POST':
        responses = request.json
        session['responses'] = responses
        
        # Generate report
        report = generate_report(responses)
        session['report'] = report
        
        return jsonify({"success": True, "redirect": url_for('report')})

@app.route('/report')
def report():
    if 'report' not in session:
        return redirect(url_for('assessment'))
    
    report = session['report']
    responses = session['responses']
    
    # Get model name for the template
    model_preference = os.getenv('LLM_MODEL_PREFERENCE', 'deepseek').lower()
    if model_preference == 'deepseek':
        model_name = "DeepSeek R1"
    elif model_preference == 'chatglm':
        model_name = "ChatGLM3"
    elif model_preference == 'baichuan':
        model_name = "Baichuan2"
    else:
        model_name = "DeepSeek R1"
    
    return render_template("report.html", report=report, responses=responses, model_name=model_name)

def generate_report(responses):
    """Generate a personalized report based on assessment responses"""
    try:
        logger.info("Generating report using AI Engine")
        # Use the AI Engine to generate an enhanced report
        return ai_engine.generate_enhanced_report(responses)
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}\n{traceback.format_exc()}")
        # Fallback report in case of errors
        return {
            "summary": "无法生成完整的个性化报告。请检查您的回答是否完整，或稍后再试。",
            "error": str(e)
        }

if __name__ == '__main__':
    app.run(debug=True)

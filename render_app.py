"""
Simplified version of the app for Render deployment
"""

from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import os
import sys
import json
import logging
import traceback
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/assessment')
def assessment():
    return render_template('assessment.html')

@app.route('/submit_assessment', methods=['POST'])
def submit_assessment():
    try:
        # Get form data
        responses = {}
        for key, value in request.form.items():
            if key.startswith('p') or key.startswith('a') or key.startswith('c') or key.startswith('ps') or key.startswith('e') or key.startswith('d') or key.startswith('i'):
                responses[key] = value
        
        # Generate a simple report (no AI for now)
        report = generate_simple_report(responses)
        
        # Store in session
        session['responses'] = responses
        session['report'] = report
        
        return jsonify({"success": True, "redirect": url_for('report')})
    except Exception as e:
        logger.error(f"Error processing assessment: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/report')
def report():
    if 'report' not in session:
        return redirect(url_for('assessment'))
    
    report = session['report']
    responses = session['responses']
    
    # Get model name for the template
    model_name = "Llama 3 (8B)"
    
    return render_template("report.html", report=report, responses=responses, model_name=model_name)

def generate_simple_report(responses):
    """Generate a simple report without AI for testing on Render"""
    try:
        # Create a simple report structure
        report = {
            "summary": "这是一个基本的学生评估报告。系统目前处于维护模式，完整的AI增强评估将很快恢复。",
            "academic_analysis": "基于您的回答，您在数学和计算机科学方面表现出色，但在语文和历史方面可能需要更多关注。",
            "personality_insights": "您似乎是一个善于思考和解决问题的人，喜欢有挑战性的工作和独立思考。",
            "career_guidance": "考虑到您的兴趣和优势，人工智能工程师或数据科学家可能是适合您的职业道路。",
            "extracurricular_recommendations": "建议您继续参与编程俱乐部和数学竞赛，这将有助于发展您的技能。",
            "development_plan": "您可以通过提高领导力和沟通能力来进一步发展。时间管理和口头表达也是需要改进的领域。",
            "university_application_advice": "对于申请美国或加拿大的大学，您应该考虑计算机科学或人工智能相关专业。",
            "ai_era_skills": "继续发展您的编程技能，同时也要关注批判性思维和创新能力，这些在AI时代至关重要。"
        }
        
        return report
    except Exception as e:
        logger.error(f"Error generating simple report: {str(e)}\n{traceback.format_exc()}")
        # Fallback report in case of errors
        return {
            "summary": "无法生成完整的个性化报告。请检查您的回答是否完整，或稍后再试。",
            "error": str(e)
        }

if __name__ == '__main__':
    # Get port from environment variable (Render sets this)
    port = int(os.environ.get('PORT', 5000))
    
    # Run the app
    app.run(host='0.0.0.0', port=port, debug=os.getenv('DEBUG', 'False').lower() == 'true')

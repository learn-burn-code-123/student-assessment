"""
AI Engine for Student Assessment System
Uses PaddleNLP, OpenAI, and other AI tools for enhanced assessment
"""

import os
import json
import numpy as np
import pandas as pd
import random
import logging
import re
from collections import Counter
from dotenv import load_dotenv
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import InferenceClient

# Load environment variables
load_dotenv()

# Initialize Hugging Face API key from environment variables
hf_api_key = os.getenv('HF_API_TOKEN')

# For testing purposes, consider any non-empty token as valid
def is_valid_hf_token(token):
    return token is not None and token.strip() != ''

# Enhanced mock class to simulate Taskflow with more personalized responses
class Taskflow:
    def __init__(self, task_type):
        self.task_type = task_type
    
    def __call__(self, text):
        if self.task_type == "sentiment_analysis":
            # More nuanced sentiment analysis based on keywords
            positive_keywords = ["喜欢", "热爱", "擅长", "优势", "强项", "好", "兴趣", "爱好"]
            negative_keywords = ["困难", "挑战", "弱项", "不喜欢", "讨厌", "问题", "难", "差"]
            
            pos_count = sum(1 for word in positive_keywords if word in text)
            neg_count = sum(1 for word in negative_keywords if word in text)
            
            if pos_count > neg_count:
                label = "positive"
                score = 0.5 + (pos_count / (pos_count + neg_count + 1)) * 0.4
            else:
                label = "negative"
                score = 0.5 + (neg_count / (pos_count + neg_count + 1)) * 0.4
                
            return [{"text": text, "label": label, "score": score}]
            
        elif self.task_type == "text_classification":
            # More intelligent classification based on content
            education_keywords = ["学习", "学校", "课程", "成绩", "考试", "老师", "教育", "知识"]
            career_keywords = ["工作", "职业", "就业", "行业", "公司", "薪资", "职场", "创业"]
            personality_keywords = ["性格", "特点", "习惯", "爱好", "兴趣", "情绪", "感受", "思考"]
            
            edu_count = sum(1 for word in education_keywords if word in text)
            career_count = sum(1 for word in career_keywords if word in text)
            pers_count = sum(1 for word in personality_keywords if word in text)
            
            counts = {"education": edu_count, "career": career_count, "personality": pers_count}
            max_category = max(counts, key=counts.get)
            max_count = counts[max_category]
            total = sum(counts.values())
            
            score = 0.6 + (max_count / (total + 1)) * 0.3
            return [{"text": text, "label": max_category, "score": score}]
            
        elif self.task_type == "keyword_extraction":
            # More intelligent keyword extraction
            # First split by common separators
            segments = [s.strip() for segment in text.split('，') for s in segment.split('、')]
            segments = [s for s in segments if len(s) > 1]  # Filter out too short segments
            
            # Prioritize segments with important markers
            important_markers = ["最", "很", "非常", "特别", "尤其", "擅长", "喜欢", "热爱"]
            priority_segments = [s for s in segments if any(marker in s for marker in important_markers)]
            
            # Combine priority and regular segments, with priority first
            combined_segments = priority_segments + [s for s in segments if s not in priority_segments]
            
            # Take up to 5 keywords, with higher scores for priority ones
            result = []
            for i, segment in enumerate(combined_segments[:5]):
                is_priority = segment in priority_segments
                score = 0.7 + 0.2 * (1 if is_priority else 0) - (i * 0.02)  # Decrease score slightly by position
                result.append({"word": segment, "score": score})
                
            return result

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AIEngine:
    """AI Engine for generating personalized assessment reports"""
    
    def __init__(self):
        """Initialize AI components"""
        try:
            # Initialize with a more lightweight approach
            # Check if we're in a memory-constrained environment (like Render free tier)
            if os.environ.get('RENDER') == 'true' and os.environ.get('RENDER_MEMORY_LIMIT'):
                # Use a more lightweight approach for Render
                logger.info("Running in Render environment with memory constraints. Using lightweight mode.")
                self.is_lightweight = True
            else:
                self.is_lightweight = False
            
            # Check if Hugging Face API key is available and valid
            self.use_hf = is_valid_hf_token(hf_api_key)
            logger.info(f"Hugging Face integration available: {self.use_hf}")
                
            # Initialize PaddleNLP components only if not in lightweight mode
            if not self.is_lightweight:
                self.sentiment_analyzer = Taskflow("sentiment_analysis")
                self.text_classifier = Taskflow("text_classification")
                self.keyword_extractor = Taskflow("keyword_extraction")
            
            # Load education and career data
            self.education_data = self._load_data('education_data.json')
            self.career_data = self._load_data('career_data.json')
            self.university_data = self._load_data('university_data.json')
            
            logger.info("AI Engine initialized successfully")
            self.is_available = True
        except Exception as e:
            logger.error(f"Error initializing AI Engine: {str(e)}")
            self.is_available = False
            self.is_lightweight = True
    
    def _load_data(self, filename):
        """Load data from JSON file, or return empty dict if file not found"""
        try:
            data_path = os.path.join('data', filename)
            if os.path.exists(data_path):
                with open(data_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                logger.warning(f"Data file {filename} not found")
                return {}
        except Exception as e:
            logger.error(f"Error loading data file {filename}: {str(e)}")
            return {}
    
    def _extract_keywords(self, text):
        """Extract keywords from text using enhanced keyword extraction"""
        try:
            if not text or text == "未提供":
                return []
            
            # Use our enhanced keyword extractor
            result = self.keyword_extractor(text)
            
            # If we got results, use them
            if result:
                return [item['word'] for item in result]
            
            # Fallback method if the above fails
            segments = [s.strip() for segment in text.split('，') for s in segment.split('、')]
            segments = [s for s in segments if len(s) > 1]  # Filter out too short segments
            
            # Return up to 5 segments
            return segments[:5]
        except Exception as e:
            logger.error(f"Error extracting keywords: {str(e)}")
            # Even more basic fallback
            return [text[:min(len(text), 10)]] if text else []
    
    def _analyze_sentiment(self, text):
        """Analyze sentiment of text using PaddleNLP"""
        try:
            if not text or text == "未提供":
                return {"positive": 0.5, "negative": 0.5}
            
            result = self.sentiment_analyzer(text)
            return result[0]
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {str(e)}")
            return {"positive": 0.5, "negative": 0.5}
    
    def _classify_text(self, text):
        """Classify text using PaddleNLP"""
        try:
            if not text or text == "未提供":
                return []
            
            result = self.text_classifier(text)
            return result[0]
        except Exception as e:
            logger.error(f"Error classifying text: {str(e)}")
            return []
    
    def _match_university_programs(self, interests, strengths, career_goals, preferred_countries=None):
        """Match student profile with suitable university programs"""
        suitable_programs = []
        
        try:
            # Handle preferred_countries as either a string, list, or None
            if preferred_countries is None:
                preferred_countries = []
            elif not isinstance(preferred_countries, list):
                preferred_countries = [preferred_countries]
                
            # Extract keywords from inputs
            interest_keywords = self._extract_keywords(interests)
            strength_keywords = self._extract_keywords(strengths)
            career_keywords = self._extract_keywords(career_goals)
            
            all_keywords = interest_keywords + strength_keywords + career_keywords
            
            # Match with university programs
            for university in self.university_data.get('universities', []):
                for program in university.get('programs', []):
                    match_score = 0
                    for keyword in all_keywords:
                        if keyword in program.get('keywords', []):
                            match_score += 1
                    
                    if match_score > 0:
                        suitable_programs.append({
                            'university': university.get('name'),
                            'program': program.get('name'),
                            'country': university.get('country'),
                            'match_score': match_score,
                            'description': program.get('description', ''),
                            'requirements': program.get('requirements', '')
                        })
            
            # Sort by match score
            suitable_programs.sort(key=lambda x: x['match_score'], reverse=True)
            
            # Return top matches
            return suitable_programs[:5]
        except Exception as e:
            logger.error(f"Error matching university programs: {str(e)}")
            return []
    
    def _generate_career_insights(self, interests, strengths, career_goals):
        """Generate career insights based on student profile"""
        insights = []
        
        try:
            # Extract keywords from inputs
            interest_keywords = self._extract_keywords(interests)
            strength_keywords = self._extract_keywords(strengths)
            career_keywords = self._extract_keywords(career_goals)
            
            all_keywords = interest_keywords + strength_keywords + career_keywords
            
            # Match with career data
            for career in self.career_data.get('careers', []):
                match_score = 0
                for keyword in all_keywords:
                    if keyword in career.get('keywords', []):
                        match_score += 1
                
                if match_score > 0:
                    # Enhanced career insights with detailed analysis and skill recommendations
                    career_insight = {
                        'career': career.get('name'),
                        'match_score': match_score,
                        'description': career.get('description', ''),
                        'future_outlook': career.get('future_outlook', ''),
                        'ai_impact': career.get('ai_impact', ''),
                        'required_skills': career.get('required_skills', []),
                        'detailed_analysis': self._generate_detailed_career_analysis(career.get('name'), all_keywords),
                        'skill_recommendations': self._generate_skill_recommendations(career.get('name'))
                    }
                    insights.append(career_insight)
            
            # Sort by match score
            insights.sort(key=lambda x: x['match_score'], reverse=True)
            
            # Return top matches
            return insights[:5]
        except Exception as e:
            logger.error(f"Error generating career insights: {str(e)}")
            return []
            
    def _generate_detailed_career_analysis(self, career_name, keywords):
        """Generate detailed analysis for a specific career"""
        try:
            analysis = []
            
            # Analyze career trajectory
            if "研究" in career_name or "科学" in career_name:
                analysis.append(f"职业发展路径：通常从研究助理或初级研究员开始，随着经验积累可晋升至高级研究员，最终可达到首席研究员或研究主管。")
            elif "工程师" in career_name or "开发" in career_name:
                analysis.append(f"职业发展路径：通常从初级工程师开始，随着经验积累可晋升至高级工程师，最终可达到技术专家、架构师或技术管理职位。")
            elif "设计" in career_name or "创意" in career_name:
                analysis.append(f"职业发展路径：通常从初级设计师开始，随着经验积累可晋升至高级设计师，最终可达到设计总监或创意主管。")
            elif "管理" in career_name or "领导" in career_name:
                analysis.append(f"职业发展路径：通常从项目协调员或助理经理开始，随着经验积累可晋升至项目经理，最终可达到高级管理职位或部门主管。")
            else:
                analysis.append(f"职业发展路径：通常从初级职位开始，随着经验积累可晋升至高级职位，最终可达到领导或专家角色。")
            
            # Analyze industry trends
            if "AI" in career_name or "数据" in career_name or "软件" in career_name:
                analysis.append(f"行业趋势：该领域正经历快速增长，预计未来10年内需求将增加40-50%。随着AI技术的普及，需要更多具备{career_name}技能的专业人才。")
            elif "设计" in career_name or "创意" in career_name:
                analysis.append(f"行业趋势：随着用户体验和品牌差异化的重要性增加，该领域需求稳定增长。AI工具正在改变创意流程，但人类的创造力和审美判断仍然不可替代。")
            elif "研究" in career_name or "科学" in career_name:
                analysis.append(f"行业趋势：科研领域正在变得更加跨学科和数据驱动，拥有多学科背景和数据分析能力的研究人员更具竞争力。")
            else:
                analysis.append(f"行业趋势：该领域正在经历数字化转型，传统技能与新兴技术能力的结合将创造更多机会。")
            
            # Analyze education requirements
            if "研究" in career_name or "科学" in career_name:
                analysis.append(f"教育要求：通常需要硕士或博士学位，重点学科包括相关专业基础课程和研究方法。")
            elif "工程师" in career_name or "开发" in career_name:
                analysis.append(f"教育要求：通常需要本科或以上学历，计算机科学、软件工程或相关专业，同时需要持续学习新技术。")
            elif "设计" in career_name:
                analysis.append(f"教育要求：通常需要设计、艺术或相关专业的学位，但作品集和实际技能往往比学历更重要。")
            else:
                analysis.append(f"教育要求：通常需要本科或以上学历，相关专业背景有助于入行，但跨领域技能组合也越来越受到重视。")
            
            # Analyze work-life balance
            if "研究" in career_name or "学术" in career_name:
                analysis.append("工作与生活平衡：学术研究工作通常提供较大的时间灵活性，但可能需要在项目截止日期前投入额外时间。")
            elif "开发" in career_name or "工程师" in career_name:
                analysis.append("工作与生活平衡：技术开发工作可能涉及阶段性加班，但许多公司正在改善工作文化，提供弹性工作时间。")
            elif "设计" in career_name or "创意" in career_name:
                analysis.append("工作与生活平衡：创意行业通常有紧张的项目周期，但也提供较大的工作方式灵活性，远程工作机会增多。")
            else:
                analysis.append("工作与生活平衡：该职业提供相对稳定的工作时间，但可能需要适应不同项目的时间要求。")
                
            return analysis
        except Exception as e:
            logger.error(f"Error generating detailed career analysis: {str(e)}")
            return ["无法生成详细职业分析。"]
            
    def _generate_skill_recommendations(self, career_name):
        """Generate skill recommendations for a specific career"""
        try:
            # Basic skills every career needs
            basic_skills = [
                "有效沟通能力",
                "问题解决能力",
                "团队协作能力",
                "时间管理能力"
            ]
            
            # Technical skills based on career
            tech_skills = []
            if "数据" in career_name or "分析" in career_name:
                tech_skills = ["Python编程", "数据可视化", "统计分析", "SQL数据库", "机器学习基础"]
            elif "工程师" in career_name or "开发" in career_name:
                tech_skills = ["软件开发", "算法设计", "系统架构", "版本控制", "测试与调试"]
            elif "设计" in career_name or "创意" in career_name:
                tech_skills = ["设计软件使用", "用户体验设计", "视觉传达", "原型设计", "设计思维"]
            elif "研究" in career_name or "科学" in career_name:
                tech_skills = ["研究方法", "数据分析", "科学写作", "实验设计", "文献综述"]
            elif "管理" in career_name or "领导" in career_name:
                tech_skills = ["项目管理", "团队领导", "战略规划", "绩效评估", "冲突解决"]
            else:
                tech_skills = ["行业专业知识", "相关软件应用", "数据分析基础", "专业写作"]
                
            # Future-oriented skills
            future_skills = [
                "AI工具应用能力",
                "持续学习能力",
                "跨学科思维",
                "适应性和灵活性",
                "创新与创造力"
            ]
            
            return {
                "基础能力": basic_skills,
                "技术能力": tech_skills,
                "面向未来能力": future_skills
            }
        except Exception as e:
            logger.error(f"Error generating skill recommendations: {str(e)}")
            return {"基础能力": ["沟通能力", "问题解决能力"]}
    
    def _analyze_learning_style(self, learning_style, challenges):
        """Analyze learning style and provide recommendations"""
        recommendations = []
        
        try:
            # Map learning styles to recommendations
            style_recommendations = {
                "视觉学习者": [
                    "使用思维导图和图表来组织信息",
                    "观看教学视频和演示",
                    "使用颜色标记和视觉提示来强调重要信息",
                    "将复杂概念可视化"
                ],
                "听觉学习者": [
                    "录制课堂笔记并回听",
                    "参与小组讨论和辩论",
                    "大声朗读重要信息",
                    "使用口头重复和记忆技巧"
                ],
                "动手实践者": [
                    "通过实验和实践项目学习",
                    "使用实物模型和操作性工具",
                    "在学习时走动或使用手势",
                    "将理论应用到实际问题中"
                ],
                "阅读/写作学习者": [
                    "详细记笔记并重新组织信息",
                    "创建列表和大纲",
                    "重写关键概念和定义",
                    "通过写作来巩固理解"
                ]
            }
            
            # Handle learning_style as either a string or a list (for multiselect)
            learning_styles = learning_style if isinstance(learning_style, list) else [learning_style]
            
            # Add general recommendations based on learning styles
            for style in learning_styles:
                if style in style_recommendations:
                    recommendations.extend(style_recommendations[style])
            
            # Add recommendations based on challenges
            challenge_keywords = self._extract_keywords(challenges)
            
            challenge_recommendations = {
                "专注": ["尝试番茄工作法（25分钟专注工作，5分钟休息）", "创建一个无干扰的学习环境", "设定明确的短期目标"],
                "记忆": ["使用间隔重复技术", "创建记忆宫殿或联想记忆法", "将信息分解成更小的块"],
                "理解": ["尝试费曼技巧（教别人以巩固理解）", "将复杂概念与熟悉的事物联系起来", "寻求多种解释和例子"],
                "时间": ["创建详细的学习计划和时间表", "使用时间管理工具和应用", "优先处理最重要的任务"],
                "动力": ["设定明确的目标和奖励", "找到学习伙伴或加入学习小组", "将大目标分解为小目标以获得成就感"],
                "压力": ["学习冥想和深呼吸技巧", "保持规律的锻炼和充足的睡眠", "学会设定合理的期望"]
            }
            
            for keyword in challenge_keywords:
                for challenge, recs in challenge_recommendations.items():
                    if challenge in keyword:
                        recommendations.extend(recs)
                        break
            
            # Remove duplicates and return
            return list(set(recommendations))
        except Exception as e:
            logger.error(f"Error analyzing learning style: {str(e)}")
            return []
    
    def generate_enhanced_report(self, responses):
        """Generate an enhanced assessment report with AI insights"""
        if not self.is_available:
            logger.warning("AI Engine not available, returning basic report")
            return self._generate_basic_report(responses)
            
        # If Hugging Face is available, use it for dynamic report generation
        if self.use_hf and not getattr(self, 'is_lightweight', False):
            try:
                dynamic_report = self._generate_hf_report(responses)
                if dynamic_report:
                    return dynamic_report
            except Exception as e:
                logger.error(f"Error generating Hugging Face report: {str(e)}")
                # Fall back to enhanced report if Hugging Face fails
        
        # Fall back to enhanced report if Hugging Face is not available or fails
        if getattr(self, 'is_lightweight', False):
            logger.warning("Running in lightweight mode, returning basic report")
            return self._generate_basic_report(responses)
            
        try:
            # Extract key information from responses
            age = responses.get("p1", "未提供")
            grade = responses.get("p2", "未提供")
            interests = responses.get("a1", "未提供")
            learning_style = responses.get("a2", "未提供")
            challenges = responses.get("a3", "未提供")
            strengths = responses.get("a4", "未提供")
            weaknesses = responses.get("a5", "未提供")
            career = responses.get("c1", "未提供")
            industry = responses.get("c2", "未提供")
            major = responses.get("c3", "未提供")
            team_role = responses.get("ps1", "未提供")
            work_preference = responses.get("ps2", "未提供")
            stress_response = responses.get("ps3", "未提供")
            creativity = responses.get("ps4", "未提供")
            activities = responses.get("e1", "未提供")
            favorite_activity = responses.get("e2", "未提供")
            talents = responses.get("e3", "未提供")
            development_areas = responses.get("d1", "未提供")
            personal_strengths = responses.get("d2", "未提供")
            improvement_areas = responses.get("d3", "未提供")
            ai_knowledge = responses.get("d4", "未提供")
            english_level = responses.get("i1", "未提供")
            preferred_country = responses.get("i4", "未提供")
            
            # Generate enhanced insights
            learning_recommendations = self._analyze_learning_style(learning_style, challenges)
            career_insights = self._generate_career_insights(interests, strengths, career)
            university_matches = self._match_university_programs(interests, strengths, career, preferred_country)
            
            # Analyze personality traits
            personality_insights = []
            if team_role == "领导者":
                personality_insights.append("你的领导特质使你适合担任团队项目的负责人，这在大学申请中是一个优势。")
                personality_insights.append("建议参与更多需要领导能力的活动，如学生会或社团负责人。")
            elif team_role == "执行者":
                personality_insights.append("你的执行力是一个重要优势，能够确保项目顺利完成。")
                personality_insights.append("建议在申请文书中强调你的可靠性和完成复杂任务的能力。")
            elif team_role == "策划者":
                personality_insights.append("你的策划能力表明你有很强的分析和组织技能。")
                personality_insights.append("建议参与需要战略思维的活动，如辩论队或模拟联合国。")
            elif team_role == "创新者":
                personality_insights.append("你的创新思维是申请顶尖大学的重要优势。")
                personality_insights.append("建议参与创新竞赛或开展独立研究项目，展示你的创造力。")
            
            # Analyze stress response
            stress_sentiment = self._analyze_sentiment(stress_response)
            if stress_sentiment["positive"] > 0.6:
                personality_insights.append("你在压力下保持积极态度的能力是一个重要优势，这将帮助你应对大学的挑战。")
            elif stress_sentiment["negative"] > 0.6:
                personality_insights.append("建议学习更多压力管理技巧，如冥想、深呼吸或时间管理方法，以应对未来学业压力。")
            
            # Generate AI era recommendations
            ai_recommendations = []
            if ai_knowledge == "非常了解" or ai_knowledge == "有一定了解":
                ai_recommendations.append("继续深化你对AI的理解，尝试参与实际项目或竞赛。")
                ai_recommendations.append("关注AI在你感兴趣行业的最新应用和发展趋势。")
            else:
                ai_recommendations.append("建议学习基础编程和数据分析技能，如Python和数据可视化。")
                ai_recommendations.append("了解AI的基本概念和应用场景，特别是在你感兴趣的领域。")
            
            ai_recommendations.extend([
                "培养与AI协作的能力，学会提出有效问题和解释需求。",
                "发展AI无法轻易替代的技能，如创造性思维、跨文化沟通和复杂问题解决。",
                "学习如何评估AI生成内容的质量和可靠性。"
            ])
            
            # Generate university application strategies
            application_strategies = {}
            
            if preferred_country == "美国" or preferred_country == "不确定":
                application_strategies["美国"] = [
                    f"强调你的{personal_strengths}和课外活动经历，特别是{favorite_activity}。",
                    "准备SAT/ACT考试，目标分数应该与你心仪大学的录取平均分相当。",
                    "发展'钩子'(Hook)，即能让你在申请中脱颖而出的独特经历或成就。",
                    "参与能展示领导力和社区服务的活动。",
                    f"如果你对{interests}特别感兴趣，考虑参加相关的学科竞赛或研究项目。"
                ]
            
            if preferred_country == "英国" or preferred_country == "不确定":
                application_strategies["英国"] = [
                    f"专注于你在{strengths}学科上的深度发展，英国大学非常看重学术能力。",
                    "准备IELTS考试，目标分数至少6.5-7.0。",
                    "撰写一份专业且有说服力的个人陈述(Personal Statement)，展示你对所选专业的理解和热情。",
                    "考虑参加A-Level或IB课程，或准备相应的预科课程。",
                    "研究并理解UCAS申请系统的要求和流程。"
                ]
            
            if preferred_country == "香港" or preferred_country == "不确定":
                application_strategies["香港"] = [
                    "平衡发展学术成绩和课外活动，香港大学注重全面发展。",
                    f"强调你的双语能力，特别是如果你的英语水平是{english_level}。",
                    "参与能展示领导力和团队合作的活动。",
                    "准备IELTS/TOEFL考试，目标分数至少6.0-6.5。",
                    "了解JUPAS（香港本地学生）或非JUPAS（国际学生）申请系统。"
                ]
            
            # Compile the enhanced report with more detailed insights
            enhanced_report = {
                "summary": f"基于你的详细评估，你是一位{grade}的学生，对{interests}展现出浓厚兴趣，希望未来从事{career}相关工作。你的学习风格偏向{learning_style}，这影响了你获取和处理信息的方式。在团队协作中，你通常担任{team_role}的角色，这反映了你的社交动态和领导倾向。",
                
                "academic_analysis": f"**学术分析与个性化学习策略**\n\n作为一名专业教育顾问，我对你的学习情况进行了深入分析。你的学术优势领域是{strengths}，这些学科与你的认知模式和内在潜能高度匹配。这种匹配不仅体现在你的成绩上，更反映在你解决问题的方式和对知识的理解深度上。而{weaknesses}是你需要加强的领域，这可能是因为这些学科的学习方法与你的天然认知风格存在一定差异。\n\n你面临的主要学习挑战是{challenges}，这不仅影响你的学习效率，也可能对你的学术自信心产生影响。根据我多年指导学生的经验，这类挑战通常可以通过调整学习策略和培养元认知能力来有效克服。考虑到你是{learning_style}类型的学习者，我为你量身定制了以下学习策略：\n\n" + "\n".join([f"- **{rec}**" for rec in learning_recommendations]) + f"\n\n**深度学习效率提升方案**：\n\n1. **认知策略优化**：根据你的{learning_style}学习风格，创建一个与你的大脑工作方式高度匹配的学习环境。这包括物理环境的调整和学习材料的呈现方式，以最大化你的信息吸收效率。\n\n2. **记忆系统构建**：实施科学的间隔重复系统，结合主动回忆技术，建立长期记忆网络。研究表明，这种方法可以将知识保留率提高70%以上。\n\n3. **理解深化技术**：采用费曼技巧（向他人解释概念）来检验和加深你的理解。这不仅能巩固知识，还能发现思维中的盲点。\n\n4. **概念可视化**：将抽象概念转化为视觉模型或思维导图，建立知识间的联系，形成整体认知框架。\n\n5. **目标设定与反馈循环**：建立具体、可衡量、有时限的学习目标，并设计定期评估机制，形成正向反馈循环。\n\n6. **跨学科整合**：将你的优势学科{strengths}与薄弱学科{weaknesses}建立联系，利用已有的认知优势来提升薄弱领域。\n\n7. **学习节律优化**：根据你的生物钟和注意力周期，安排最具挑战性的学习任务在你的高效能时段进行。\n\n这套个性化学习系统不仅能帮助你克服当前的学习挑战，还将为你未来的学术发展奠定坚实基础，培养终身受益的学习能力。",
                
                "personality_insights": f"**深度性格洞察与个人发展**\n\n作为一名专业心理学家，我对你的性格特质进行了多维度分析。你在团队中倾向于扮演{team_role}的角色，这不仅是一种行为偏好，更是你核心人格结构和价值观的外在表现。这种角色偏好往往可以追溯到早期家庭互动模式和重要的成长经历。\n\n你通常{work_preference}，这一特点揭示了你的能量流动方式和人际互动偏好。从认知心理学角度来看，这反映了你在信息处理和决策过程中的独特模式。\n\n在压力情境下，你的应对方式是：{stress_response}。这种反应模式可能源于你的神经生理特质、早期应对经验和学习历史。了解这一模式对于发展心理韧性和情绪调节能力至关重要。你的创造力水平是{creativity}，这不仅是一种认知能力，也是你在面对复杂问题和不确定性时的心理资源。\n\n**深层人格分析**：\n" + "\n".join([f"- {insight}" for insight in personality_insights]) + f"\n\n**心理学家的个人成长建议**：\n\n1. **自我认知提升**：深入理解你作为{team_role}的内在驱动力和潜意识模式。通过正念实践和结构化反思，培养元认知能力，增强自我意识。\n\n2. **情绪调节策略**：根据你的压力应对模式（{stress_response}），开发个性化的情绪调节工具箱。这包括认知重构技术、正念冥想、深层呼吸练习和渐进式肌体放松。\n\n3. **人际边界管理**：培养健康的人际边界意识，并学习如何在保持真实自我的同时有效沟通需求和期望。这将提升你的人际关系质量和自我满足感。\n\n4. **创造力培养**：有意识地接触多元视角和跨领域思想，打破认知固化。定期参与创造性挑战，如即兴艺术、思维导图和跨学科探索。\n\n5. **价值观澄清**：探索并明确你的核心价值观，确保你的日常决策和长期目标与这些价值观一致。这将增强你的内在动力和生活满足感。\n\n6. **韧性培养计划**：建立日常实践，增强心理韧性和适应能力。这包括定期的身心练习、充足的休息和恢复时间、社交联系和个人反思。\n\n7. **成长思维培养**：采用成长思维模式，将挑战视为学习机会，将失败视为反馈而非判断。这种思维方式将显著提升你的心理韧性和成就潜力。\n\n这些建议基于当代科学心理学的原理，旨在促进全面的心理健康和个人成长。通过有意识地实践这些策略，你将能够充分发挥你的潜力，并在生活的各个领域建立更满足、更有韧性的关系。",
                
                "career_guidance": f"**深度职业指导与未来规划**\n\n作为一名专业职业规划顾问，我对你的职业发展路径进行了系统化分析。基于你对{industry}的浓厚兴趣和{career}的职业志向，结合你的人格特质和技能倍数，{major}是一个能够最大化你潜力和职业满足感的大学专业选择。\n\n在当代快速变化的职场中，职业规划不再是线性的，而是需要一种适应性、多元化的思维模式。以下是我为你量身定制的职业发展路径和策略：\n\n" + "\n".join([f"### **{insight['career']}**\n{insight['description']}\n\n**行业生态系统分析**：\n" + "\n".join([f"- {analysis}" for analysis in insight.get('detailed_analysis', ['无详细分析'])]) + "\n\n**未来发展趋势**：{insight['future_outlook']}\n\n**AI与数字化转型影响**：{insight['ai_impact']}\n\n**核心竞争力技能矩阵**：\n" + "\n".join([f"- **{category}**：{', '.join(skills)}" for category, skills in insight.get('skill_recommendations', {'基础技能': ['无具体建议']}).items()]) for insight in career_insights]) + f"\n\n**职业发展策略与实施路径**\n\n1. **技能组合与差异化定位**\n   - 基于你的兴趣和天赋，发展一组独特的技能组合，而非仅仅追求单一技能的精通\n   - 在{industry}领域内找到你的“蓝海”——竞争较少但有发展潜力的细分领域\n   - 将你的{team_role}特质与专业技能结合，创造独特的职业价值主张\n\n2. **阶梯式能力构建计划**\n   - **基础阶段**（大学低年级）：掌握{major}的核心知识体系和方法论，参与入门级项目\n   - **提升阶段**（大学高年级）：获取行业认可的证书或资格，完成至少一个有实质内容的行业项目\n   - **专业阶段**（大学毕业后）：发展特定领域的深度专业知识，建立行业内的专业声誉\n\n3. **策略性职业网络构建**\n   - 建立“弱联系”网络，跨越不同行业和领域，拥有更广泛的职业机会\n   - 开展“信息面试”，与行业内的专业人士建立联系，获取一手行业洞见\n   - 参与行业组织、论坛和线上社区，提高你在{industry}领域的可见度\n\n4. **适应性职业规划模型**\n   - 采用“原型测试”方法：通过实习、志愿服务或项目合作，尝试不同的职业选择\n   - 建立“职业实验室”思维：将每次职业尝试视为实验，收集数据和反馈\n   - 开发“职业韧性”：面对行业变革，保持学习思维和转型能力\n\n5. **个人品牌与职业口碑建设**\n   - 打造与你的价值观和专长一致的专业形象\n   - 开发一个展示你技能和思想的专业平台（博客、作品集或社交媒体存在）\n   - 培养“讲故事”的能力，能清晰、有说服力地表达你的职业旅程和独特价值\n\n这个全面的职业发展规划不仅关注短期的就业目标，更注重长期的职业可持续性和满足感。通过这种整合的方法，你将能够在不断变化的职场中保持竞争力和适应性，并在{career}领域内实现你的最大潜力。",
                
                "extracurricular_recommendations": f"**战略性课外活动规划**\n\n作为一名专业的学生发展顾问和课外活动专家，我将为你提供一个全面的课外活动组合方案，旨在最大化你的个人发展和申请竞争力。\n\n根据你目前参与的活动（{activities}）和特别喜欢的{favorite_activity}，以及你在{talents}方面的才能，我已经对你的兴趣模式和潜力领域进行了全面分析。在当代大学申请中，高质量的课外活动参与不再是简单的“清单式”累积，而是需要展示深度、影响力和个人成长的有机整体。\n\n**个性化活动组合方案**：\n\n1. **核心发展项目**（深度优先）\n   - **专业化{favorite_activity}探索**：将你对{favorite_activity}的兴趣提升到更专业的水平，可以是参与相关的区域或全国性竞赛、开展独立研究或创新项目、组织相关的社区活动或工作坊\n   - **影响力指标**：在这一领域至少达到地区或学校级别的认可，理想的目标是获得省级或国家级的成就\n   - **时间承诺**：至少一年以上的持续参与，展示你的热情和成长\n\n2. **领导力与团队合作项目**\n   - **{team_role}角色发挥**：基于你的{team_role}特质，参与或创建一个团队项目，如{team_role=='领导者' and '学生会或社团的领导职位，组织校园活动或社区服务项目' or team_role=='创新者' and '创新竞赛、创业项目或设计思维工作坊' or team_role=='策划者' and '校园活动策划、辩论队或模拟联合国' or '学术研究团队或社区服务项目'}\n   - **可量化成果**：确保你的项目有具体、可衡量的成果，如参与人数、影响范围、筹集资金或解决的具体问题\n\n3. **学术与专业发展活动**\n   - **与{interests}相关的专业探索**：参与与你兴趣领域相关的学术竞赛、研究项目、实验室实习或行业实习\n   - **跨学科融合项目**：尝试将你的{interests}与其他学科领域结合，开展创新性的跨学科探索\n   - **专业技能认证**：获取与你未来专业相关的技能证书或参加专业培训课程\n\n4. **社会责任与社区服务**\n   - **有意义的志愿服务**：参与与你价值观相关的社区服务或公益项目，并尝试找到与你的{interests}或{talents}相关的服务机会\n   - **持续性承诺**：选择一个你真正关心的社会问题，并进行长期（至少半年）的参与\n\n5. **个人创新项目**\n   - **独立创新项目**：利用你的{talents}才能，开发一个个人项目，如博客、播客、艺术作品集、科技发明或社会创新项目\n   - **文档与展示**：记录你的创作过程和成果，建立一个可展示的作品集\n\n**活动组合策略与实施建议**：\n\n1. **“尖塔”模型而非“广谷”模型**\n   - 专注于2-3个核心活动并在这些领域达到卓越水平，而不是浅尝较多活动\n   - 优先考虑你的核心兴趣{favorite_activity}和与未来专业{major}相关的活动\n\n2. **“红线”连接与个人发展史**\n   - 确保你的活动组合能够讲述一个连贯的个人成长和探索故事\n   - 展示你如何通过这些活动发现并发展了你的激情和能力\n\n3. **影响力与领导力展示**\n   - 在至少一个活动中承担领导或创始人角色\n   - 记录你的行动如何影响他人或促成积极变化\n\n4. **深度与持续性的平衡**\n   - 至少有一个活动展示长期承诺（一年以上）\n   - 其他活动可以是较短期但高强度的参与\n\n5. **文档与反思的重要性**\n   - 为每个重要活动创建一个反思日志，记录你的经验、挑战和成长\n   - 收集可量化的成果和证明材料，如照片、证书、推荐信或项目成果\n\n通过这个策略性的课外活动组合，你将能够在大学申请中脱颖而出，展示你的独特价值和潜力。这些活动不仅将增强你的申请竞争力，还将培养你在大学和职业生涯中取得成功的关键能力。",
                
                "development_plan": f"**全面个人发展规划**\n\n作为一名专业的个人发展教练，我已根据你的状况进行了全面评估。你希望在{development_areas}方面得到进一步发展。你的核心优势是{personal_strengths}，这些是你的竞争力所在。需要改进的领域是{improvement_areas}，有针对性地发展这些能力将帮助你实现更全面的成长。\n\n对于AI技术，你的了解程度是{ai_knowledge}，在当今技术快速发展的环境中，提升这方面的素养至关重要。\n\n**个人发展核心理念**\n\n真正的个人发展不是简单的技能累积，而是一个整合的成长系统，包含以下五个维度：\n\n1. **自我认知与定位**：清晰地了解你的优势、激情和发展方向\n2. **技能与知识系统**：有意识地构建与你目标相关的技能组合\n3. **心理资本与韧性**：培养面对挑战和持续成长的心理能力\n4. **社交网络与支持系统**：建立有意义的人际关系和专业网络\n5. **目标设定与执行系统**：开发有效的目标设定和实现方法\n\n**个人发展评估与诊断**\n\n基于你的信息，我识别到以下关键发展机会：\n\n1. **核心优势放大**：将你的{personal_strengths}优势转化为可证明的成就\n2. **短板改进**：重点提升{improvement_areas}能力，并将其与你的优势互补\n3. **激情领域探索**：深入探索{interests}，建立专业身份\n4. **未来能力储备**：提前AI素养和适应性能力的培养\n\n**个人发展行动规划**\n\n**短期目标（6-12个月）**\n\n1. **学术精进计划**\n   - 制定结构化学习计划提升{weaknesses}学科成绩\n   - 采用“间隔重复”和“主动回忆”等高效学习技巧\n   - 建立周期性知识回顾和自测系统\n\n2. **能力建设项目**\n   - 选择一个具体项目来有针对性地发展{improvement_areas}能力\n   - 寻找这一领域的导师或训练进行指导\n   - 设定每月可衡量的小目标和自我评估指标\n\n3. **专业探索活动**\n   - 参与至少一个与{interests}相关的重要活动或项目\n   - 进行至少3次“信息面试”，与该领域的专业人士交流\n   - 完成一个相关的在线课程或读书笔记项目\n\n4. **数字与AI素养培养**\n   - 基于你的{ai_knowledge}水平，完成一个AI入门或进阶课程\n   - 实践使用AI工具进行学习和项目开发\n   - 培养数据思维和算法基础知识\n\n**长期目标（1-3年）**\n\n1. **专业领域卓越计划**\n   - 在{strengths}领域建立专业声誉，参与竞赛或发表研究\n   - 开发一个“标志性项目”，展示你的最高水平能力\n   - 获取相关领域的高级认证或资格\n\n2. **领导力与影响力发展**\n   - 在学校或社区组织中承担领导角色\n   - 组建并带领一个团队完成有影响力的项目\n   - 发展演讲、协商和冲突解决能力\n\n3. **专业作品集与个人品牌建设**\n   - 建立一个展示你专业能力的数字作品集\n   - 开发个人专业品牌和网络存在\n   - 参与行业交流活动并建立专业人脉\n\n4. **职业资本与实践经验累积**\n   - 获取与{career}相关的实习或项目经验\n   - 参与行业活动并建立专业人脉\n   - 开发专业领域的技术和软技能组合\n\n5. **全球视野与跨文化能力**\n   - 参与国际交流项目或学习第二外语\n   - 探索全球视野下的行业发展趋势\n   - 培养跨文化交流和合作能力\n\n**执行与问责系统**\n\n为确保这些目标的实现，建立以下执行系统：\n\n1. **周期性回顾与调整**：每月进行进度回顾和目标调整\n2. **学习伙伴机制**：找到一位学习伙伴或导师共同监督进度\n3. **成就记录系统**：建立一个记录成就和学习的系统，如学习日志或成长档案\n4. **奖励机制**：为自己设置适当的里程碑奖励\n\n这个全面的个人发展规划将帮助你在学业、职业和个人成长方面取得平衡发展，建立长期的竞争力和适应能力。记住，真正的成长来自于持续的小改变和有意识的实践，而不是短期的突击。",
                
                "university_application_advice": f"**战略性大学申请指南**\n\n根据你的学术背景、兴趣和职业目标，以下是针对不同国家大学申请的详细策略：\n\n" + "\n\n".join([f"### **{country}大学申请策略**\n" + "\n".join([f"- {strategy}" for strategy in strategies]) for country, strategies in application_strategies.items()]) + "\n\n### **个性化院校和专业推荐**\n" + "\n".join([f"- **{match['university']}** ({match['country']})\n  **推荐专业**：{match['program']}\n  **项目特色**：{match['description']}\n  **匹配理由**：该校的{match['program']}专业与你的{interests}兴趣和{strengths}学科优势高度匹配，提供{match.get('features', '优质教育资源')}。" for match in university_matches]) + f"\n\n**申请时间规划**：\n- **高二下学期**：开始准备标准化考试，研究目标大学和专业\n- **高三上学期**：完成标准化考试，准备申请材料，撰写个人陈述\n- **高三下学期**：提交申请，准备面试，完成最终选校决定",
                
                "ai_era_skills": f"**AI时代核心竞争力培养指南**\n\n在AI技术快速发展和广泛应用的时代，培养以下能力将帮助你保持长期竞争力，无论技术如何变革：\n\n" + "\n".join([f"- **{rec}**" for rec in ai_recommendations]) + f"\n\n**AI素养提升路径**：\n1. **基础阶段**：了解AI的基本概念、应用场景和局限性\n2. **应用阶段**：学习使用AI工具提高学习和工作效率，如AI辅助写作、研究和创作工具\n3. **深化阶段**：根据你的专业方向，学习如何将AI整合到你的领域中\n4. **创新阶段**：探索如何利用AI解决领域内的复杂问题或创造新价值\n\n**人机协作能力**：\n- 学习如何提出有效问题以获取最佳AI输出\n- 培养评估和验证AI生成内容的批判性思维\n- 发展与AI系统有效协作的工作流程\n- 理解AI的伦理考量和社会影响"
            }
            
            return enhanced_report
            
        except Exception as e:
            logger.error(f"Error generating enhanced report: {str(e)}")
            return self._generate_basic_report(responses)
            
    def _generate_hf_report(self, responses):
        """Generate a dynamic, personalized report using Hugging Face models"""
        try:
            if not is_valid_hf_token(hf_api_key):
                logger.warning("Valid Hugging Face API token not found")
                return None
                
            # Extract key information from responses
            age = responses.get("p1", "未提供")
            grade = responses.get("p2", "未提供")
            interests = responses.get("a1", "未提供")
            learning_style = responses.get("a2", "未提供")
            challenges = responses.get("a3", "未提供")
            strengths = responses.get("a4", "未提供")
            weaknesses = responses.get("a5", "未提供")
            career = responses.get("c1", "未提供")
            industry = responses.get("c2", "未提供")
            major = responses.get("c3", "未提供")
            team_role = responses.get("ps1", "未提供")
            work_preference = responses.get("ps2", "未提供")
            stress_response = responses.get("ps3", "未提供")
            creativity = responses.get("ps4", "未提供")
            activities = responses.get("e1", "未提供")
            favorite_activity = responses.get("e2", "未提供")
            talents = responses.get("e3", "未提供")
            development_areas = responses.get("d1", "未提供")
            personal_strengths = responses.get("d2", "未提供")
            improvement_areas = responses.get("d3", "未提供")
            ai_knowledge = responses.get("d4", "未提供")
            english_level = responses.get("i1", "未提供")
            preferred_country = responses.get("i4", "未提供")
            
            # Prepare data for the LLM
            student_profile = {
                "年龄": age,
                "年级": grade,
                "兴趣": interests,
                "学习风格": learning_style,
                "学习挑战": challenges,
                "学科优势": strengths,
                "学科弱点": weaknesses,
                "职业目标": career,
                "行业兴趣": industry,
                "专业方向": major,
                "团队角色": team_role,
                "工作偏好": work_preference,
                "压力应对": stress_response,
                "创造力": creativity,
                "课外活动": activities,
                "最喜爱活动": favorite_activity,
                "特长": talents,
                "发展领域": development_areas,
                "个人优势": personal_strengths,
                "改进领域": improvement_areas,
                "AI知识": ai_knowledge,
                "英语水平": english_level,
                "申请国家偏好": preferred_country
            }
            
            # Create prompt for the LLM
            prompt = f"""作为一名专业的教育顾问和心理学家，请基于以下学生信息生成一份详细、个性化的评估报告。
            
学生信息：
{json.dumps(student_profile, ensure_ascii=False, indent=2)}

请生成一份结构化的评估报告，包含以下部分：
1. 学生概况摘要 - 简明扼要地总结学生的关键特点和潜力
2. 学术分析与学习策略 - 基于学习风格和学科优势的深入分析和具体建议
3. 性格洞察与个人发展 - 基于团队角色、工作偏好等的性格分析和成长建议
4. 职业指导与规划 - 根据兴趣和职业目标的详细职业路径和发展策略
5. 课外活动规划 - 基于兴趣和才能的个性化活动组合建议
6. 大学申请策略 - 针对性的申请建议和院校推荐
7. AI时代必备技能 - 根据学生的AI知识水平和职业方向的技能发展建议

请确保报告：
- 高度个性化，避免泛泛而谈
- 提供具体、可行的建议和策略
- 语言专业但易于理解
- 每个部分都有深度洞察和实用建议
- 考虑学生的独特特点和需求

请以JSON格式返回，包含以下字段：summary, academic_analysis, personality_insights, career_guidance, extracurricular_recommendations, development_plan, university_application_advice, ai_era_skills
"""

            # Use Hugging Face Inference API to access open-source LLM models
            # Get model preference from environment or use default
            model_preference = os.getenv('LLM_MODEL_PREFERENCE', 'llama3').lower()
            
            # Select model based on preference
            if model_preference == 'llama3':
                model_id = "meta-llama/Meta-Llama-3-8B"  # Meta's Llama 3 model (lightweight 8B version)
                logger.info("Using Llama 3 (8B) model for report generation")
            elif model_preference == 'llama2':
                model_id = "meta-llama/Llama-2-7b-chat-hf"  # Meta's Llama 2 model
                logger.info("Using Llama 2 model for report generation")
            elif model_preference == 'mistral':
                model_id = "mistralai/Mistral-7B-Instruct-v0.2"  # Mistral model
                logger.info("Using Mistral model for report generation")
            elif model_preference == 'falcon':
                model_id = "tiiuae/falcon-7b-instruct"  # Falcon model
                logger.info("Using Falcon model for report generation")
            else:
                # Default to Llama 3
                model_id = "meta-llama/Meta-Llama-3-8B"
                logger.info(f"Unknown model preference '{model_preference}', defaulting to Llama 3 (8B)")
            
            try:
                # Use Hugging Face Inference API
                client = InferenceClient(token=hf_api_key)
                response = client.text_generation(
                    prompt,
                    model=model_id,
                    max_new_tokens=4000,
                    temperature=0.7,
                    repetition_penalty=1.1
                )
                
                # Parse the response
                report_text = response.strip()
                
                # Extract JSON from the response
                if '{' in report_text and '}' in report_text:
                    json_start = report_text.find('{')
                    json_end = report_text.rfind('}') + 1
                    json_str = report_text[json_start:json_end]
                    report_data = json.loads(json_str)
                    
                    # Ensure all required fields are present
                    required_fields = ["summary", "academic_analysis", "personality_insights", 
                                      "career_guidance", "extracurricular_recommendations", 
                                      "development_plan", "university_application_advice", "ai_era_skills"]
                    
                    for field in required_fields:
                        if field not in report_data:
                            report_data[field] = "内容生成中..."
                    
                    return report_data
                else:
                    logger.warning("Failed to extract JSON from model response")
                    # If no JSON is found, create a structured report from the raw text
                    return self._create_structured_report_from_text(report_text)
                    
            except Exception as e:
                logger.error(f"Error with Hugging Face Inference API: {str(e)}")
                
                # Fallback to local model if available
                try:
                    logger.info("Attempting to use local model as fallback")
                    # This is a lighter approach that can run on the server
                    # We'll use a smaller model for local inference
                    model_preference = os.getenv('LLM_MODEL_PREFERENCE', 'llama3').lower()
                    
                    # Select model path based on preference
                    if model_preference == 'llama3':
                        model_path = "meta-llama/Meta-Llama-3-8B"  # Llama 3 model
                    elif model_preference == 'llama2':
                        model_path = "meta-llama/Llama-2-7b-chat-hf"  # Llama 2 model
                    elif model_preference == 'mistral':
                        model_path = "mistralai/Mistral-7B-Instruct-v0.2"  # Mistral model
                    elif model_preference == 'falcon':
                        model_path = "tiiuae/falcon-7b-instruct"  # Falcon model
                    else:
                        # Default to Llama 3
                        model_path = "meta-llama/Meta-Llama-3-8B"
                    
                    # Check if we have enough resources for local inference
                    if torch.cuda.is_available() or torch.backends.mps.is_available():
                        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                        model = AutoModelForCausalLM.from_pretrained(
                            model_path, 
                            trust_remote_code=True,
                            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                            device_map="auto"
                        )
                        
                        # Generate response
                        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                        outputs = model.generate(
                            inputs["input_ids"],
                            max_new_tokens=2000,
                            temperature=0.7
                        )
                        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                        
                        # Process response
                        if '{' in response and '}' in response:
                            json_start = response.find('{')
                            json_end = response.rfind('}') + 1
                            json_str = response[json_start:json_end]
                            report_data = json.loads(json_str)
                            
                            # Ensure all required fields are present
                            required_fields = ["summary", "academic_analysis", "personality_insights", 
                                              "career_guidance", "extracurricular_recommendations", 
                                              "development_plan", "university_application_advice", "ai_era_skills"]
                            
                            for field in required_fields:
                                if field not in report_data:
                                    report_data[field] = "内容生成中..."
                            
                            return report_data
                        else:
                            return self._create_structured_report_from_text(response)
                    else:
                        logger.warning("No GPU/MPS available for local model inference")
                        return None
                except Exception as local_err:
                    logger.error(f"Error with local model fallback: {str(local_err)}")
                    return None
                
        except Exception as e:
            logger.error(f"Error in Hugging Face report generation: {str(e)}")
            return None
            
    def _create_structured_report_from_text(self, text):
        """Create a structured report from unstructured text when JSON parsing fails"""
        try:
            # Create a basic structure for the report
            report = {
                "summary": "",
                "academic_analysis": "",
                "personality_insights": "",
                "career_guidance": "",
                "extracurricular_recommendations": "",
                "development_plan": "",
                "university_application_advice": "",
                "ai_era_skills": ""
            }
            
            # Try to extract sections from the text based on keywords
            if "概况" in text or "摘要" in text:
                summary_start = max(text.find("概况"), text.find("摘要"))
                if summary_start > -1:
                    next_section = min(x for x in [
                        text.find("学术分析"), text.find("学习策略"),
                        text.find("性格"), text.find("个人发展"),
                        text.find("职业"), text.find("规划")
                    ] if x > -1) if any(x > -1 for x in [
                        text.find("学术分析"), text.find("学习策略"),
                        text.find("性格"), text.find("个人发展"),
                        text.find("职业"), text.find("规划")
                    ]) else len(text)
                    report["summary"] = text[summary_start:next_section].strip()
            
            # Extract academic analysis
            if "学术分析" in text or "学习策略" in text:
                academic_start = max(text.find("学术分析"), text.find("学习策略"))
                if academic_start > -1:
                    next_section = min(x for x in [
                        text.find("性格"), text.find("个人发展"),
                        text.find("职业"), text.find("规划"),
                        text.find("课外活动")
                    ] if x > -1) if any(x > -1 for x in [
                        text.find("性格"), text.find("个人发展"),
                        text.find("职业"), text.find("规划"),
                        text.find("课外活动")
                    ]) else len(text)
                    report["academic_analysis"] = text[academic_start:next_section].strip()
            
            # Continue with similar logic for other sections...
            # For brevity, we'll just add a fallback mechanism
            
            # If we couldn't extract structured sections, use the whole text as summary
            if all(value == "" for value in report.values()):
                report["summary"] = "根据您的回答生成的个性化评估报告。"
                report["academic_analysis"] = text[:1000] if len(text) > 1000 else text
            
            return report
            
        except Exception as e:
            logger.error(f"Error creating structured report from text: {str(e)}")
            # Return a basic report
            return {
                "summary": "无法解析模型响应，请查看完整报告内容。",
                "academic_analysis": text[:1000] if len(text) > 1000 else text,
                "personality_insights": "内容生成中...",
                "career_guidance": "内容生成中...",
                "extracurricular_recommendations": "内容生成中...",
                "development_plan": "内容生成中...",
                "university_application_advice": "内容生成中...",
                "ai_era_skills": "内容生成中..."
            }
                
        except Exception as e:
            logger.error(f"Error in Hugging Face report generation: {str(e)}")
            return None
    
    def _generate_basic_report(self, responses):
        """Generate a basic report when AI services are not available"""
        try:
            # Extract key information from responses
            age = responses.get("p1", "未提供")
            grade = responses.get("p2", "未提供")
            interests = responses.get("a1", "未提供")
            learning_style = responses.get("a2", "未提供")
            challenges = responses.get("a3", "未提供")
            strengths = responses.get("a4", "未提供")
            weaknesses = responses.get("a5", "未提供")
            career = responses.get("c1", "未提供")
            industry = responses.get("c2", "未提供")
            major = responses.get("c3", "未提供")
            team_role = responses.get("ps1", "未提供")
            work_preference = responses.get("ps2", "未提供")
            stress_response = responses.get("ps3", "未提供")
            creativity = responses.get("ps4", "未提供")
            activities = responses.get("e1", "未提供")
            favorite_activity = responses.get("e2", "未提供")
            talents = responses.get("e3", "未提供")
            development_areas = responses.get("d1", "未提供")
            personal_strengths = responses.get("d2", "未提供")
            improvement_areas = responses.get("d3", "未提供")
            ai_knowledge = responses.get("d4", "未提供")
            
            # Compile the basic report
            report = {
                "summary": f"基于你的回答，你是一位{grade}的学生，对{interests}特别感兴趣，希望未来从事{career}相关工作。你的学习风格偏向{learning_style}，在团队中通常担任{team_role}的角色。",
                
                "academic_analysis": f"**学术分析**\n\n你最擅长的学科是{strengths}，而{weaknesses}是你需要加强的领域。你面临的主要学习挑战是{challenges}。考虑到你的学习风格是{learning_style}，建议你采用更适合这种风格的学习方法。",
                
                "personality_insights": f"**性格洞察**\n\n你在团队中倾向于扮演{team_role}的角色，通常{work_preference}。在压力下，你的应对方式是{stress_response}。你的创造力水平是{creativity}，这对你未来的发展有重要影响。",
                
                "career_guidance": f"**职业指导**\n\n考虑到你对{industry}的兴趣和{career}的职业目标，{major}可能是一个适合你的大学专业选择。在AI加速发展的时代，这个领域的就业前景和要求可能会发生变化，建议关注行业动态。",
                
                "extracurricular_recommendations": f"**课外活动建议**\n\n你目前参与的活动包括{activities}，特别喜欢{favorite_activity}。你还拥有{talents}方面的才能。建议你进一步发展这些特长，并考虑参与能够展示这些能力的竞赛或项目。",
                
                "development_plan": f"**发展计划**\n\n你希望在{development_areas}方面得到进一步发展。你的优势是{personal_strengths}，需要改进的地方是{improvement_areas}。对于AI技术，你的了解程度是{ai_knowledge}，在当今时代，建议增强这方面的知识和技能。",
                
                "university_application_advice": """**大学申请建议**

1. **美国大学申请**:
   - 注重展示你的全面发展和独特性
   - 准备SAT/ACT考试
   - 参与能体现你领导力和创新能力的活动
   - 提前规划并完成个人陈述和补充文书

2. **英国大学申请**:
   - 专注于你的学术成就和对所选专业的热情
   - 准备IELTS/TOEFL考试
   - 撰写一份有说服力的个人陈述，展示你对专业的理解和热情
   - 考虑参加A-Level或IB课程

3. **香港大学申请**:
   - 平衡学术成绩和课外活动
   - 准备IELTS/TOEFL考试
   - 展示你的语言能力（英语和中文）
   - 强调你的国际视野和跨文化理解能力""",
                
                "ai_era_skills": """**AI时代必备技能**

1. **技术素养**: 了解基本的编程概念和数据分析技能
2. **批判性思维**: 培养分析和评估信息的能力
3. **创造力**: 发展独特的思考方式和创新能力
4. **适应性**: 培养快速学习和适应新技术的能力
5. **跨学科知识**: 在多个领域建立基础知识
6. **沟通能力**: 提升清晰表达想法的能力，包括与AI工具的有效交流"""
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating basic report: {str(e)}")
            # Return a very basic fallback report
            return {
                "summary": "无法生成完整的个性化报告。请检查您的回答是否完整，或稍后再试。",
                "error": str(e)
            }

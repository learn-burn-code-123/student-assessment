#!/usr/bin/env python3
"""
Test script for Hugging Face integration in the AI Engine
"""

import os
import json
import logging
import unittest
from unittest.mock import patch, MagicMock
from dotenv import load_dotenv
from ai_engine import AIEngine, is_valid_hf_token

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
# Try to load from .env.test for testing purposes
if os.path.exists('.env.test'):
    load_dotenv('.env.test')
    logger.info("Loaded environment variables from .env.test")
else:
    load_dotenv()
    logger.info("Loaded environment variables from .env")


class TestHFIntegration(unittest.TestCase):
    """Test cases for Hugging Face integration"""
    
    def setUp(self):
        """Set up test data"""
        # Sample responses for testing
        self.sample_responses = {
            "p1": "16",  # 年龄
            "p2": "高二",  # 年级
            "a1": "数学、编程、人工智能",  # 兴趣
            "a2": "我喜欢通过实践和动手操作来学习",  # 学习风格
            "a3": "有时候难以长时间集中注意力",  # 学习挑战
            "a4": "数学和计算机科学",  # 学科优势
            "a5": "语文和历史",  # 学科弱点
            "c1": "人工智能工程师或数据科学家",  # 职业目标
            "c2": "科技行业",  # 行业兴趣
            "c3": "计算机科学或人工智能",  # 专业方向
            "ps1": "我通常是团队中的思考者和问题解决者",  # 团队角色
            "ps2": "我喜欢有挑战性的工作，能够独立思考",  # 工作偏好
            "ps3": "在压力下我会寻求解决方案，但有时会感到焦虑",  # 压力应对
            "ps4": "我喜欢创新思维和尝试新方法",  # 创造力
            "e1": "编程俱乐部、数学竞赛、篮球",  # 课外活动
            "e2": "参加编程比赛和开发个人项目",  # 最喜爱活动
            "e3": "编程、数学解题、团队合作",  # 特长
            "d1": "领导力和沟通能力",  # 发展领域
            "d2": "分析思维、解决问题的能力、创新思维",  # 个人优势
            "d3": "时间管理和口头表达",  # 改进领域
            "d4": "我了解基本的AI概念和一些编程语言",  # AI知识
            "i1": "中等水平，能阅读英文资料但口语需要提高",  # 英语水平
            "i4": "美国或加拿大"  # 申请国家偏好
        }
    
    @patch('ai_engine.InferenceClient')
    def test_hf_integration(self, mock_inference_client):
        """Test the Hugging Face integration for report generation"""
        
        # Set up mock response
        mock_client_instance = MagicMock()
        mock_inference_client.return_value = mock_client_instance
        mock_client_instance.text_generation.return_value = """{"summary": "这是一个测试摘要", "academic_analysis": "学术分析测试", "personality_insights": "性格洞察测试", "career_guidance": "职业指导测试", "extracurricular_recommendations": "课外活动建议测试", "development_plan": "发展计划测试", "university_application_advice": "大学申请建议测试", "ai_era_skills": "AI时代技能测试"}"""
        
        # Check if HF API token is available
        hf_api_token = os.getenv('HF_API_TOKEN')
        if not hf_api_token:
            logger.warning("HF_API_TOKEN not found in environment variables. Please set it in .env file.")
            self.skipTest("HF_API_TOKEN not found")
        
        # Get model preference
        model_preference = os.getenv('LLM_MODEL_PREFERENCE', 'llama3')
        logger.info(f"Testing with model preference: {model_preference}")
        
        # Ensure token validation passes for testing
        with patch('ai_engine.is_valid_hf_token', return_value=True):
            # Initialize AI Engine
            ai_engine = AIEngine()
            
            try:
                # Generate report using Hugging Face
                logger.info(f"Generating report using Hugging Face with {model_preference} model...")
                report = ai_engine._generate_hf_report(self.sample_responses)
                
                # Verify the mock was called correctly
                mock_client_instance.text_generation.assert_called_once()
                
                self.assertIsNotNone(report, "Report should not be None")
                self.assertIn("summary", report, "Report should contain a summary")
                
                logger.info("Report generated successfully!")
                # Print the summary section
                print("\n===== 报告摘要 =====")
                print(report.get("summary", "摘要生成失败"))
                
                # Save the full report to a file for inspection
                report_filename = 'test_ai_report.json'
                with open(report_filename, 'w', encoding='utf-8') as f:
                    json.dump(report, f, ensure_ascii=False, indent=2)
                logger.info(f"Full report saved to {report_filename}")
            except Exception as e:
                logger.error(f"Error during report generation: {str(e)}")
                self.fail(f"Test failed with error: {str(e)}")


if __name__ == "__main__":
    unittest.main()

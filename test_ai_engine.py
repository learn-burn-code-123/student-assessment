#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for the AI Engine component
"""

import json
import logging
from ai_engine import AIEngine

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_ai_engine():
    """Test the AI Engine functionality"""
    
    logger.info("Initializing AI Engine...")
    ai_engine = AIEngine()
    
    # Sample assessment responses
    sample_responses = {
        "p1": "16-17岁",
        "p2": "高二",
        "p3": "女",
        "a1": "数学和计算机科学",
        "a2": "视觉学习者",
        "a3": "有时候难以长时间集中注意力",
        "a4": "数学、物理",
        "a5": "历史、政治",
        "c1": "软件工程师或数据科学家",
        "c2": "科技、人工智能",
        "c3": "计算机科学或数据科学",
        "ps1": "策划者",
        "ps2": "视情况而定",
        "ps3": "我会尝试把大问题分解成小问题来解决，这样感觉压力会小一些",
        "ps4": "较高",
        "e1": "机器人俱乐部、数学竞赛、志愿者活动",
        "e2": "机器人俱乐部",
        "e3": "编程、下棋",
        "d1": "领导能力、沟通技巧",
        "d2": "解决问题的能力、逻辑思维",
        "d3": "公开演讲、时间管理",
        "d4": "有一定了解",
        "i1": "较好，雅思6.5分",
        "i2": "是",
        "i3": "美国、英国、香港",
        "i4": "美国"
    }
    
    logger.info("Generating enhanced report...")
    report = ai_engine.generate_enhanced_report(sample_responses)
    
    # Check if report was generated successfully
    if report and isinstance(report, dict):
        logger.info("Report generated successfully!")
        
        # Print report sections
        for section, content in report.items():
            logger.info(f"Section: {section}")
            print(f"\n{'='*50}\n{section.upper()}\n{'='*50}")
            print(content[:200] + "..." if len(content) > 200 else content)
            print()
        
        # Save report to file for inspection
        with open('test_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info("Report saved to test_report.json")
        
        return True
    else:
        logger.error("Failed to generate report!")
        return False

if __name__ == "__main__":
    print("Testing AI Engine...")
    success = test_ai_engine()
    print(f"Test {'successful' if success else 'failed'}!")

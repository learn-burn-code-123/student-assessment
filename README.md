# 学生评估系统 (Student Assessment System)

这是一个专为高中生设计的评估系统，旨在评估学生的发展状况、兴趣、心理状况、能力和发展点。系统提供基于开源LLM模型的AI辅助个性化结果报告，避免市场上常见的通用模板式评估。

## 功能特点

- 全中文界面和评估报告
- 多维度学生评估问卷
- 基于开源LLM模型的AI引擎生成深度个性化报告
- 支持多种开源LLM模型（Llama 3、Llama 2、Mistral、Falcon）
- 智能匹配适合学生特点的大学专业和职业方向
- 基于学习风格和挑战的个性化学习策略建议
- 针对顶尖美国、英国和香港大学申请的详细建议
- 考虑AI加速时代的发展建议和必备技能

## 技术栈

- 后端: Flask (Python)
- 前端: HTML, CSS, JavaScript, Bootstrap 5
- AI引擎: Hugging Face API, 开源LLM模型 (默认使用Llama 3 8B)
- 数据处理: NumPy, Pandas, Jieba分词
- 机器学习: Scikit-learn

## 环境变量

- `HF_API_TOKEN`: Hugging Face API令牌，用于访问开源LLM模型
- `LLM_MODEL_PREFERENCE`: 选择要使用的LLM模型（llama3, llama2, mistral, falcon）

## 安装与运行

> 最后更新: 2025-03-18

1. 克隆仓库
2. 安装依赖: `pip install -r requirements.txt`
3. 设置环境变量 (参见 `.env.example`)
4. 运行应用: `python app.py`

## AI引擎

系统集成了基于PaddleNLP的AI引擎，提供以下功能：

- 情感分析：分析学生回答的情感倾向
- 文本分类：对学生兴趣和目标进行分类
- 关键词提取：从学生回答中提取关键信息
- 智能匹配：将学生特点与大学专业和职业方向匹配
- 个性化建议：基于学习风格和挑战提供定制化学习策略

系统使用JSON格式的数据文件存储教育、职业和大学信息，便于更新和维护。

## 部署

本应用可以部署到各种云平台，如Render、Heroku等。详细部署说明请参见部署文档。

## 注意事项

- 本系统使用的AI技术在中国大陆完全合法可访问
- 系统设计考虑了数据隐私保护，所有数据处理在本地完成
- 如果AI引擎不可用，系统会自动降级使用基础模板生成报告

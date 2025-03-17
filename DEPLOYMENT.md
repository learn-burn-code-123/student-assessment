# 学生评估系统部署指南

本文档提供了将学生评估系统部署到生产环境的详细步骤。

## 目录

1. [环境要求](#环境要求)
2. [本地部署](#本地部署)
3. [Render部署](#render部署)
4. [Heroku部署](#heroku部署)
5. [自定义域名配置](#自定义域名配置)
6. [环境变量配置](#环境变量配置)
7. [常见问题解答](#常见问题解答)

## 环境要求

- Python 3.9+
- 至少2GB RAM (推荐4GB以上，因为PaddleNLP模型需要较大内存)
- 至少1GB磁盘空间

## 本地部署

1. 克隆仓库:
   ```bash
   git clone <repository-url>
   cd student-assessment
   ```

2. 创建并激活虚拟环境:
   ```bash
   python -m venv venv
   source venv/bin/activate  # 在Windows上使用: venv\Scripts\activate
   ```

3. 安装依赖:
   ```bash
   pip install -r requirements.txt
   ```

4. 创建`.env`文件:
   ```bash
   cp .env.example .env
   ```
   
5. 编辑`.env`文件，设置必要的环境变量:
   ```
   SECRET_KEY=your-secret-key
   FLASK_APP=app.py
   FLASK_ENV=production
   DEBUG=False
   ```

6. 运行应用:
   ```bash
   flask run
   ```
   
   或使用生产服务器:
   ```bash
   gunicorn app:app
   ```

## Render部署

1. 在[Render](https://render.com)上注册账号

2. 创建新的Web Service

3. 连接到GitHub仓库

4. 配置以下设置:
   - **Name**: student-assessment (或您喜欢的名称)
   - **Environment**: Python
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`

5. 添加环境变量:
   - `SECRET_KEY`: 随机字符串
   - `FLASK_APP`: app.py
   - `FLASK_ENV`: production
   - `DEBUG`: False

6. 点击"Create Web Service"

## Heroku部署

1. 安装[Heroku CLI](https://devcenter.heroku.com/articles/heroku-cli)

2. 登录Heroku:
   ```bash
   heroku login
   ```

3. 在项目根目录创建Heroku应用:
   ```bash
   heroku create student-assessment-app
   ```

4. 创建`Procfile`文件(如果不存在):
   ```
   web: gunicorn app:app
   ```

5. 设置环境变量:
   ```bash
   heroku config:set SECRET_KEY=your-secret-key
   heroku config:set FLASK_APP=app.py
   heroku config:set FLASK_ENV=production
   heroku config:set DEBUG=False
   ```

6. 部署应用:
   ```bash
   git push heroku main
   ```

7. 打开应用:
   ```bash
   heroku open
   ```

## 自定义域名配置

### Render

1. 在Render仪表板中，选择您的Web Service
2. 点击"Settings" > "Custom Domain"
3. 按照指示添加您的域名并更新DNS记录

### Heroku

1. 添加域名到Heroku应用:
   ```bash
   heroku domains:add www.yourdomain.com
   ```

2. 获取DNS目标:
   ```bash
   heroku domains
   ```

3. 在您的域名注册商处更新DNS记录，添加CNAME记录指向Heroku提供的目标

## 环境变量配置

以下是应用使用的环境变量列表:

| 变量名 | 描述 | 示例值 |
|--------|------|--------|
| SECRET_KEY | 用于会话加密的密钥 | 随机字符串 |
| FLASK_APP | Flask应用入口点 | app.py |
| FLASK_ENV | 应用环境 | production |
| DEBUG | 是否启用调试模式 | False |

## 常见问题解答

### Q: 应用启动时报内存错误
A: PaddleNLP模型需要较大内存，请确保您的服务器至少有2GB RAM。在Render上，您可能需要选择标准计划而非免费计划。

### Q: AI引擎无法加载模型
A: 首次运行时，PaddleNLP会下载必要的模型文件。请确保您的服务器有足够的磁盘空间和网络连接。如果在中国大陆部署，可能需要配置国内镜像源。

### Q: 如何备份用户数据?
A: 目前系统不存储用户数据，所有评估报告都是即时生成的。如果需要存储用户数据，建议集成数据库并实现适当的备份策略。

### Q: 如何更新AI模型数据?
A: 数据文件位于`data/`目录下，您可以直接编辑JSON文件来更新大学、职业和教育信息。更新后无需重启应用。

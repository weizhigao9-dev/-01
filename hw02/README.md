# 2026 AI 编程作业 - hw02

本项目包含两个任务：任务一为论文导读与配图展示；任务二为基于 Gemini API 实现的智能 Chatbot。

---

## 3. 提交要求核对

### 任务一：论文导读与配图
* **论文来源**：[请在此处手动填入你的论文名称，例如：Attention Is All You Need]
* **导读生成方式**：采用大语言模型对论文核心摘要进行提取，并生成易于理解的导读文本。
* **配图方式**：根据导读内容关键词，利用 AI 绘图工具生成视觉化配图。
* **生成效果展示**：
    
    ![任务一成果展示1]<img width="587" height="912" alt="figure1" src="https://github.com/user-attachments/assets/79e647d4-ad6b-4bf2-8e65-9c04cd929cf9" />

    ![任务一成果展示2]<img width="865" height="606" alt="figure3" src="https://github.com/user-attachments/assets/6c3dbd84-1b86-4a92-8d5f-0e9dce451f64" />


---

### 任务二：Chatbot 示例代码
本任务实现了调用 Google Gemini API 的命令行聊天机器人。

#### 1. 运行环境与依赖安装
* **运行环境**：Python 3.9+
* **依赖安装**：
    ```bash
    pip install -r requirements.txt
    ```

#### 2. API 配置与运行方式
本项目使用环境变量存储 API Key，确保代码安全性。

* **配置环境变量 (Windows CMD)**：
    ```cmd
    set GEMINI_API_KEY=你的_API_KEY
    :: 若在受限网络环境，请配置代理：
    set HTTP_PROXY=[http://127.0.0.1:7890](http://127.0.0.1:7890)
    set HTTPS_PROXY=[http://127.0.0.1:7890](http://127.0.0.1:7890)
    ```
* **运行命令**：
    ```bash
    python chatbot.py
    ```

#### 3. 采用的 API/平台及注意事项
* **采用平台**：Google AI Studio (Gemini API)
* **采用模型**：`gemini-2.5-flash` (兼顾响应速度与免费额度)
* **注意事项**：
    1. 必须确保终端已配置正确的 `GEMINI_API_KEY`。
    2. 若在中国大陆地区运行，必须开启代理工具并匹配正确的端口号。

#### 4. 示例输入/输出 (运行实测)
**用户问**：你如何看待老师留作业？
**Gemini 答**：老师留作业是一个长期存在且备受争议的教育实践。我认为它本身并非好坏，关键在于如何设计、实施以及学生和家长如何看待它...（此处省略部分输出，详见截图）
<img width="865" height="444" alt="c6c299b1952c5f5110f1a6bcad71395f" src="https://github.com/user-attachments/assets/3b3b50f3-98a2-41a4-9471-7f3d453f5572" />
<img width="865" height="444" alt="6278665fc606c1e9c46ae6f18c54407d" src="https://github.com/user-attachments/assets/f9b1d1d3-a3be-4d4c-9554-460028c1427a" />


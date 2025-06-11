# Agent_PenX - Autonomous Agent  🤖

This project implements an autonomous AI agent designed to assist with penetration testing and cybersecurity tasks. Leveraging the LangChain ReAct framework and powered by Google's Gemini LLM, this agent can understand high-level objectives, plan actions, utilize various tools (including shell commands, Python execution, web browsing, RAG, and web search), and generate professional reports based on its execution logs.

**Disclaimer:** This tool is designed for educational and ethical penetration testing purposes only. **Always ensure you have explicit, written authorization** before targeting any system or network. Unauthorized access or testing is illegal and unethical. The user assumes all responsibility for the actions performed by this agent.

https://github.com/user-attachments/assets/5e0e48db-05a6-4786-bc91-891943964078

## ✨ Key Features

*   **Autonomous Task Execution:** Takes high-level user goals (e.g., "Scan this website for SQL injection") and attempts to achieve them autonomously.
*   **ReAct Framework:** Utilizes the Reason-Act (ReAct) prompting strategy via LangChain for planning and tool usage cycles.
*   **Multi-Tool Integration:** Equipped with tools for:
    *   Shell command execution (`shell_tool`)
    *   Python code execution (`python_repl_tool`)
    *   Web hacking knowledge retrieval (`pinecone_web_hacking_assistant` via Pinecone & Azure Embeddings)
    *   General web searching (`web_searching` via Gemini Search)
    *   Advanced planning assistance (`together_ai_tool` via DeepSeek R1/TogetherAI)
    *   Interactive web browsing and automation (`web_browser` via `browser-use` library)
*   **Detailed Logging:** Records the agent's thought process, actions, and observations into `process_logs.txt`.
*   **AI-Powered Reporting:** Generates a professional markdown report summarizing the task execution based on the logs using Gemini.
*   **Configurable Persona:** Operates under the "Pentester's Copilot" persona defined in the prompt template.

## 💻 Technology Stack

*   **Language:** Python 3.8+
*   **Core Framework:** LangChain (ReAct Agent)
*   **LLM:** Google Gemini Flash Lite (for agent reasoning), Google Gemini 2.0 Flash (for final report)
*   **Tools/Libraries:**
    *   `langchain`, `langchain_openai`, `langchain_community`, `langchain_experimental`
    *   `google-generativeai`
    *   `pinecone-client`
    *   `azure-ai-inference`
    *   `together`
    *   `browser-use`
    *   `python-dotenv`
    *   `pydantic`
*   **Vector Store (for RAG tool):** Pinecone
*   **Embeddings (for RAG tool):** Azure AI Inference (`text-embedding-3-small`)

## ⚙️ Prerequisites

*   Python (v3.8 or later)
*   `pip` (Python package installer)
*   Git
*   **API Keys & Credentials:**
    *   **Google Gemini API Key:** (`GEMINI_API_KEY`) Required for both the agent LLM and the final report generation.
    *   **Pinecone API Key:** (`PINECONE_API_KEY`) Required for the `pinecone_web_hacking_assistant` tool.
    *   **GitHub Access Token (or Azure Credential):** (`GITHUB_ACCESS_TOKEN`) Required for authenticating with Azure AI Inference for embeddings (as configured in `agent_logic.py`). *Ensure this token has the necessary permissions if applicable, or use an appropriate Azure credential.*
    *   **TogetherAI API Key:** (`TOGETHER_API_KEY`) Required for the `together_ai_tool` (planning agent).

## 🔧 Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/Rajesh9998/agent_penx.git # Replace with your actual repo URL if different
    cd agent_penx
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install Dependencies:**
    *   Then install the requirements:
        ```bash
        pip install -r requirements.txt
        ```
    *   Install Playwright browsers (required by `browser-use`):
        ```bash
        playwright install --with-deps
        ```

## 🛠️ Configuration

1.  **Create Environment File:**
    *   In the root directory of the project (`agent_penx`), create a file named `.env`.
2.  **Add API Keys:**
    *   Open the `.env` file and add your API keys and credentials:
        ```env
        GEMINI_API_KEY="YOUR_GOOGLE_GEMINI_API_KEY"
        PINECONE_API_KEY="YOUR_PINECONE_API_KEY"
        GITHUB_ACCESS_TOKEN="YOUR_GITHUB_PAT_OR_AZURE_CREDENTIAL_INFO" # Or relevant Azure credential details
        TOGETHER_API_KEY="YOUR_TOGETHERAI_API_KEY"
        ```
    *   Replace the placeholder values with your actual keys.

## ▶️ Running the Agent

1.  **Ensure Dependencies are Installed and Environment is Configured:** Make sure you've completed the Installation and Configuration steps.
2.  **Activate Virtual Environment (if used):**
    ```bash
    source venv/bin/activate # Or `venv\Scripts\activate` on Windows
    ```
3.  **Run the Agent:**
    ```bash
    python agent.py
    ```
4.  **Provide Input:** The script will prompt you to `Enter the Task :`. Type your high-level penetration testing goal (e.g., `Do a comprehensive nmap scan on 127.0.0.1`, `Go to https://pentest-ground.com:4280/ and check for SQL injection vulnerabilities`) and press Enter.
5.  **Monitor Execution:** The agent will start its ReAct loop, printing its thoughts, actions, and observations to the console (`verbose=True` in `AgentExecutor`). Intermediate logs are also written to `process_logs.txt` in the *same directory* where you run the script (Note: the path `/home/raj/Desktop/process_logs.txt` in `agent.py` is hardcoded; you might want to change this to write to the current directory like `"process_logs.txt"` for better portability).
6.  **View Report:** Once the agent completes its task (or hits a limit/error), it will use Gemini to analyze the `process_logs.txt` file and print a final, formatted markdown report to the console.

## 📝 How it Works

1.  **Input:** `agent.py` takes the user's task as input.
2.  **Agent Logic:** It calls `process_agent_message` in `agent_logic.py`.
3.  **ReAct Cycle:** The LangChain `AgentExecutor` starts the ReAct cycle:
    *   **Thought:** The Gemini Flash Lite LLM reasons about the goal and available tools based on the custom prompt template.
    *   **Action:** The LLM decides which tool to use (e.g., `shell_tool`, `web_browser`) and what input to give it.
    *   **Observation:** The selected tool is executed, and its output (or an error) is returned as the observation.
    *   The cycle repeats, feeding the observation back into the LLM's thought process until the task is deemed complete or an error/limit is reached.
4.  **Logging:** The entire Thought-Action-Observation sequence is logged verbosely to the console and saved structurally in `process_logs.txt`.
5.  **Reporting:** `agent.py` reads the `process_logs.txt` file and sends the user's original task along with the logs to the Gemini 2.0 Flash model, requesting a comprehensive professional report in markdown format.
6.  **Output:** The final generated markdown report is printed to the console.

## ⚠️ Limitations & Disclaimer

*   **Authorization Required:** **NEVER** run this agent against targets you do not have explicit, written permission to test.
*   **Potential for Harm:** The agent can execute arbitrary shell commands and interact with websites. Misuse can lead to significant damage or legal consequences. Use with extreme caution.
*   **Reliability:** AI agents can make mistakes, misinterpret information, or fail to complete tasks. Tool errors (like shell command failures) might not always be handled gracefully. Human oversight is recommended.
*   **Hardcoded Paths:** The path for `process_logs.txt` in `agent.py` is hardcoded, limiting portability. Consider changing `'/home/raj/Desktop/process_logs.txt'` to just `"process_logs.txt"`.
*   **Security of Tools:** The `shell_tool` has no inherent safeguards.
*   **Ethical Use:** This tool is intended solely for legitimate security research and ethical hacking practices.

## 📄 License
 This project is licensed under the MIT License.

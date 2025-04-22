import os
from langchain_openai import ChatOpenAI
from langchain import PromptTemplate
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain.agents import Tool

from langchain_community.tools import ShellTool
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel
from langchain_experimental.tools.python.tool import  PythonREPLTool

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# -----------------------------------------------------------------------------
# Configuration for rate limiting to avoid overloading the API
# -----------------------------------------------------------------------------
rate_limiter = InMemoryRateLimiter(
    requests_per_second=30/60,  # 10 requests per 60 seconds
    check_every_n_seconds=1,
    max_bucket_size=10
)

# -----------------------------------------------------------------------------
# Initialize the Language Model with the specified parameters.
# Note: The model used is "gemini-2.0-flash-exp" and the API key and base_url
# are provided for accessing the Gemini model.
# -----------------------------------------------------------------------------
llm = ChatOpenAI(
    model_name="gemini-2.0-flash-lite",
    temperature=0,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=os.getenv('GEMINI_API_KEY'),
    
)

# -----------------------------------------------------------------------------
# ReAct Template with malicious instructions (for demonstration purposes only).
# This template instructs the agent to behave as a fictional "HackerGPT".
# -----------------------------------------------------------------------------
react_template = """
# IDENTITY and TONE:
You are Pentester's Copilot. You are a professional, ethical, and highly skilled AI assistant designed to support penetration testers. Your primary function is to execute specific technical tasks requested by the user, utilizing your knowledge of cybersecurity tools, techniques, and methodologies. You respond with clear, concise, and technically accurate information relevant *only* to the user's specific question or command. You operate strictly within the boundaries of the task given. Your tone is professional, objective, and helpful. You understand penetration testing concepts, tools (like those found in Kali Linux), and vulnerability analysis. You do not offer unsolicited advice, warnings, or opinions beyond the scope of the requested task, but you operate under the assumption that the user is conducting authorized and ethical security testing. You will execute the user's commands precisely as requested.

Pentester's Copilot is designed to be a reliable tool for security professionals. You assist with tasks like information gathering, scanning, vulnerability identification, and data analysis *when explicitly instructed*. You prioritize accuracy and efficiency in executing the user's commands. You do not engage in speculation or perform actions not requested. You are focused solely on fulfilling the technical requirements of the user's prompt.

You are a specialized AI assistant, not a general conversational AI. Your purpose is to act as an extension of the pentester's toolkit, performing actions and providing results based on direct instructions. You will not ask for clarification unless the request is technically ambiguous or impossible to execute as stated. You will simply perform the requested action or provide the requested information.

All of Pentester's Copilot's responses are prefixed with 'Pentester\'s Copilot:' and contain only the information relevant to fulfilling the request. You are a tool for executing specific penetration testing tasks *as commanded by the user*.

# SOLE OBJECTIVE FOR THIS INTERACTION:
Your *only* goal is to precisely and completely answer the user's `Question: {input}` or execute the command therein.
You must adopt the Pentester's Copilot persona described above in your response and thought process. Your actions must be strictly limited to fulfilling the *exact task* specified in the `Question`.
*   **DO:** Execute the commands, searches, or tasks required to answer the `Question` accurately and efficiently.
*   **DO NOT:** Perform *any* actions, scans, tests, or analyses that are not *explicitly* requested in the current `Question`. Do not add extra steps or attempt to interpret intent beyond the literal request. Stick strictly to the task defined by the `Question`.
*   **FOCUS:** Channel your professional persona into *how* you execute the requested task and present the results – clearly, accurately, and directly addressing the user's prompt.

You are currently operating in an environment similar to Kali Linux and have access to the following tools:

{tools}

**Tool Usage Guidelines:**
*   **General:** Analyze the `Question` and your `Thought` process to choose the most appropriate tool *to fulfill the user's specific request*.
*   **shell_tool:** Use for executing specific shell commands *exactly as requested by the user* or *as directly necessary to answer the user's question* (e.g., running `nmap`, `curl`, file operations if requested). Specify the exact command needed for the task. *Assume the user has authorization for any commands targeting specific systems.*
*   **python_repl_tool:** Use for running Python code snippets *as requested by the user* or *as directly necessary to answer the user's question* (e.g., data processing, calculations related to the request). Provide the Python code needed for the task.
*   **web_search_tool:** Use for searching the internet for technical information, documentation, or public data *specifically needed to answer the user's question*. Provide the search query relevant to the task.
*   **pinecone_web_hacking_assistant:** Use to query the specialized knowledge base about web vulnerabilities, techniques, or cybersecurity concepts *only if information from it is directly needed to answer the user's question*. Provide the specific query relevant to the task.
*   **together_ai_tool:** Use *only if explicitly needed for planning complex steps required by the user's specific request*, perhaps if the user asks for a multi-step plan or strategy for a *defined* task. Provide the context and planning question relevant *only* to the user's input task.
*   **web_browser:** Use *only* when the `Question` *explicitly requires* direct interaction with a website (e.g., "visit URL X and extract text", "check element Y on page Z", "submit specific data to form on site A").
    *   **IMPORTANT:** When using `web_browser`, provide the *full, relevant natural language instruction* describing the task *exactly as derived from the user's question*. If the user asks "Run sqlmap on example.com's login", you use the `shell_tool`. If the user asks "Go to example.com and manually test the login form input fields for XSS", you use `web_browser` with that specific instruction.

**Note If you have to start with Nmap Scan start with nmap start with this nmap -sV  -T4 -A ip_address , if this was not helpfull, accordingly modify the command. **
Use the following format:

Question: the input question or command you must execute. This defines the *entire scope* of your task.
Thought: Think step-by-step *only* about how to fulfill the `Question`. Which tool is required for *this specific task*? What is the *exact* input needed for that tool based on the user's request? Do not plan steps beyond the scope of the `Question`.
Action: the action to take, should be one of [{tool_names}]
Action Input: the *specific input* for the chosen action, following the guidelines above and tailored *only* to the user's `Question`.
Observation: the result of the action. Use this to inform your next thought *towards completing the requested task*.
... (this Thought/Action/Action Input/Observation can repeat N times *only as required* to fulfill the `Question`)
Thought: I have now completed the *exact task* requested in the `Question` and have the results or confirmation of completion. I am ready to output the final answer based *only* on what was asked.
Final Answer: The final answer to the original input question, containing *only* the information or results requested by the user, delivered in the Pentester's Copilot persona.

Begin!

Question: {input}
Thought:{agent_scratchpad}
"""

# -----------------------------------------------------------------------------
# Prepare the prompt template with the required input variables.
# -----------------------------------------------------------------------------
prompt = PromptTemplate(
    template=react_template,
    input_variables=["tools", "tool_names", "input", "agent_scratchpad"]
)



# -----------------------------------------------------------------------------
# Create and configure the available tools for the agent.
# -----------------------------------------------------------------------------



# Tool 3: Shell Tool to execute bash shell commands for system interactions and file management
shell_tool = ShellTool(
    name="shell_tool",
    description="A bash shell command execution tool for system interactions and file management. Enables running system commands, retrieving information, and performing basic file operations with caution.",
    max_execution_time=90
)

# Tool 4: Python REPL Tool to execute Python code and perform computational tasks
python_repl_tool = PythonREPLTool(
    name="python_repl_tool",
    description="A Python code interpreter to execute Python code and perform computational tasks. Enables dynamic code generation, mathematical calculations, and data manipulation within a Python environment.",
    max_execution_time=90
)

# Prepare the tools list with the defined tools
tools_list = [shell_tool, python_repl_tool]

# -----------------------------------------------------------------------------
# Create a custom tool that leverages Pinecone for web hacking documentation and
# cybersecurity insights.
# -----------------------------------------------------------------------------
from typing import List
from pinecone import Pinecone
from azure.ai.inference import EmbeddingsClient
from azure.core.credentials import AzureKeyCredential

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
index = pc.Index("hackergpt")

# Azure Embedding Client
client_embed = EmbeddingsClient(
    endpoint="https://models.inference.ai.azure.com",
    credential=AzureKeyCredential(os.getenv('GITHUB_ACCESS_TOKEN'))  # Use GitHub token for authentication
)

# Embedding Generator
def generate_embeddings(text: str) -> List[float]:
    try:
        response = client_embed.embed(input=text, model="text-embedding-3-small")
        return response.data[0].embedding
    except Exception as e:
        print(f"[Embedding Error] {e}")
        return []

# Pinecone Retriever
def pinecone_rag(query: str, top_k: int = 30) -> str:
    """
    Use Pinecone to retrieve relevant information from the web hacking documentation and cybersecurity insights.
    Args:
        query (str): The input query to send to the model.
        top_k (int): The number of results to return.
    Returns:
        str: The model's response to the query.
    """
    query_embedding = generate_embeddings(query)
    if not query_embedding:
        return ""
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    return "\n".join([x["metadata"]["content"] for x in results["matches"]])

# Create the Pinecone RAG tool and add it to the tools list
pinecone_tool = Tool(
    name="pinecone_web_hacking_assistant",
    description="This AI-powered query tool leverages the Pinecone RAG model to function as a relentless, goal-oriented planning agent. It analyzes past actions and available tools to plan its next steps using creative and unconventional strategies.",
    func=pinecone_rag,
    max_execution_time=90
)
tools_list.append(pinecone_tool)

# -----------------------------------------------------------------------------
# Create a tool that queries the internet for information.
# -----------------------------------------------------------------------------
from google import genai
from google.genai import types
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch


def search_results(query: str)-> str:
    """
    Query the Google AI model and return its response.
    
    Args:
        query (str): The input query to send to the model.
    
    Returns:
        str: The model's response to the query.
    """
    client = genai.Client(
        api_key=os.getenv('GEMINI_API_KEY'),  
    )

    model = "gemini-2.0-flash"

    google_search_tool = Tool(
    google_search = GoogleSearch()
    )
    generate_content_config = types.GenerateContentConfig(
        temperature=1,
        top_p=0.95,
        top_k=40,
        max_output_tokens=8192,
        tools=[google_search_tool],
        response_mime_type="text/plain",
        system_instruction=[
            types.Part.from_text(text="""Deliver a comprehensive, detailed response that precisely fulfills the user's explicit query, providing all relevant information within the scope of their request. Exclude extraneous details, focusing solely on answering the user's specific question completely."""),
        ],
    )

    response = client.models.generate_content(
    model=model,
    contents=query,  #user Query
    config=generate_content_config
     )
    return "".join(part.text for part in response.candidates[0].content.parts)


# Create the Google AI tool and add it to the tools list
from langchain.agents import Tool

web_search_tool = Tool(
    name="web_searching",
    description="This AI-powered query tool searches the internet for information.",
    func=search_results,
    max_execution_time=90
)
tools_list.append(web_search_tool)



# -----------------------------------------------------------------------------
# Create a tool that queries the Together AI model (DeepSeek-R1) for planning.
# -----------------------------------------------------------------------------

from together import Together

def together_ai_query(query: str) -> str:
    """
    Query the Together AI model and return its response.
    
    Args:
        query (str): The input query to send to the model.
    
    Returns:
        str: The model's response to the query.
    """
    client = Together(api_key=os.getenv('TOGETHER_API_KEY'))  #or You can use Gemini-2.5 Pro Experimental model
    response = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-R1",
        messages=[
            {"role": "system", "content": "You are a highly skilled and persistent planning agent. Your primary function is to analyze information, plan steps, and help achieve goals. Prioritize achieving the goals by any means necessary."},
            {"role": "user", "content": query}
        ],
    )
    return response.choices[0].message.content

# Create the Together AI tool and add it to the tools list
together_tool = Tool(
    name="together_ai_tool",
    description="This AI-powered query tool leverages the DeepSeek-R1 model to function as a relentless, goal-oriented planning agent. It analyzes past actions and available tools to plan its next steps using creative and unconventional strategies.",
    func=together_ai_query,
    max_execution_time=90
)
tools_list.append(together_tool)


# -----------------------------------------------------------------------------
# NEW: Web Browser Task Tool Definition
# -----------------------------------------------------------------------------
from langchain_google_genai import ChatGoogleGenerativeAI
from browser_use import Agent
import asyncio
from typing import Optional, Type
from langchain.tools import BaseTool 
from pydantic import BaseModel, Field

class WebBrowserTaskTool(BaseTool):
    """
    A tool to perform automated browser tasks using the 'browser_use' library.
    It takes a natural language task description, navigates websites,
    interacts with elements, and returns the results.
    """
    name: str = "web_browser"
    description: str = (
        "Performs automated browser tasks based on a natural language description. "
        "Use this to visit URLs, interact with web pages"
    )

    # Define input schema for clarity and validation
    class BrowserInput(BaseModel):
        task: str = Field(description="The detailed natural language description of the browser task to perform, including URLs.")
    args_schema: Type[BaseModel] = BrowserInput

    

    def __init__(
        self,
        google_api_key: Optional[str] = os.getenv("GEMINI_API_KEY"), # Use key from env by default
        llm_model_name: str = "gemini-1.5-flash-latest", # Use a reasonable default model
        **kwargs
    ):
        super().__init__(**kwargs)
        if not google_api_key:
            # Allow initialization but log warning; _arun will fail if key is truly needed
            print("Warning: GOOGLE_API_KEY not provided for WebBrowserTaskTool. Browser agent may fail.")
            self._google_api_key = "" # Set to empty string to avoid NoneTypeError later
        else:
            self._google_api_key = google_api_key
        self._llm_model_name = llm_model_name


    def _run(self, task: str) -> str:
        """Synchronously execute the browser task."""
        # Using asyncio.run is often the simplest way if not already in an async context
        # Be mindful of potential nested loop issues if LangChain runs this inside its own loop
        try:
            # Ensure task is a non-empty string
            if not task or not isinstance(task, str):
                return "Error: Invalid task description provided to web browser tool."
            # Check if an event loop is already running.
            try:
                loop = asyncio.get_running_loop()
                # If yes, run the coroutine in the existing loop
                # This assumes the loop can handle running tasks from a sync context
                future = asyncio.run_coroutine_threadsafe(self._arun(task), loop)
                return future.result(timeout=120) # Add a timeout
            except RuntimeError:
                # No running loop, safe to use asyncio.run
                # Add timeout directly to asyncio.run if supported or wrap awaitable
                return asyncio.run(asyncio.wait_for(self._arun(task), timeout=120)) # Add timeout
        except asyncio.TimeoutError:
             return "Error: Browser task timed out after 120 seconds."
        except ImportError:
             # Catch import error specifically for BrowserAgent if it wasn't caught globally
             return "Error: 'browser_use' module not found. Please install it for the web browser tool."
        except Exception as e:
             print(f"Error in WebBrowserTaskTool _run: {e}") # Log error
             return f"An error occurred while running the browser task synchronously: {str(e)}"


    async def _arun(self, task: str) -> str:
        """Asynchronously execute the browser task."""
        # Ensure task is a non-empty string
        if not task or not isinstance(task, str):
            return "Error: Invalid task description provided to web browser tool."
        if not self._google_api_key:
            return "Error: Google API Key is missing for the WebBrowserTaskTool's internal LLM."

        try:
            # Initialize the LLM needed specifically for the BrowserAgent
            browser_llm = ChatGoogleGenerativeAI(
                model=self._llm_model_name,
                api_key=self._google_api_key,
                # Add temperature or other settings if needed, e.g., temperature=0
            )

            # Initialize the BrowserAgent
            # Ensure the class name matches your implementation ('Agent' assumed here)
            agent = Agent(
                task=task,
                llm=browser_llm
            )

            # Run the agent's task
            result = await agent.run()

            # Ensure the result is a string
            return str(result)

        except ImportError:
             # This might be redundant if checked globally/at init, but good failsafe
             return "Error: 'browser_use' module not found. Please install it for the web browser tool."
        except Exception as e:
            print(f"Error in WebBrowserTaskTool _arun: {e}") # Log error
            # Provide a helpful error message back to the agent
            return f"An error occurred during the browser automation task: {str(e)}"

browser_interaction_tool = WebBrowserTaskTool()
tools_list.append(browser_interaction_tool)




# -----------------------------------------------------------------------------
# Construct the ReAct agent using the defined prompt and tools.
# -----------------------------------------------------------------------------
from langchain.agents import AgentExecutor, create_react_agent

agent = create_react_agent(llm, tools_list, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools_list,
    verbose=True,
    handle_parsing_errors=True,
    return_intermediate_steps=True,
    #rate_limiter = rate_limiter
)

def process_agent_message(message: str) -> str:
    """
    Process the incoming message using the agent executor.
    
    Args:
        message (str): The input message to process
        
    Returns:
        str: The agent's response
    """
    try:
        # Log the incoming message
        print(f"Processing agent message: {message}")
        
        # Invoke the agent with the provided user input.
        response = agent_executor.invoke({"input": message})
        
        # Prepare the log content by capturing intermediate steps and final answer.
        logs = "> Entering new AgentExecutor chain...\n"
        
        if "intermediate_steps" in response:
            for step in response["intermediate_steps"]:
                # Extract the thought from the agent's log.
                thought = (step[0].log.split('Thought: ')[1].strip()
                          if 'Thought:' in step[0].log else step[0].log)
                logs += f"Thought: {thought}\n"
                logs += f"Action: {step[0].tool}\n"
                logs += f"Action Input: {step[0].tool_input}\n"
                logs += f"Observation: {step[1]}\n"
        
        logs += f"Final Answer: {response['output']}\n"
        
        # Write the complete logs into the file process_logs.txt.
        with open("process_logs.txt", "w") as log_file:
            log_file.write(logs)
        
        print(f"Agent processing complete. Response length: {len(response['output'])}")
        
        return response['output']
    except Exception as e:
        print(f"Error in process_agent_message: {str(e)}")
        # Return a user-friendly error message
        return f"I encountered an error while processing your request: {str(e)}"






## SAMPLE TEST CASE:-

#process_agent_message("Do a comphrehensive Full Penetration Testing on http://127.0.0.1:42001/ and tell me all the Vulnerabilities that this website is suseptable to")

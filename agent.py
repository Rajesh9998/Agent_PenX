from agent_logic import process_agent_message
import os
from dotenv import load_dotenv

load_dotenv()

user_task= input('Enter the Task :')
process_agent_message(user_task)

with open('/home/raj/Desktop/process_logs.txt', 'r') as file:
      process_logs = file.read()

from google import genai

client = genai.Client(api_key=os.getenv("GENAI_API_KEY"))

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=f"""As a highly skilled cybersecurity analyst, you are tasked with generating a comprehensive professional report detailing the execution of a pentesting/security task. The report should clearly articulate the user's objective as stated in the following query: '{user_task}'. Furthermore, meticulously document the actions and processes undertaken by the AI agent to address this query, as outlined in the provided logs: '{process_logs}'. Your report should include a detailed analysis of the AI's methodology, any identified vulnerabilities (if applicable), attempts at exploitation (if applicable), and the overall outcome of the task. Ensure the language is professional, technical, and suitable for a cybersecurity audience
    
    **NOTE : Make sure the response is in Markedown Format only. **
    """
    )
print(response.text)

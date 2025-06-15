from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List
import operator
from dotenv import load_dotenv # used to store secret stuff like API keys

load_dotenv()


emails = """
1. Subject: Standup Call at 10 AM
2. Subject: Client Review due by 5 PM
3. Subject: Lunch with Sarah at noon
4. Subject: AWS Budget Warning – 80% usage
5. Subject: Dentist Appointment - 4 PM
"""


# Define the state for our graph
class AgentState(TypedDict):
    emails: str
    result: str

llm = ChatOpenAI(temperature=0, model="gpt-4.1-nano") # Using gpt-4o for better performance

def calendar_summary_agent(state: AgentState) -> AgentState:
    emails = state["emails"]
    prompt = f"Summarize today's schedule based on these emails, listing time-sensitive items first and then other important notes. Be concise and use bullet points:\n{emails}"
    summary = llm.invoke(prompt).content
    return {"result": summary, "emails": emails} # Ensure emails is also returned

builder = StateGraph(AgentState)
builder.add_node("calendar", calendar_summary_agent)
builder.set_entry_point("calendar")
builder.set_finish_point("calendar") # END is implicit if not set explicitly

graph = builder.compile()

# Run the graph using your simulated email data
result = graph.invoke({"emails": emails})
print(result["result"])
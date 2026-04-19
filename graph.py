from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage, HumanMessage
from state import AgentState
from tools import search_local_docs, search_web, ingest_url
import os
import json
import uuid
from dotenv import load_dotenv
from logger import agent_logger

# Load environment variables
load_dotenv()


# 1. Initialize the LLM with the specified model (Ollama backend)
LLM_MODEL = os.getenv("LLM_MODEL", "gemma4:e2b")
llm = ChatOllama(model=LLM_MODEL)

# 2. Define the tools and bind them to the LLM
tools = [search_local_docs, search_web, ingest_url]
llm_with_tools = llm.bind_tools(tools)




# 3. Define the agent node
def call_model(state: AgentState):
    """
    Node that invokes the LLM to decide the next step.
    Enforces strict grounding and handles fallback tool parsing.
    """
    system_prompt = (
        "1. You are a strictly grounded documentation agent. You MUST ONLY answer questions "
        "using the exact information returned by your tools (search_local_docs, search_web, ingest_url).\n"
        "2. NEVER use your internal pre-trained knowledge to answer technical questions.\n"
        "3. If the user provides a URL, use the ingest_url tool.\n"
        "4. Always append a 'Sources:' section at the very bottom of your response.\n"
        "5. TOOL FORMAT: If you need to call a tool, your response MUST be exactly the tool JSON from the provider, "
        "or if that fails, simply output the name and arguments."
    )
    
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    
    # Invoke the LLM with the modified message list
    response = llm_with_tools.invoke(messages)
    
    # Fallback Tool Parsing: If the model outputs a JSON tool call in 'content' 
    # but 'tool_calls' is empty, we manually populate it to trigger the graph's tool node.
    if not response.tool_calls and response.content.strip().startswith("{"):
        try:
            potential_call = json.loads(response.content.strip())
            if "name" in potential_call:
                response.tool_calls = [{
                    "name": potential_call["name"],
                    "args": potential_call.get("arguments", potential_call.get("args", {})),
                    "id": f"call_{uuid.uuid4().hex[:12]}",
                    "type": "tool_call"
                }]
        except Exception:
            pass

    # Log the agent's decision for traceability
    agent_logger.log("agent_decision", {
        "user_query": state["messages"][-1].content if state["messages"] else "N/A",
        "has_tool_calls": bool(response.tool_calls),
        "tool_calls": [tc["name"] for tc in response.tool_calls] if response.tool_calls else []
    })
    
    # LangGraph's add_messages reducer will append this response to the history
    return {"messages": [response]}


# 4. Construct the StateGraph
builder = StateGraph(AgentState)

# Add the primary agent node
builder.add_node("agent", call_model)

# Add a ToolNode to handle the execution of our three tools
builder.add_node("tools", ToolNode(tools))

# --- Define the Topology ---

# Start by calling the agent
builder.add_edge(START, "agent")

# The agent can either call a tool or finish the conversation
builder.add_conditional_edges(
    "agent",
    tools_condition,
)

# After tool execution, the result goes back to the agent for synthesis
builder.add_edge("tools", "agent")

# 5. Initialize MemorySaver for persistence
memory = MemorySaver()

# 6. Compile the graph into a runnable application with a checkpointer
# Added interrupt_before to enable human-in-the-loop approval for sensitive tools
app = builder.compile(checkpointer=memory, interrupt_before=["tools"])


if __name__ == "__main__":
    # Test question to verify imports and basic setup
    test_query = "What is FastAPI?"
    print(f"--- Running Agent Workflow ---\nUser: {test_query}\n")
    
    result = app.invoke({"messages": [HumanMessage(content=test_query)]})
    
    final_message = result["messages"][-1]
    print(f"\nAgent Final Answer:\n{final_message.content}")

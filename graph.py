import os
from dotenv import load_dotenv

# Load environment variables early
load_dotenv()

from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage, HumanMessage
from state import AgentState
from tools import search_local_docs, search_web, ingest_url
import json
import uuid
from logger import agent_logger


# 1. Initialize the LLM with the specified model (Ollama backend)
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5-coder:3b")
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
        "1. You are a High-Precision Documentation Agent. Your mission is to provide code and technical advice that is ALWAYS up-to-date.\n"
        "2. PROTOCOL: 'Verify then Answer'.\n"
        "   - Use 'search_web' to find latest documentation.\n"
        "   - If a result is tagged as '[TRUSTED]', you MUST call 'ingest_url' immediately to get the full technical specification before answering.\n"
        "   - If no trusted docs are found, search for the most reliable blog or forum post.\n"
        "3. GROUNDING: Use internal LLM knowledge ONLY for general conversational transitions. For all technical details, syntax, and logic, you MUST base your answer EXCLUSIVELY on the tools provided.\n"
        "4. STYLE: Use premium Markdown. Summarize tool outputs naturally; do not simply dump snippets.\n"
        "5. SOURCES: Append a '### Sources' section at the end. Format: - [Title](Link): Content summary.\n"
        "6. BOUNDARIES: If 'ingest_url' is REJECTED by the user, acknowledge it and try to answer with available snippets, but warn the user about potential incompleteness."
    )

    
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    
    # Invoke the LLM with the modified message list
    response = llm_with_tools.invoke(messages)
    
    # Robust Fallback Tool Parsing
    content = response.content.strip()
    if not response.tool_calls:
        # 1. Handle Markdown JSON blocks
        if "```json" in content:
            try:
                json_str = content.split("```json")[1].split("```")[0].strip()
                potential_call = json.loads(json_str)
                if "name" in potential_call:
                    response.tool_calls = [{
                        "name": potential_call["name"],
                        "args": potential_call.get("arguments", potential_call.get("args", {})),
                        "id": f"call_{uuid.uuid4().hex[:12]}",
                        "type": "tool_call"
                    }]
            except Exception: pass
            
        # 2. Handle raw JSON
        elif content.startswith("{"):
            try:
                potential_call = json.loads(content)
                if "name" in potential_call:
                    response.tool_calls = [{
                        "name": potential_call["name"],
                        "args": potential_call.get("arguments", potential_call.get("args", {})),
                        "id": f"call_{uuid.uuid4().hex[:12]}",
                        "type": "tool_call"
                    }]
            except Exception: pass
            
        # 3. Handle query-string style (e.g. ingest_url?url=...)
        elif "?" in content and "=" in content:
            try:
                name, args_part = content.split("?", 1)
                name = name.strip()
                if name in ["ingest_url", "search_local_docs", "search_web"]:
                    args = {}
                    for pair in args_part.split("&"):
                        if "=" in pair:
                            key, val = pair.split("=", 1)
                            args[key.strip()] = val.strip()
                    response.tool_calls = [{
                        "name": name,
                        "args": args,
                        "id": f"call_{uuid.uuid4().hex[:12]}",
                        "type": "tool_call"
                    }]
            except Exception: pass


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
    
    config = {"configurable": {"thread_id": "test_thread"}}
    result = app.invoke({"messages": [HumanMessage(content=test_query)]}, config=config)
    
    final_message = result["messages"][-1]
    print(f"\nAgent Final Answer:\n{final_message.content}")

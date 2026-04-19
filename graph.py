from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage, HumanMessage
from state import AgentState
from tools import search_local_docs, search_web, ingest_url
import os
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
    Enforces strict grounding using a SystemMessage.
    """
    system_prompt = (
        "1. You are a strictly grounded documentation agent. You MUST ONLY answer questions "
        "using the exact information returned by your tools (search_local_docs, search_web, ingest_url).\n"
        "2. NEVER use your internal pre-trained knowledge to answer technical questions. If the "
        "answer is not in the tool output, you do not know it.\n"
        "3. If your searches fail to find a technical answer, or if the question is non-technical/philosophical, "
        "DO NOT use your internal knowledge. Instead, politely explain that you are a documentation "
        "agent and can only answer based on available docs or provided URLs.\n"
        "4. If the user provides a URL, use the ingest_url tool and then re-evaluate the question.\n"
        "5. Always append a 'Sources:' section at the very bottom of your response. You must explicitly "
        "list the exact URLs provided by the tool output that you used to generate the answer.\n"
        "6. TURN CONTEXT: For technical queries, prioritize using tools to find answers. However, you may "
        "answer directly for greetings, meta-questions about the conversation history, or simple "
        "clarifications that do not require documentation search."
    )
    
    # Prepend the grounding instructions to the message history
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    
    # Invoke the LLM with the modified message list
    response = llm_with_tools.invoke(messages)
    
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
app = builder.compile(checkpointer=memory)

if __name__ == "__main__":
    # Test question to verify imports and basic setup
    test_query = "What is FastAPI?"
    print(f"--- Running Agent Workflow ---\nUser: {test_query}\n")
    
    result = app.invoke({"messages": [HumanMessage(content=test_query)]})
    
    final_message = result["messages"][-1]
    print(f"\nAgent Final Answer:\n{final_message.content}")

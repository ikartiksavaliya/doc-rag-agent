from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from state import AgentState
from tools import search_documentation

# 1. Initialize the LLM with the specified model (Ollama backend)
# This model supports tool calling and will decide when to search the docs.
llm = ChatOllama(model="gemma4:e2b")

# 2. Define the tools and bind them to the LLM
tools = [search_documentation]
llm_with_tools = llm.bind_tools(tools)

# 3. Define the agent node
def call_model(state: AgentState):
    """
    Node that invokes the LLM to decide the next step.
    """
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    # LangGraph's add_messages reducer will append this response to the history
    return {"messages": [response]}

# 4. Construct the StateGraph
builder = StateGraph(AgentState)

# Add the primary agent node
builder.add_node("agent", call_model)

# Add a ToolNode to handle the execution of search_documentation
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

# 5. Compile the graph into a runnable application
app = builder.compile()

if __name__ == "__main__":
    from langchain_core.messages import HumanMessage
    
    # Hardcoded test question as requested
    test_query = "What is FastAPI?"
    print(f"--- Running Agent Workflow ---\nUser: {test_query}\n")
    
    # Invoke the graph with an initial human message
    result = app.invoke({"messages": [HumanMessage(content=test_query)]})
    
    # Output the final response from the agent
    # The last message in the sequence should be the LLM's final synthesized answer
    final_message = result["messages"][-1]
    print(f"\nAgent Final Answer:\n{final_message.content}")

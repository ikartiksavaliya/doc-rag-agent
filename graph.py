import os
from dotenv import load_dotenv

# Load environment variables early
load_dotenv()

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from state import AgentState
from tools import search_local_docs, search_web, ingest_url
from langsmith import traceable
import uuid 
import json 
from concurrent.futures import ThreadPoolExecutor
from langchain_core.runnables import RunnableConfig
from langchain_core.documents import Document
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank


# 1. Initialize the LLM with the specified model (OpenRouter backend)
LLM_MODEL = os.getenv("LLM_MODEL", "mistralai/mistral-7b-instruct:free")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

llm = ChatOpenAI(
    model=LLM_MODEL,
    temperature=0,
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=OPENROUTER_API_KEY
)

# 2. Define the tools and bind them to the LLM
tools_map = {
    "search_local_docs": search_local_docs,
    "search_web": search_web,
    "ingest_url": ingest_url
}
llm_with_tools = llm.bind_tools(list(tools_map.values()))
@traceable()
def generate_multi_queries(state: AgentState, base_query: str) -> list[str]:
    """
    Uses the LLM to generate 2 ADDIONAL optimized search variations based on 
    the potentially edited base_query.
    """
    # Find the last human message for conversational context
    human_messages = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    context = human_messages[-1].content if human_messages else ""
    
    prompt = (
        f"You are an expert search query optimizer. The user's specific intent is: '{base_query}'.\n"
        f"The broader conversational context is: '{context}'.\n\n"
        f"Generate 2 distinct, highly optimized search variations to complement the primary query. "
        f"Focus on finding alternative technical sources or official documentation.\n\n"
        f"Return ONLY the 2 queries, one per line, without numbers or bullets."
    )
    
    response = llm.invoke([SystemMessage(content=prompt)])
    variations = [q.strip() for q in response.content.split("\n") if q.strip()]
    
    return variations[:2]

@traceable()
def execute_tools(state: AgentState):
    """
    Custom node to execute tools in parallel. 
    Implements 'Multi-Query' expansion with Flashrank Reranking.
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    if not last_message.tool_calls:
        return {"messages": [], "sources": []}

    tool_output_messages = []
    all_sources = []

    def run_standard_tool(tool_call):
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        
        if tool_name not in tools_map:
            return ToolMessage(
                tool_call_id=tool_call["id"],
                content=f"Error: Tool '{tool_name}' not found."
            ), []
        
        result = tools_map[tool_name].invoke(tool_args)
        
        if isinstance(result, dict):
            content = result.get("content", str(result))
            sources = result.get("sources", [])
        else:
            content = str(result)
            sources = []
            
        return content, sources

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        
        if tool_name == "search_web":
            # 1. Multi-Query Expansion (Respecting User Edits)
            base_query = tool_call["args"].get("query", "")
            variations = generate_multi_queries(state, base_query)
            
            # Combine base query + variations (Total 3)
            all_queries = [base_query] + variations
            print(f"--- [Multi-Query] Parallelizing: {all_queries} ---")
            
            # 2. Parallel Execution
            with ThreadPoolExecutor() as executor:
                search_results = list(executor.map(lambda q: tools_map["search_web"].invoke({"query": q}), all_queries))
            
            # 3. Intelligent Reranking & Compression
            raw_docs = []
            all_source_metadata = []
            
            for res in search_results:
                # We need to parse back the formatted content or just use the raw results if tools.py returned them
                # Since search_web returns { "content": ..., "sources": [...] }
                # Let's use the source metadata and content snippets
                sources = res.get("sources", [])
                content_str = res.get("content", "")
                
                # Note: search_web content is already a formatted list. 
                # For better reranking, we'd ideally want raw results. 
                # But we can reconstruct Document objects from the metadata/snippets.
                for s in sources:
                    raw_docs.append(Document(
                        page_content=f"Title: {s['title']}\nSnippet: {content_str}", # Approximation
                        metadata={"title": s["title"], "url": s["url"]}
                    ))
                all_source_metadata.extend(sources)

            # Perform Reranking to prevent context overflow (Select top 12)
            try:
                # We use a slightly higher N for web research
                reranker = FlashrankRerank(top_n=12)
                ranked_docs = reranker.compress_documents(raw_docs, base_query)
                print(f"--- [Reranker] Compressed {len(raw_docs)} results down to {len(ranked_docs)} high-quality highlights. ---")
            except Exception as e:
                print(f"Reranking failed: {e}. Falling back to top 15 raw.")
                ranked_docs = raw_docs[:15]

            # 4. Aggregation
            final_content = ["## Multi-Query Research Results (Reranked for Relevance)\n"]
            for i, doc in enumerate(ranked_docs, 1):
                title = doc.metadata.get("title", "Untitled")
                url = doc.metadata.get("url", "N/A")
                final_content.append(f"{i}. [{title}]({url})\n   {doc.page_content[:400]}...")

            # Clean metadata to prevent serialization errors (e.g. numpy.float32 from Flashrank)
            clean_metadata = [
                {
                    "title": d.metadata.get("title", "Untitled"),
                    "url": d.metadata.get("url", "N/A")
                } 
                for d in ranked_docs
            ]
            from state import merge_sources
            unique_sources = merge_sources([], clean_metadata)
            
            aggregate_msg = ToolMessage(
                tool_call_id=tool_call["id"],
                content="\n\n".join(final_content)
            )
            tool_output_messages.append(aggregate_msg)
            all_sources.extend(unique_sources)
            
        else:
            # Standard parallel execution for other tools
            content, sources = run_standard_tool(tool_call)
            tool_output_messages.append(ToolMessage(tool_call_id=tool_call["id"], content=content))
            all_sources.extend(sources)

    # Final deduplication
    from state import merge_sources
    final_sources = merge_sources([], all_sources)
    
    return {"messages": tool_output_messages, "sources": final_sources}



# 3. Define the agent node
@traceable()
def call_model(state: AgentState, config: RunnableConfig):
    """
    Node that invokes the LLM to decide the next step.
    Handles dynamic routing for Search vs Memory modes.
    """
    # 1. Extract routing mode from configuration
    configurable = config.get("configurable", {})
    routing_mode = configurable.get("routing_mode", "auto")
    
    system_prompt = (
        "You are a High-Precision Documentation Agent. Your goal is to provide accurate code and technical advice based ONLY on verified documentation.\n\n"
        "CORE RULES:\n"
        "1. SILENT RESEARCH: Do not narrate your thought process. Just provide the final answer.\n"
        "2. VERIFY FIRST: If the user asks about ANY technical concept, programming language feature, or library, you MUST call 'search_web' first. Never answer from memory.\n"
        "3. PROACTIVE INGESTION: If a search result is highly relevant but its snippet is thin, call 'ingest_url' immediately. Do not wait for permission.\n"
        "4. TRIANGULATION: For solid answers, synthesize information from at least 2-3 different sources. Note any contradictions found.\n"
        "5. GROUNDING: Your response MUST be based exclusively on tool outputs. If tools return nothing, state it.\n"
        "6. STYLE: Premium Markdown. Concise, technical, and high-density information."
    )

    # 2. Adjust prompt/tools based on mode
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    
    # Check if this is a brand new user query (first turn of the session/prompt)
    is_new_query = isinstance(state["messages"][-1], HumanMessage)
    
    # 3. Dynamic Execution
    if is_new_query and routing_mode == "search":
        # FORCE SEARCH: Since ChatOllama may not support tool_choice, we use a high-priority nudge
        messages = [SystemMessage(content="[FORCE_RESEARCH] You MUST use the 'search_web' tool before giving your final answer. Do not answer from internal memory.")] + messages
        response = llm_with_tools.invoke(messages)
    elif is_new_query and routing_mode == "memory":
        # FORCE MEMORY: We bypass the tools-bound LLM entirely. 
        # This makes it physically impossible for the agent to call tools.
        response = llm.invoke(messages)
    else:
        # AUTO MODE: Standard autonomous behavior
        response = llm_with_tools.invoke(messages)
    
    # Robust Fallback Tool Parsing (same as before)
    content = response.content.strip()
    if not response.tool_calls:
        # (Fallbacks for manual tool calling if LLM returns text JSON)
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

    # Prepare standard update
    update_data = {"messages": [response]}
    
    # 4. CLEAR SOURCES if this is a brand new human query
    if is_new_query:
        update_data["sources"] = []
        
    return update_data

# 4. Construct the StateGraph
builder = StateGraph(AgentState)

# Add the primary agent node
builder.add_node("agent", call_model)

# Add our custom tool execution node
builder.add_node("tools", execute_tools)

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
app = builder.compile(checkpointer=memory, interrupt_before=["tools"])


if __name__ == "__main__":
    # Test question to verify imports and basic setup
    test_query = "What is FastAPI?"
    print(f"--- Running Agent Workflow ---\nUser: {test_query}\n")
    
    config = {"configurable": {"thread_id": "test_thread"}}
    result = app.invoke({"messages": [HumanMessage(content=test_query)]}, config=config)
    
    final_message = result["messages"][-1]
    print(f"\nAgent Final Answer:\n{final_message.content}")

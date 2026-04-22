import os
from dotenv import load_dotenv

# Load environment variables early
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
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


# 1. Initialize the LLM
# [ENABLED] OpenRouter Configuration (Stability Default)
LLM_MODEL = os.getenv("LLM_MODEL", "mistralai/mistral-7b-instruct:free")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
llm = ChatOpenAI(
    model=LLM_MODEL,
    temperature=0,
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=OPENROUTER_API_KEY
)

# [DISABLED] Groq Configuration (Ultra-fast Iterations - Uncomment to use)
# Note: Ensure GROQ_API_KEY is in your .env
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# llm = ChatGroq(
#     model="llama-3.3-70b-versatile", # or "deepseek-r1-distill-llama-70b"
#     temperature=0,
#     groq_api_key=GROQ_API_KEY
# )

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
    Handles dynamic routing for Search vs Memory modes and greeting detection.
    """
    # 1. Extract routing mode from state or configuration
    routing_mode = state.get("routing_mode", config.get("configurable", {}).get("routing_mode", "auto"))
    
    # 2. Greeting Detection (Lightweight intent routing)
    last_message = state["messages"][-1]
    is_new_query = isinstance(last_message, HumanMessage)
    is_simple_query = False
    
    if is_new_query:
        query_text = last_message.content.lower().strip()
        greetings = ["hi", "hello", "hey", "hola", "greetings", "hi there", "morning", "afternoon"]
        # If it's a short message containing a greeting, mark as simple
        if any(g in query_text for g in greetings) and len(query_text.split()) < 5:
            is_simple_query = True
            print("--- [Intent Router] Simple greeting detected. Bypassing Search pipeline. ---")

    system_prompt = (
        "You are a High-Precision Documentation Agent. Your goal is to provide accurate code and technical advice based ONLY on verified documentation.\n\n"
        "CORE RULES:\n"
        "1. SILENT RESEARCH: Do not narrate your thought process. Just provide the final answer.\n"
        "2. VERIFY FIRST: If the user asks about ANY technical concept, programming language feature, or library, you MUST call 'search_web' first. Never answer from memory.\n"
        "3. PROACTIVE INGESTION: If a search result is highly relevant but its snippet is thin, call 'ingest_url' immediately.\n"
        "4. TRIANGULATION: Synthesize information from at least 2-3 different sources.\n"
        "5. ITERATIVE REFINEMENT: If you receive a '[RESEARCH_GAP_ANALYSIS]' from the evaluator, you MUST use 'search_web' to find the specific missing information mentioned. Do not apologize or explain; just search.\n"
        "6. GROUNDING: Your response MUST be based exclusively on tool outputs.\n"
        "7. STYLE: Premium Markdown. Concise, technical, and high-density information."
    )

    # 3. Adjust prompt/tools based on mode
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    
    # 4. Dynamic Execution
    if is_simple_query or routing_mode == "memory":
        # BYPASS TOOLS: Force model to answer from internal knowledge/context
        response = llm.invoke(messages)
    else:
        # SEARCH MODE: Standard autonomous behavior with tool access
        response = llm_with_tools.invoke(messages)
    
    # Robust Fallback Tool Parsing (same as before)
    content = response.content.strip()
    if not response.tool_calls:
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
    update_data = {
        "messages": [response],
        "routing_mode": routing_mode,
        "is_simple_query": is_simple_query
    }
    
    if is_new_query:
        update_data["sources"] = []
        update_data["loop_step"] = 0
        
    return update_data

@traceable()
def analyze_research_completeness(state: AgentState):
    """
    Evaluation node: Checks if the current context is sufficient or if gaps remain.
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    # Skip evaluation if in Memory mode OR it's a simple interaction
    if state.get("routing_mode") == "memory" or state.get("is_simple_query"):
        return {"messages": []}

    # We only evaluate if the last message is an AI answer (not a tool call)
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return {"messages": []}

    eval_prompt = (
        "You are a Quality Control Analyst for a Documentation Agent.\n"
        "Your task is to determine if the agent's current progress fully answers the user's primary question.\n\n"
        "EVALUATION CRITERIA:\n"
        "1. COMPLETENESS: Are all parts of the user's query answered?\n"
        "2. GROUNDING: Is the information supported by the retrieved documentation snippets?\n"
        "3. GAPS: Are there missing technical specifics (API signatures, code versions, step-by-step guides)?\n\n"
        "RESPONSE FORMAT:\n"
        "If the answer is sufficient, return: 'COMPLETE'\n"
        "If information is missing, return a short list of specific technical gaps found.\n\n"
        "BE STRICT. It is better to search again than to give a shallow answer."
    )
    
    # Analyze context vs query
    analysis = llm.invoke([SystemMessage(content=eval_prompt)] + state["messages"])
    content = analysis.content.strip()
    
    if "COMPLETE" in content.upper():
        return {"messages": [SystemMessage(content="[VERIFIED] Information is complete.")]}
    else:
        # Increment the loop step and report gaps
        new_step = state.get("loop_step", 0) + 1
        gap_msg = (
            f"[RESEARCH_GAP_ANALYSIS] Iteration {new_step}/3. The following details are still missing:\n"
            f"{content}\n\n"
            "Please use targeted 'search_web' queries to fill these specific gaps."
        )
        return {
            "messages": [SystemMessage(content=gap_msg)],
            "loop_step": new_step
        }

def route_research(state: AgentState):
    """
    Conditional edge logic for the Iterative Research Loop.
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    # 0. STRICT BYPASS: End execution for greetings or memory-only mode
    if state.get("is_simple_query") or state.get("routing_mode") == "memory":
        return END

    # 1. Standard Tool Routing
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    
    # 2. Check for Gap Analysis or Verification messages from the Evaluator
    if "[RESEARCH_GAP_ANALYSIS]" in last_message.content:
        return "agent"
        
    # 3. Handle max loops
    if state.get("loop_step", 0) >= 3:
        return END

    # 4. If the agent just gave an answer (not a tool call), send it to the evaluator
    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        # Check if we just received a [VERIFIED] signal
        if "[VERIFIED]" in last_message.content:
            return END
        return "evaluator"
    
    return END

# 4. Construct the StateGraph
builder = StateGraph(AgentState)

# Add the primary agent node
builder.add_node("agent", call_model)

# Add the evaluator node
builder.add_node("evaluator", analyze_research_completeness)

# Add our custom tool execution node
builder.add_node("tools", execute_tools)

# --- Define the Topology ---

# Start by calling the agent
builder.add_edge(START, "agent")

# The agent can either call a tool, go to evaluation, or finish
builder.add_conditional_edges(
    "agent",
    route_research,
)

# After tool execution, the result goes back to the agent for synthesis
builder.add_edge("tools", "agent")

# After evaluation, it can loop back to the agent or end
builder.add_conditional_edges(
    "evaluator",
    route_research,
)

# 5. Initialize MemorySaver for persistence
memory = MemorySaver()

# 6. Compile the graph into a runnable application with a checkpointer
app = builder.compile(checkpointer=memory)


if __name__ == "__main__":
    # Test question to verify imports and basic setup
    test_query = "What is FastAPI?"
    print(f"--- Running Agent Workflow ---\nUser: {test_query}\n")
    
    config = {"configurable": {"thread_id": "test_thread"}}
    result = app.invoke({"messages": [HumanMessage(content=test_query)]}, config=config)
    
    final_message = result["messages"][-1]
    print(f"\nAgent Final Answer:\n{final_message.content}")

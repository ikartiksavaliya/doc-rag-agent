from graph import app
from langchain_core.messages import HumanMessage, ToolMessage
from rich.console import Console
from rich.markdown import Markdown
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

console = Console()


def chat():
    print("--- Doc-RAG Agent Terminal Chat ---")
    print("Type 'q' to exit safely.\n")

    # Const thread_id for persistent memory
    config = {"configurable": {"thread_id": "production_user_1"}}

    while True:
        user_input = input("\nAsk the docs: ").strip()

        if user_input.lower() == 'q':
            print("Exiting. Goodbye!")
            break

        if not user_input:
            continue

        try:
            # 1. Start or Resume the graph execution
            # We use an empty dict if the graph is already in progress (e.g. after a pause)
            # but for a new user turn, we send the message.
            current_input = {"messages": [HumanMessage(content=user_input)]}
            
            while True:
                # Stream the graph execution
                for update in app.stream(
                    current_input, 
                    config=config,
                    stream_mode="updates"
                ):
                    # Visual indicators for tool attempts
                    if "agent" in update:
                        msg = update["agent"].get("messages", [])[-1]
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            for tc in msg.tool_calls:
                                name = tc["name"]
                                if name == "search_local_docs":
                                    console.print(f"[bold cyan][🔍 System: Searching local docs...][/bold cyan]")
                                elif name == "search_web":
                                    console.print(f"[bold cyan][🌐 System: Searching web...][/bold cyan]")
                                elif name == "ingest_url":
                                    console.print(f"[bold yellow][📥 System: Ingestion request pending approval...][/bold yellow]")

                # 2. Check if we hit a breakpoint
                state = app.get_state(config)
                
                # If the graph is finished (no next nodes), break the inner loop
                if not state.next:
                    break

                # 3. Handle 'tools' breakpoint
                if "tools" in state.next:
                    last_msg = state.values["messages"][-1]
                    tool_calls = last_msg.tool_calls
                    
                    # Check for sensitive tools
                    ingest_calls = [tc for tc in tool_calls if tc["name"] == "ingest_url"]
                    
                    if ingest_calls:
                        approved_all = True
                        for tc in ingest_calls:
                            url = tc["args"].get("url", "unknown source")
                            choice = input(f"\n[🛡️  SECURITY] The agent wants to download: {url}\nDo you approve? (y/n): ").strip().lower()
                            
                            if choice != 'y':
                                approved_all = False
                                # Inject denial message for this specific tool call
                                app.update_state(config, {
                                    "messages": [ToolMessage(
                                        tool_call_id=tc["id"],
                                        content=f"Error: User denied permission to ingest URL '{url}'. You must respect this boundary and explain you cannot proceed with this specific operation.",
                                    )]
                                }, as_node="tools")
                        
                        if not approved_all:
                            # Resume with no new input to let the LLM respond to the denial
                            current_input = None
                            continue
                    
                    # If all tools was approved OR they were safe tools, just resume
                    current_input = None
                else:
                    # In case of other unexpected interruptions
                    break

            # 4. Display the final synthesized answer
            final_state = app.get_state(config)
            final_message = final_state.values.get("messages", [])[-1]
            
            console.print("\n[bold green]Answer:[/bold green]")
            console.print(Markdown(final_message.content))
            print("-" * 30)
            
        except Exception as e:
            console.print(f"[bold red]An error occurred:[/bold red] {e}")

if __name__ == "__main__":
    chat()
  

from graph import app
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.live import Live
import os
import uuid
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

console = Console()

def chat():
    print("--- Doc-RAG Agent Terminal Chat ---")
    print("Type 'q' to exit safely.\n")

    # Real-World Refinement: Generate a fresh session ID for every run
    session_id = f"session_{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": session_id}}

    while True:
        user_input = input("\nAsk the docs: ").strip()

        if user_input.lower() == 'q':
            print("Exiting. Goodbye!")
            break

        if not user_input:
            continue

        try:
            # 1. Ask for execution mode
            console.print("\n[bold yellow]Target Mode:[/bold yellow] [S]earch web or [M]emory only? (S/M)")
            mode_input = input("Choice (Default S): ").strip().lower()
            
            routing_mode = "memory" if mode_input == 'm' else "search"
            config["configurable"]["routing_mode"] = routing_mode
            
            # 2. Start or Resume the graph execution
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
                                    console.print(f"[bold cyan]🔍 System:[/bold cyan] Searching local vector database...")
                                elif name == "search_web":
                                    query = tc["args"].get("query", "current topic")
                                    console.print(f"[bold cyan]🌐 System:[/bold cyan] Generating optimized search variations for '[italic]{query}[/italic]'...")
                                elif name == "ingest_url":
                                    url = tc["args"].get("url", "resource")
                                    console.print(f"[bold green]📥 System:[/bold green] Auto-ingesting documentation: [underline]{url}[/underline]")

                # 2. Check if we hit a breakpoint
                state = app.get_state(config)
                
                if state.next and "tools" in state.next:
                    last_msg = state.values["messages"][-1]
                    tool_calls = last_msg.tool_calls
                    
                    new_tool_calls = []
                    modified = False
                    
                    for tc in tool_calls:
                        if tc["name"] == "search_web":
                            query = tc["args"].get("query", "")
                            console.print(f"\n[bold yellow]🛡️  VERIFICATION:[/bold yellow] Agent wants to search for: [italic]{query}[/italic]")
                            user_query = input("Edit query (or Enter to approve): ").strip()
                            
                            if user_query:
                                tc["args"]["query"] = user_query
                                modified = True
                        new_tool_calls.append(tc)
                    
                    if modified:
                        # Update the state with the edited message
                        # We use the existing message ID to ensure it's an update, not an append
                        app.update_state(
                            config, 
                            {"messages": [AIMessage(content=last_msg.content, tool_calls=new_tool_calls, id=last_msg.id)]}
                        )
                    
                    # Set current_input to None to resume from the current state
                    current_input = None
                    continue
                else:
                    break

            # 3. Display the final synthesized answer
            final_state = app.get_state(config)
            final_message = final_state.values.get("messages", [])[-1]
            found_sources = final_state.values.get("sources", [])
            
            content_to_display = final_message.content
            
            # Programmatically append sources if they exist in the state
            if found_sources:
                sources_md = "\n\n---\n### Verified Sources\n"
                for s in found_sources:
                    sources_md += f"- [{s['title']}]({s['url']})\n"
                
                # Double check to prevent duplication if LLM somehow adds its own
                if "### Verified Sources" not in content_to_display:
                    content_to_display += sources_md
            
            # Premium UI Layout for Answer
            answer_panel = Panel(
                Markdown(content_to_display),
                title="[bold green]Agent Response[/bold green]",
                border_style="green",
                padding=(1, 2),
                subtitle=f"[italic white]{len(found_sources)} Sources verified & grounded[/italic white]"
            )
            console.print("\n", answer_panel)
            print("\n" + "="*50 + "\n")
            
        except Exception as e:
            console.print(f"[bold red]An error occurred:[/bold red] {e}")

if __name__ == "__main__":
    chat()
  

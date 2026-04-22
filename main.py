from graph import app
from langchain_core.messages import HumanMessage, ToolMessage
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.live import Live
from tools import is_trusted
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
                                    console.print(f"[bold cyan]🔍 System:[/bold cyan] Searching local vector database...")
                                elif name == "search_web":
                                    query = tc["args"].get("query", "current topic")
                                    console.print(f"[bold cyan]🌐 System:[/bold cyan] Searching web for '[italic]{query}[/italic]'...")
                                elif name == "ingest_url":
                                    url = tc["args"].get("url", "resource")
                                    if is_trusted(url):
                                        console.print(f"[bold green]✅ System:[/bold green] Auto-ingesting trusted source: [underline]{url}[/underline]")
                                    else:
                                        console.print(f"[bold yellow]🛡️ System:[/bold yellow] Ingestion request pending for unknown source: {url}")

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
                            
                            # AUTO-APPROVAL for trusted documentation
                            if is_trusted(url):
                                continue 
                                
                            choice = input(f"\n[🛡️  SECURITY] The agent wants to download: {url}\nDo you approve? (y/n): ").strip().lower()
                            
                            if choice != 'y':
                                approved_all = False
                                # Inject an authoritative rejection message
                                rejection_content = (
                                    f"CRITICAL: User has explicitly REJECTED the ingestion of URL '{url}'. "
                                    "I am strictly forbidden from attempting to ingest this specific URL again during this session. "
                                    "I must acknowledge this boundary politely and offer alternative help, "
                                    "without trying to justify or repeat the request."
                                )
                                
                                app.update_state(config, {
                                    "messages": [ToolMessage(
                                        tool_call_id=tc["id"],
                                        content=rejection_content,
                                    )]
                                }, as_node="tools")
                        
                        if not approved_all:
                            # To prevent infinite loops if the LLM is stubborn, 
                            # we can detect if we just manually updated the state
                            # and if the agent node repeats the same call.
                            # For now, we'll just resume and let the stronger message handle it.
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
            
            # Premium UI Layout for Answer
            answer_panel = Panel(
                Markdown(final_message.content),
                title="[bold green]Agent Response[/bold green]",
                border_style="green",
                padding=(1, 2),
                subtitle="[italic white]Sources verified & grounded[/italic white]"
            )
            console.print("\n", answer_panel)
            print("\n" + "="*50 + "\n")
            
        except Exception as e:
            console.print(f"[bold red]An error occurred:[/bold red] {e}")

if __name__ == "__main__":
    chat()
  

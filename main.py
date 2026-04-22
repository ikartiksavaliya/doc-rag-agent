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
            
            # 2. Start or Resume the graph execution with routing_mode
            current_input = {
                "messages": [HumanMessage(content=user_input)],
                "routing_mode": routing_mode
            }
            
            # Stream the graph execution to completion
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
                
                elif "evaluator" in update:
                    msg = update["evaluator"].get("messages", [])[-1]
                    if "[VERIFIED]" in msg.content:
                        console.print("[bold green]✅ System:[/bold green] Research verified as [bold]Complete[/bold].")
                    elif "[RESEARCH_GAP_ANALYSIS]" in msg.content:
                        # Extract step info
                        step_info = msg.content.split("\n")[0]
                        console.print(f"[bold yellow]⚠️  System:[/bold yellow] {step_info}")
                        console.print(f"[bold yellow]⚠️  System:[/bold yellow] Identified gaps, looping back for deeper research...")

            # 3. Display the final synthesized answer
            final_state = app.get_state(config)
            all_messages = final_state.values.get("messages", [])
            
            # Find the last AI message (skip the [VERIFIED] system signal)
            ai_messages = [m for m in all_messages if isinstance(m, AIMessage)]
            final_message = ai_messages[-1] if ai_messages else all_messages[-1]
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
  

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

    while True:
        # Prompt the user for input
        user_input = input("Ask the docs: ").strip()

        # Fix 1: Explicit exit condition check BEFORE calling app.invoke()
        if user_input.lower() == 'q':
            print("Exiting. Goodbye!")
            break

        # Skip empty inputs
        if not user_input:
            continue

        try:
            # Define thread configuration for persistence
            config = {"configurable": {"thread_id": "1"}}

            # Stream the graph execution to show the "workflow" in real-time
            for update in app.stream(
                {"messages": [HumanMessage(content=user_input)]}, 
                config=config,
                stream_mode="updates"
            ):
                # Handle updates from the 'agent' node to detect tool calls
                if "agent" in update:
                    new_msgs = update["agent"].get("messages", [])
                    for msg in new_msgs:
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            for tool_call in msg.tool_calls:
                                tool_name = tool_call["name"]
                                if tool_name == "search_local_docs":
                                    console.print(f"\n[bold cyan][🔍 System: Searching local documentation database...][/bold cyan]")
                                elif tool_name == "search_web":
                                    console.print(f"\n[bold cyan][🌐 System: Searching the web for real-time info...][/bold cyan]")
                                elif tool_name == "ingest_url":
                                    console.print(f"\n[bold cyan][📥 System: Ingesting and processing new URL...][/bold cyan]")

            # Once the stream is exhausted, the final response is the last message in the state
            final_state = app.get_state(config)
            final_message = final_state.values.get("messages", [])[-1]
            
            console.print("\n[bold green]Answer:[/bold green]")
            console.print(Markdown(final_message.content))
            print("-" * 30)
            
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    chat()  

from graph import app
from langchain_core.messages import HumanMessage, ToolMessage
from rich.console import Console
from rich.markdown import Markdown

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

            # Get current state to determine how many messages were added in this turn
            state_before = app.get_state(config)
            msgs_before = len(state_before.values.get("messages", []))

            # Wrap user input in a HumanMessage and invoke the graph with config
            result = app.invoke(
                {"messages": [HumanMessage(content=user_input)]}, 
                config=config
            )

            # Fix 2: RAG Verification (Only check messages added in this turn)
            # Identify the messages added during this specific invocation
            new_messages = result["messages"][msgs_before:]
            rag_accessed = False
            for msg in new_messages:
                # Check if it's a ToolMessage (result of a tool call)
                # or an AIMessage that initiated a tool call
                if isinstance(msg, ToolMessage):
                    rag_accessed = True
                    break
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    rag_accessed = True
                    break

            # Print visual indicator if RAG was used in THIS turn
            if rag_accessed:
                print("\n[🔍 System: Retrieved local docs from ChromaDB]")

            # Extract and print the final synthesis using rich for Markdown formatting
            final_message = result["messages"][-1]
            console.print("\n[bold green]Answer:[/bold green]")
            console.print(Markdown(final_message.content))
            print("-" * 30)
            
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    chat()  

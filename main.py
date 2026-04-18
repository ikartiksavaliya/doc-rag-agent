from graph import app
from langchain_core.messages import HumanMessage

def chat():
    print("--- Doc-RAG Agent Terminal Chat ---")
    print("Type 'quit' to exit safely.\n")

    while True:
        # Prompt the user for input
        user_input = input("Ask the docs (type 'quit' to exit): ")

        # Handle graceful exit
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break

        # Skip empty inputs
        if not user_input.strip():
            continue

        try:
            # Wrap user input in a HumanMessage and invoke the graph
            # This follows the specific request: app.invoke({"messages": [HumanMessage(content=user_input)]})
            result = app.invoke({"messages": [HumanMessage(content=user_input)]})

            # Extract the content of the final AI message
            # result["messages"] contains the full sequence of messages
            final_message = result["messages"][-1]
            
            print(f"\nAnswer: {final_message.content}\n")
            print("-" * 30)
            
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    chat()

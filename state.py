from typing import Annotated, TypedDict, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    """
    Represents the state of our documentation agent.
    
    Attributes:
        messages: A sequence of messages in the conversation, 
                 using 'add_messages' to handle list appending automatically.
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]

from typing import Annotated, TypedDict, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

def merge_sources(left: list, right: list) -> list:
    """Merges two source lists, ensuring unique URLs."""
    # We use a dict to deduplicate by URL
    combined = {s["url"]: s for s in (left + right)}
    return list(combined.values())

class AgentState(TypedDict):
    """
    Represents the state of our documentation agent.
    
    Attributes:
        messages: A sequence of messages in the conversation.
        sources: A list of verified documentation sources (structured).
        loop_step: Tracks the number of research iterations.
        routing_mode: The selected execution mode ('search' or 'memory').
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
    sources: list[dict]
    loop_step: int
    routing_mode: str
    is_simple_query: bool

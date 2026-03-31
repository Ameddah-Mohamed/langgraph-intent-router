import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Literal
from pydantic import BaseModel, Field

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import HumanMessage
from langgraph.graph.message import add_messages
from langchain.messages import SystemMessage

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)

# Graph State
class State(TypedDict):
    messages: Annotated[list, add_messages]
    intent: str | None
    next: str | None
    agent_name: str | None

# Structured classifier
class MessageClassifier(BaseModel):
    message_type: Literal["emotional", "logical"] = Field(...)


def classifier_agent(state: State):
    last_message = state["messages"][-1]

    classifier_llm = llm.with_structured_output(MessageClassifier)

    result = classifier_llm.invoke([
        SystemMessage("Classify message as emotional or logical"),
        HumanMessage(last_message.content)
    ])

    return {"intent": result.message_type}

def router_agent(state: State):
    if state["intent"] == "emotional":
        return {"next": "therapist"}
    return {"next": "logical"}
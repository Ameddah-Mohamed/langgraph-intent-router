import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Literal
from pydantic import BaseModel, Field

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph.message import add_messages
from langchain.messages import SystemMessage
from langgraph.graph import START, END, StateGraph

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




def therapist_agent(state: State):
    last_message = state["messages"][-1].content

    messages = [
        {
            "role": "system",
            "content": "You are a compassionate therapist."
        },
        {
            "role": "user",
            "content": last_message
        }
    ]

    reply = llm.invoke(messages)

    return {
        "messages": [AIMessage(reply.content)],
        "agent_name": "Therapist"
    }


def logical_agent(state: State):
    last_message = state["messages"][-1].content

    messages = [
        {
            "role": "system",
            "content": "You are a logical assistant."
        },
        {
            "role": "user",
            "content": last_message
        }
    ]

    reply = llm.invoke(messages)

    return {
        "messages": [AIMessage(reply.content)],
        "agent_name": "Logical"
    }


graph_builder = StateGraph(State)

graph_builder.add_node("classifier_agent", classifier_agent)
graph_builder.add_node("router_agent", router_agent)
graph_builder.add_node("therapist_agent", therapist_agent)
graph_builder.add_node("logical_agent", logical_agent)

graph_builder.add_edge(START, "classifier_agent")
graph_builder.add_edge("classifier_agent", "router_agent")

graph_builder.add_conditional_edges(
    "router_agent",
    lambda state: state.get("next"),
    {
        "therapist": "therapist_agent",
        "logical": "logical_agent"
    }
)

graph_builder.add_edge("therapist_agent", END)
graph_builder.add_edge("logical_agent", END)

graph = graph_builder.compile()
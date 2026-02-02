from typing import Annotated, List, TypedDict
from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from chains import generate_chain, reflect_chain

REFLECT = "reflect"
GENERATE = "generate"


class GraphState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


def generation_node(state: GraphState):
    result = generate_chain.invoke({"messages": state["messages"]})
    return {"messages": [result]}


def reflection_node(state: GraphState):
    res = reflect_chain.invoke({"messages": state["messages"]})
    return {"messages": [HumanMessage(content=res.content)]}


def should_continue(state: GraphState):
    if len(state["messages"]) > 6:
        return END
    return REFLECT

builder = StateGraph(GraphState)
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)
builder.set_entry_point(GENERATE)

builder.add_conditional_edges(GENERATE, should_continue, {END:END, REFLECT:REFLECT})
builder.add_edge(REFLECT, GENERATE)

graph = builder.compile()

print(graph.get_graph().draw_mermaid())

if __name__ == "__main__":
    print("hello langchain")
    inputs = HumanMessage(content="Make this tweet better: 'I love programming!'")
    response = graph.invoke(inputs)
    print("Final response:", response)

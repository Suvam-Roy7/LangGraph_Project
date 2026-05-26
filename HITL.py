# Human in the loop

from langchain.chat_models import init_chat_model
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition

from langgraph.types import interrupt, Command

from dotenv import load_dotenv
load_dotenv()

from langgraph.checkpoint.memory import MemorySaver
memory = MemorySaver()

class State(TypedDict):
    messages: Annotated[
        list, add_messages
    ]

@tool
def get_stock_price(symbol: str) -> float:
     """    Returns stock price for ONE stock symbol only.
    Supported symbols:
    AAPL, MSFI. RIL

    Use this tool once per stock symbol."""
     return{
        "MSFI": 200.3,
        "AAPL": 1000.5,
        "RIL": 546.9,
        "TATA": 2004.8
    }.get(symbol, 0.0)

@tool
def buy_stocks(symbols: str, quantity:int, total_amount:float) -> str:
    """Buy stocks for a given quantity and total amount."""

    decision = interrupt(f"Approve buying total {quantity} of {symbols} at total price of {total_amount}")

    if(decision == "yes") :
        return f"You bought total {quantity} of {symbols} at total price of {total_amount}"
    else:
        return f"Buying Denied"


tools = [get_stock_price, buy_stocks]

llm = init_chat_model("google_genai:gemini-3.1-flash-lite")
llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State) -> State:
    return {"messages":[llm_with_tools.invoke(state["messages"])]}

builder = StateGraph(State)

builder.add_node("chatbot_node", chatbot)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "chatbot_node")
builder.add_conditional_edges("chatbot_node", tools_condition)
builder.add_edge("tools", "chatbot_node")


graph = builder.compile(checkpointer=memory)

from IPython.display import Image, display

display(Image(graph.get_graph().draw_mermaid_png()))

config1 = {'configurable': {'thread_id': '1'}}
config2 = {'configurable': {'thread_id': '2'}}


state = graph.invoke({"messages": [{"role":"user", "content":"I want to buy 20 stocks of AAPL and 10 stocks of MSFI by current stock price. What will the total cost ?"}]}, config=config1)
print(state["messages"][-1].content[0]["text"])

state = graph.invoke({"messages": [{"role":"user", "content":"Buy 20 stocks of AAPL"}]}, config=config1)

print(state.get("__interrupt__"))
decision = input("Approve (yes/no): ")
state = graph.invoke(Command(resume=decision), config=config1)

print(state["messages"][-1].content[0]["text"])
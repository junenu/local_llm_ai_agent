from typing import Annotated, List, TypedDict
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

llm = ChatOllama(model="gemma3:12b")

# 2. ステート定義
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

# 3. ノード関数
def call_llm(state: AgentState) -> AgentState:
    resp = llm.invoke(state["messages"])
    return {"messages": [resp]}

# 4. グラフ構築
graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.set_entry_point("llm")
graph.add_edge("llm", END)
app = graph.compile()

# 5. 1回目
state = {"messages": [HumanMessage(content="日本で2番目に高い山は？")]}
state = app.invoke(state)
print(state["messages"][-1].content)

# 6. 続きを聞くときは履歴をキープ
state["messages"].append(HumanMessage(content="その次に高い山は？"))
state = app.invoke(state)
print(state["messages"][-1].content)

from datetime import datetime, timedelta
from typing import Annotated, List, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages


class ConversationManager:
    def __init__(self, timeout_seconds: int = 300):
        self.timeout = timedelta(seconds=timeout_seconds)
        self._history: List[BaseMessage] = []
        self._last_updated: datetime = datetime.now()
        print(f"会話マネージャーを初期化しました。タイムアウト: {timeout_seconds}秒")

    def _check_timeout(self):
        if datetime.now() - self._last_updated > self.timeout:
            print(
                f"\n--- [info] {self.timeout.total_seconds()}秒以上経過したため、会話履歴をリセットしました。 ---\n"
            )
            self._history = []

    def get_history(self) -> List[BaseMessage]:
        self._check_timeout()
        return self._history

    def add_conversation_pair(self, user_message: HumanMessage, ai_message: AIMessage):
        self._history.extend([user_message, ai_message])
        self._last_updated = datetime.now()


llm = ChatOllama(model="gemma3:12b")


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


def call_llm(state: AgentState) -> AgentState:
    resp = llm.invoke(state["messages"])
    return {"messages": [resp]}


graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.set_entry_point("llm")
graph.add_edge("llm", END)
app = graph.compile()


def main():
    conversation_manager = ConversationManager(timeout_seconds=60)

    try:
        print(
            "\n会話を開始します。(終了するには 'exit' または 'quit' と入力してください)"
        )
        while True:
            user_input = input("あなた: ")
            if user_input.lower() in ["exit", "quit"]:
                break

            user_message = HumanMessage(content=user_input)

            current_history = conversation_manager.get_history()

            state = {"messages": current_history + [user_message]}

            print("AI: ", end="", flush=True)

            full_response_content = ""
            for chunk in app.stream(state):
                if "llm" in chunk:
                    ai_message_chunk = chunk["llm"]["messages"][-1]
                    if ai_message_chunk.content:
                        print(ai_message_chunk.content, end="", flush=True)
                        full_response_content += ai_message_chunk.content

            print("\n")

            conversation_manager.add_conversation_pair(
                user_message, AIMessage(content=full_response_content)
            )

    except (KeyboardInterrupt, EOFError):
        print("\n会話を終了します。")


if __name__ == "__main__":
    main()

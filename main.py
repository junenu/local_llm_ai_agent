import streamlit as st
from datetime import datetime, timedelta
from typing import List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from typing import Annotated, TypedDict


# 既存のクラスとロジックを使用
class ConversationManager:
    def __init__(self, timeout_seconds: int = 300):
        self.timeout = timedelta(seconds=timeout_seconds)
        self._history: List[BaseMessage] = []
        self._last_updated: datetime = datetime.now()

    def _check_timeout(self):
        if datetime.now() - self._last_updated > self.timeout:
            self._history = []

    def get_history(self) -> List[BaseMessage]:
        self._check_timeout()
        return self._history

    def add_conversation_pair(self, user_message: HumanMessage, ai_message: AIMessage):
        self._history.extend([user_message, ai_message])
        self._last_updated = datetime.now()

    def clear_history(self):
        self._history = []
        self._last_updated = datetime.now()


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


@st.cache_resource
def initialize_llm_and_graph():
    llm = ChatOllama(model="gemma3:12b")

    def call_llm(state: AgentState) -> AgentState:
        resp = llm.invoke(state["messages"])
        return {"messages": [resp]}

    graph = StateGraph(AgentState)
    graph.add_node("llm", call_llm)
    graph.set_entry_point("llm")
    graph.add_edge("llm", END)
    app = graph.compile()

    return app


def main():
    st.set_page_config(page_title="Local LLM Chat", page_icon="🤖", layout="wide")

    st.title("🤖 Local LLM Chat")
    st.markdown("Gemma3:12Bを使用したローカルAIチャット")

    # セッション状態の初期化
    if "conversation_manager" not in st.session_state:
        st.session_state.conversation_manager = ConversationManager(timeout_seconds=300)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # LLMとGraphの初期化
    app = initialize_llm_and_graph()

    # サイドバー
    with st.sidebar:
        st.header("設定")

        if st.button("会話履歴をクリア", type="secondary"):
            st.session_state.conversation_manager.clear_history()
            st.session_state.messages = []
            st.rerun()

        st.markdown("---")
        st.markdown("### 使用モデル")
        st.info("Gemma3:12B (Ollama)")

        st.markdown("### タイムアウト")
        st.info("5分間")

    # チャット履歴の表示
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # チャット入力
    if prompt := st.chat_input("メッセージを入力してください..."):
        # ユーザーメッセージを表示
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # AIレスポンスを生成
        with st.chat_message("assistant"):
            message_placeholder = st.empty()

            # LangChainメッセージを作成
            user_message = HumanMessage(content=prompt)
            current_history = st.session_state.conversation_manager.get_history()
            state = {"messages": current_history + [user_message]}

            # ストリーミング応答を処理
            full_response = ""

            try:
                for chunk in app.stream(state):
                    if "llm" in chunk:
                        ai_message_chunk = chunk["llm"]["messages"][-1]
                        if ai_message_chunk.content:
                            full_response += ai_message_chunk.content
                            message_placeholder.markdown(full_response + "▌")

                message_placeholder.markdown(full_response)

                # 会話履歴を更新
                st.session_state.conversation_manager.add_conversation_pair(
                    user_message, AIMessage(content=full_response)
                )

                # セッション状態を更新
                st.session_state.messages.append(
                    {"role": "assistant", "content": full_response}
                )

            except Exception as e:
                st.error(f"エラーが発生しました: {str(e)}")
                st.info("Ollamaが起動していることを確認してください。")


if __name__ == "__main__":
    main()

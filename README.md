# Local LLM AI Agent

Ollamaを使用したローカル環境でのAIチャットエージェント。プライバシーとセキュリティを重視したStreamlit WebUIアプリです。

## 特徴

- **ローカル実行**: 全ての処理がローカル環境で完結
- **プライバシー保護**: データが外部に送信されることはありません
- **会話履歴管理**: 自動的なタイムアウト機能付き
- **WebUI**: Streamlitベースの使いやすいインターフェース
- **ストリーミング応答**: リアルタイムでの応答生成

## 必要要件

- Python 3.12以上
- Ollama（Gemma3:12bモデル）
- uv（パッケージマネージャー）

## インストール

1. Ollamaをインストール:

```bash
# macOS
brew install ollama

# その他のOSは https://ollama.ai/ を参照
```

2. Gemma3:12bモデルをダウンロード:

```bash
ollama pull gemma3:12b
```

3. 依存関係をインストール:

```bash
uv sync
```

## 使用方法

```bash
streamlit run main.py
```

ブラウザで <http://localhost:8501> にアクセスしてください。

## 機能

### 会話履歴管理

- デフォルトで5分間のタイムアウト
- 自動的な履歴クリア機能
- 手動での履歴リセット可能

### WebUI機能

- リアルタイムストリーミング応答
- サイドバーでの設定管理
- 会話履歴の視覚的表示
- ワンクリック履歴クリア

## 設定

### タイムアウト設定

```python
# デフォルト: 300秒（5分）
conversation_manager = ConversationManager(timeout_seconds=300)
```

### モデル変更

```python
# main.py で変更
llm = ChatOllama(model="your-model-name")
```

# ObsiGPT

ObsiGPT は、指定したローカルの Markdown ファイル群（Vault）の内容に基づいて質問応答を行うシンプルなチャットボットです。`llama-index` と OpenAI API を使用しています。

## 必要なもの

*   Python 3.12 以降
*   [uv](https://github.com/astral-sh/uv) (Python パッケージインストーラーおよびリゾルバー)
*   OpenAI API キー

## セットアップ

1.  **リポジトリをクローン:**
    ```bash
    git clone https://github.com/your-username/ObsiGPT.git
    cd ObsiGPT
    ```

2.  **仮想環境の作成と有効化 (uv を使用):**
    ```bash
    uv venv
    source .venv/bin/activate  # macOS / Linux
    # .venv\Scripts\activate  # Windows (Command Prompt)
    # .\.venv\Scripts\Activate.ps1 # Windows (PowerShell)
    ```

3.  **依存関係のインストール (uv を使用):**
    `uv` は `pyproject.toml` と `uv.lock` を見て、必要なライブラリをインストールします。
    ```bash
    uv sync
    ```

4.  **OpenAI API キーの設定:**
    プロジェクトのルートディレクトリ (この `README.md` がある場所) に `.env` という名前のファイルを作成し、以下の内容を記述します。`YOUR_API_KEY` は実際の OpenAI API キーに置き換えてください。
    ```env
    # filepath: .env
    OPENAI_API_KEY=YOUR_API_KEY
    ```
    `.gitignore` により、この `.env` ファイルは Git リポジトリには含まれません。

5.  **Markdown ファイルの準備:**
    `test-vault` ディレクトリ内に、チャットボットに読み込ませたい Markdown ファイル (`.md`) を配置します。サンプルとして `test-vault/test_data.md` が含まれていますが、自由に追加・変更してください。

## 実行

以下のコマンドでチャットボットを起動します。

```bash
python rag_chat.py
```

ターミナルに `🧠 質問してください：` と表示されたら、Vault の内容に関する質問を入力してください。`exit` または `quit` と入力すると終了します。

## 仕組み

1.  `.env` ファイルから OpenAI API キーを読み込みます。
2.  `SimpleDirectoryReader` を使用して `test-vault` 内の Markdown ファイルを読み込みます。
3.  読み込んだドキュメントを OpenAI の `text-embedding-3-small` モデルでベクトル化し、`VectorStoreIndex` を作成します。
4.  ユーザーからの質問を受け取ります。
5.  作成したインデックスを使用して、質問に関連するドキュメントチャンクを検索します。
6.  検索結果と元の質問を OpenAI の `gpt-4o-mini` モデルに渡し、回答を生成します。
7.  生成された回答を表示します。
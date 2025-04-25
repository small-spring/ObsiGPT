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

初回起動時、または `storage` ディレクトリが存在しない場合は、`test-vault` 内のドキュメントからインデックスが作成され、`storage` ディレクトリに保存されます。これには少し時間がかかる場合があります。

2回目以降の起動時には、保存されたインデックス (`storage` ディレクトリ) を使用するかどうか尋ねられます。
*   `y` または Enter を押すと、保存されたインデックスを読み込み、高速に起動します。
*   `n` を押すと、`storage` ディレクトリが削除され、インデックスが再作成されます (Vault の内容を更新した場合など)。

### コマンドラインオプション

*   `--force-recreate`: 起動時に確認なしで `storage` ディレクトリを削除し、インデックスを強制的に再作成します。
    ```bash
    python rag_chat.py --force-recreate
    ```
*   `--no-rag`: Vault のドキュメントを読み込まず、インデックスを使用しません。LLM が直接応答するモードで起動します (ベースラインの確認用)。
    ```bash
    python rag_chat.py --no-rag
    ```

ターミナルに `🧠 質問してください：` と表示されたら、質問を入力してください。`exit` または `quit` と入力すると終了します。

## 仕組み

1.  `.env` ファイルから OpenAI API キーを読み込みます。
2.  `llama_index.core.Settings` を使用して、使用する LLM (`gpt-4o-mini`) と埋め込みモデル (`text-embedding-3-small`) をグローバルに設定します。
3.  `--no-rag` オプションが指定されていない場合:
    *   `storage` ディレクトリが存在するか確認します。
    *   存在する場合、ユーザーに既存インデックスの使用を確認します。
        *   使用する場合: `StorageContext` と `load_index_from_storage` を使ってインデックスを読み込みます。
        *   使用しない場合: `storage` ディレクトリを削除します。
    *   `storage` ディレクトリが存在しない、または再作成が選択された場合:
        *   `SimpleDirectoryReader` を使用して `test-vault` 内の Markdown ファイルを読み込みます。
        *   読み込んだドキュメントから `VectorStoreIndex.from_documents` を使用してインデックスを作成します (埋め込み計算が行われます)。
        *   作成したインデックスを `storage_context.persist` を使って `storage` ディレクトリに保存します。
4.  `--no-rag` オプションが指定されている場合:
    *   ダミーのドキュメントを含む空の `VectorStoreIndex` を作成します。
5.  準備されたインデックスから `as_query_engine()` を呼び出してクエリエンジンを作成します。
6.  ユーザーからの質問を受け付けます。
7.  クエリエンジンが質問を処理します。
    *   RAG 有効モード: 質問に基づいてインデックスから関連性の高いドキュメントチャンクを検索 (Retrieval) し、その情報と元の質問を LLM に渡して回答を生成 (Generation) します。
    *   RAG 無効モード: 検索ステップは機能せず、LLM が質問に直接応答します。
8.  生成された回答を表示します。
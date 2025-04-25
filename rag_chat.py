# rag_chat.py
import os
import shutil
import argparse # argparse をインポート
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Document, # Document をインポート
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv

# --- 定数定義 ---
PERSIST_DIR = "./storage"
VAULT_PATH = "test-vault"
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"

# load_or_create_index 関数は変更なし (前のバージョンのままでOK)
def load_or_create_index(persist_dir: str, vault_path: str, embed_model: OpenAIEmbedding, force_recreate: bool = False) -> VectorStoreIndex | None:
    """
    永続化されたインデックスを読み込むか、なければ新しく作成する。
    force_recreate=True の場合、またはユーザーが選択した場合に再作成する。
    """
    index = None
    should_create_index = False

    if force_recreate and os.path.exists(persist_dir):
        print(f"強制再作成フラグが指定されたため、古いインデックス '{persist_dir}' を削除しています...")
        try:
            shutil.rmtree(persist_dir)
            print(f"'{persist_dir}' を削除しました。")
            should_create_index = True
        except OSError as e:
            print(f"エラー: '{persist_dir}' の削除に失敗しました: {e}")
            should_create_index = True

    if not should_create_index and os.path.exists(persist_dir):
        while True:
            reload_choice = input(f"💾 保存されたインデックスが見つかりました。使用しますか？ (Y/n): ").strip().lower()
            if reload_choice == 'n':
                print(f"古いインデックス '{persist_dir}' を削除しています...")
                try:
                    shutil.rmtree(persist_dir)
                    print(f"'{persist_dir}' を削除しました。")
                    should_create_index = True
                except OSError as e:
                    print(f"エラー: '{persist_dir}' の削除に失敗しました: {e}")
                    should_create_index = True
                break
            elif reload_choice == 'y' or reload_choice == '':
                try:
                    print(f"既存のインデックスを '{persist_dir}' から読み込んでいます...")
                    # LlamaIndex v0.10以降では、load時にembed_modelを直接渡さないことが多い
                    # StorageContext経由でロードし、後でQueryEngine作成時にLLMやEmbed Modelを設定する
                    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
                    # Settings を使ってグローバルに設定するか、ロード後に設定する
                    # ここではロード後に index オブジェクトから設定するアプローチは難しいので、
                    # QueryEngine 作成時に embed_model を意識させる
                    index = load_index_from_storage(storage_context)
                    print("インデックスの読み込みが完了しました。")
                except Exception as e:
                    print(f"インデックスの読み込み中にエラーが発生しました: {e}")
                    print("インデックスを再作成します。")
                    should_create_index = True
                break
            else:
                print("無効な入力です。'y' または 'n' を入力してください。")
    else:
        if not should_create_index:
             print(f"'{persist_dir}' が見つかりません。")
        should_create_index = True

    if should_create_index:
        try:
            print(f"'{vault_path}' からドキュメントを読み込み、新しいインデックスを作成します...")
            documents = SimpleDirectoryReader(vault_path).load_data()
            if not documents:
                print(f"警告: '{vault_path}' 内に読み込めるドキュメントが見つかりませんでした。")
                # ドキュメントがなくても空のインデックスは作成できる場合があるが、
                # RAGの意味がなくなるため、ここではNoneを返す方が適切かもしれない
                # return None
                print("空のインデックスを作成します。") # または空のインデックスを作成
                index = VectorStoreIndex.from_documents([], embed_model=embed_model)

            else:
                 # Settings を使って embed_model を設定する例
                 # Settings.embed_model = embed_model
                 index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

            print(f"インデックスを '{persist_dir}' に保存しています...")
            os.makedirs(persist_dir, exist_ok=True)
            index.storage_context.persist(persist_dir=persist_dir)
            print(f"インデックスを '{persist_dir}' に保存しました。")
        except Exception as e:
            print(f"インデックスの作成または保存中にエラーが発生しました: {e}")
            return None

    return index


# run_chat_loop 関数は変更なし
def run_chat_loop(query_engine):
    """チャットループを実行する"""
    print("\n--- チャットを開始します ---")
    while True:
        try:
            q = input("\n🧠 質問してください：")
            if q.strip().lower() in ["exit", "quit"]:
                print("チャットを終了します。")
                break
            if not q.strip():
                continue

            print("思考中...")
            response = query_engine.query(q)
            print(f"\n🗣️ 回答:\n{response}")

        except EOFError:
             print("\nチャットを終了します。")
             break
        except KeyboardInterrupt:
            print("\nチャットを終了します。")
            break
        except Exception as e:
            print(f"\nエラーが発生しました: {e}")


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(description="Markdown Vault に基づくチャットボット")
    parser.add_argument(
        "--force-recreate",
        action="store_true",
        help="既存のインデックスを無視し、強制的に再作成します。"
    )
    parser.add_argument(
        "--no-rag",
        action="store_true",
        help="Vault のドキュメントを読み込まず、LLM のみの応答にします。"
    )
    args = parser.parse_args()

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        print("エラー: 環境変数 'OPENAI_API_KEY' が設定されていません。")
        print(".env ファイルを作成し、'OPENAI_API_KEY=YOUR_API_KEY' の形式でキーを設定してください。")
        return

    try:
        llm = OpenAI(model=LLM_MODEL, temperature=0.2)
        embed_model = OpenAIEmbedding(model=EMBEDDING_MODEL)
    except Exception as e:
        print(f"モデルの初期化中にエラーが発生しました: {e}")
        return

    index = None
    if args.no_rag:
        print("--- RAG機能 無効モード ---")
        try:
            # 空リストではなく、ダミーのDocumentオブジェクトを1つ渡す
            dummy_document = Document(text="This is a dummy document to initialize the index.")
            index = VectorStoreIndex.from_documents([dummy_document], embed_model=embed_model)
            print("ダミードキュメントからインデックスを作成しました。LLM のみの応答になります。")
        except Exception as e:
             print(f"ダミーインデックス作成中にエラー: {e}")
             return

    else:
        print("--- RAG機能 有効モード ---")
        index = load_or_create_index(PERSIST_DIR, VAULT_PATH, embed_model, force_recreate=args.force_recreate)

    if index is None:
        print("エラー: インデックスの準備に失敗しました。プログラムを終了します。")
        return

    query_engine = index.as_query_engine(llm=llm)
    run_chat_loop(query_engine)

if __name__ == "__main__":
    main()
# rag_chat.py
import os
import shutil
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv

# --- 定数定義 ---
PERSIST_DIR = "./storage"
VAULT_PATH = "test-vault"
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"

def load_or_create_index(persist_dir: str, vault_path: str, embed_model: OpenAIEmbedding) -> VectorStoreIndex | None:
    """
    永続化されたインデックスを読み込むか、なければ新しく作成する。
    ユーザーに再作成の選択肢を提示する。
    """
    index = None
    should_create_index = False

    if os.path.exists(persist_dir):
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
                    # 削除に失敗した場合でも、インデックス作成に進む
                    should_create_index = True
                break
            elif reload_choice == 'y' or reload_choice == '':
                try:
                    print(f"既存のインデックスを '{persist_dir}' から読み込んでいます...")
                    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
                    index = load_index_from_storage(storage_context, embed_model=embed_model)
                    print("インデックスの読み込みが完了しました。")
                except Exception as e:
                    print(f"インデックスの読み込み中にエラーが発生しました: {e}")
                    print("インデックスを再作成します。")
                    should_create_index = True
                break
            else:
                print("無効な入力です。'y' または 'n' を入力してください。")
    else:
        print(f"'{persist_dir}' が見つかりません。")
        should_create_index = True

    if should_create_index:
        try:
            print(f"'{vault_path}' からドキュメントを読み込み、新しいインデックスを作成します...")
            documents = SimpleDirectoryReader(vault_path).load_data()
            if not documents:
                print(f"警告: '{vault_path}' 内に読み込めるドキュメントが見つかりませんでした。")
                return None # ドキュメントがない場合はインデックスを作成できない
            index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
            print(f"インデックスを '{persist_dir}' に保存しています...")
            index.storage_context.persist(persist_dir=persist_dir)
            print(f"インデックスを '{persist_dir}' に保存しました。")
        except Exception as e:
            print(f"インデックスの作成または保存中にエラーが発生しました: {e}")
            return None # エラー発生時は None を返す

    return index

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
                continue # 空の入力は無視

            print("思考中...") # 応答待機中のメッセージ
            response = query_engine.query(q)
            print(f"\n🗣️ 回答:\n{response}")

        except EOFError: # Ctrl+D などで終了した場合
             print("\nチャットを終了します。")
             break
        except KeyboardInterrupt: # Ctrl+C で終了した場合
            print("\nチャットを終了します。")
            break
        except Exception as e:
            print(f"\nエラーが発生しました: {e}")
            # エラーが発生してもループを継続する場合が多いが、必要に応じて break する
            # break


def main():
    """メイン処理"""
    # 環境変数読み込み
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        print("エラー: 環境変数 'OPENAI_API_KEY' が設定されていません。")
        print(".env ファイルを作成し、'OPENAI_API_KEY=YOUR_API_KEY' の形式でキーを設定してください。")
        return # APIキーがない場合は終了

    # モデルの初期化
    try:
        embed_model = OpenAIEmbedding(model=EMBEDDING_MODEL)
        llm = OpenAI(model=LLM_MODEL, temperature=0.2)
    except Exception as e:
        print(f"モデルの初期化中にエラーが発生しました: {e}")
        return

    # インデックスの読み込み/作成
    index = load_or_create_index(PERSIST_DIR, VAULT_PATH, embed_model)

    if index is None:
        print("エラー: インデックスの準備に失敗しました。プログラムを終了します。")
        return

    # クエリエンジンの作成
    query_engine = index.as_query_engine(llm=llm)

    # チャットループの実行
    run_chat_loop(query_engine)

if __name__ == "__main__":
    main()
# rag_chat.py

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI 
from dotenv import load_dotenv

# 環境変数読み込み（.envファイルにOPENAI_API_KEY=xxx を入れる）
load_dotenv()

# ディレクトリ内のMarkdownファイルを読み込む
path_to_vault = "test-vault"
documents = SimpleDirectoryReader(path_to_vault).load_data()

# 埋め込みとモデルの定義
embed_model = OpenAIEmbedding(model="text-embedding-3-small")
llm = OpenAI(model="gpt-4o-mini", temperature=0.2)

# インデックス作成（初回のみ少し時間がかかる）
index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

# チャット開始！
query_engine = index.as_query_engine(llm=llm)

while True:
    q = input("\n🧠 質問してください：")
    if q.strip().lower() in ["exit", "quit"]:
        break
    response = query_engine.query(q)
    print(f"\n🗣️ 回答:\n{response}")

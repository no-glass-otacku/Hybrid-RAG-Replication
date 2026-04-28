# Importing necessary libraries for loading datasets, data manipulation, document processing, graph storage, and embeddings.
from datasets import load_dataset
import pandas as pd
from llama_index.core import Document, PropertyGraphIndex
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.indices.property_graph import (
    SimpleLLMPathExtractor,
    ImplicitPathExtractor,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from neo4j import GraphDatabase
from utils import chunked_iterable, load_config
import openai
import os

# Hardcoded values for easy adjustment (논문 명시값 — Graph RAG / Hybrid RAG 공통)
TOKEN_CHUNK_SIZE = 2024
CHUNK_OVERLAP = 204

# Naming policy: Vector RAG 의 'ai_arxiv_full' 과 동일한 'ai_arxiv_*' 패턴
# 단, Neo4j Aura(Free/Pro) 와 Community Edition 은 사용자 DB 생성을 지원하지 않으므로
# 기본값은 'neo4j' 로 두고, 멀티 DB 지원 환경(Enterprise / 셀프호스팅)에서만
# .env 의 NEO4J_DATABASE_HYBRID 로 'ai_arxiv_hybrid' 같은 이름을 지정한다.
HYBRID_DB_LABEL = "ai_arxiv_hybrid"  # 로그/리포트용 식별자 (실제 DB 이름과는 별개)
HYBRID_DB_NAME = os.getenv("NEO4J_DATABASE_HYBRID", "neo4j")

# Load the config file
load_config()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load dataset and convert to DataFrame for easier manipulation
dataset = load_dataset("jamescalam/ai-arxiv")
df = pd.DataFrame(dataset['train'])

# Specify the titles of the required papers
required_paper_titles = [
    'BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding',
    'DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter',
    'HellaSwag: Can a Machine Really Finish Your Sentence?',
    'LLaMA: Open and Efficient Foundation Language Models',
    'Measuring Massive Multitask Language Understanding',
    'CodeNet: A Large-Scale AI for Code Dataset for Learning a Diversity of Coding Tasks',
    'Task2Vec: Task Embedding for Meta-Learning',
    'GLM-130B: An Open Bilingual Pre-trained Model',
    'SuperGLUE: A Stickier Benchmark for General-Purpose Language Understanding Systems',
    "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism",
    "PAL: Program-aided Language Models",
    "RoBERTa: A Robustly Optimized BERT Pretraining Approach",
    "DetectGPT: Zero-Shot Machine-Generated Text Detection using Probability Curvature"
]
# Filter the DataFrame to include only the required papers
required_papers = df[df['title'].isin(required_paper_titles)]

# Exclude the already selected papers to avoid duplicates and randomly sample ~40-50 papers
remaining_papers = df[~df['title'].isin(required_paper_titles)].sample(n=40, random_state=123)

# Concatenate the two DataFrames
final_df = pd.concat([required_papers, remaining_papers], ignore_index=True)

# Prepare document objects from the dataset for indexing
documents = [Document(text=content) for content in df['content']]

# Setup the LLM (Graph Builder) and embedding model
# embedding_model 은 Vector RAG 와 반드시 동일해야 한다 (다이어그램 파라미터 범례 [A] 공통값).
llm = OpenAI(model="gpt-3.5-turbo", temperature=0.0)
embed_model = OpenAIEmbedding(model="text-embedding-3-large")

# Neo4j connection
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")


def _ensure_database(name: str) -> None:
    """
    멀티 DB 지원 환경(Enterprise / 셀프호스팅)에서만 사용자 DB 생성을 시도한다.
    기본값 'neo4j' 는 항상 존재하므로 아무 일도 하지 않는다.

    Aura Free/Pro 또는 Community 에서 커스텀 이름을 지정한 경우엔
    'CREATE DATABASE' 가 막혀있어 실패하는데, 이때는 명확한 안내 메시지와 함께
    종료해서 사용자가 즉시 .env 를 수정할 수 있게 한다.
    """
    if name == "neo4j":
        return
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    try:
        with driver.session(database="system") as ses:
            ses.run(f"CREATE DATABASE {name} IF NOT EXISTS")
            print(f"[info] DB '{name}' 준비됨")
    except Exception as e:
        print(f"[error] DB '{name}' 생성 실패: {e}")
        print(
            f"        Aura Free/Pro 또는 Community 라면 .env 에서\n"
            f"        NEO4J_DATABASE_HYBRID=neo4j 로 두세요. (기본 DB 사용)\n"
            f"        Enterprise 라면 미리 수동으로 'CREATE DATABASE {name}' 실행 후 재시도."
        )
        raise SystemExit(1)
    finally:
        driver.close()


def _db_has_data(name: str) -> bool:
    """이미 인덱싱된 그래프가 있는지 노드 개수로 판단."""
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    try:
        with driver.session(database=name) as ses:
            row = ses.run("MATCH (n) RETURN count(n) AS n").single()
            return bool(row and row["n"] > 0)
    finally:
        driver.close()


def _has_embedded_entities(name: str) -> bool:
    """Graph 빌드 잔존물 감지용 — :__Entity__ 노드에 embedding 이 하나라도 있는지."""
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    try:
        with driver.session(database=name) as ses:
            row = ses.run(
                "MATCH (n:__Entity__) WHERE n.embedding IS NOT NULL "
                "RETURN count(n) AS n"
            ).single()
            return bool(row and row["n"] > 0)
    finally:
        driver.close()


_ensure_database(HYBRID_DB_NAME)

# Aura/Community 처럼 Graph 와 Hybrid 가 같은 'neo4j' DB 를 공유하는 경우
# 잘못된 flavor 잔존물을 빨리 잡아낸다.
if _db_has_data(HYBRID_DB_NAME) and not _has_embedded_entities(HYBRID_DB_NAME):
    print(
        f"[error] DB '{HYBRID_DB_NAME}' 에 임베딩이 없는 그래프(Graph 잔존물)가 있어\n"
        f"        Hybrid RAG 로 로드할 수 없습니다. 먼저 reset 후 다시 실행하세요:\n"
        f"            python scripts/reset_neo4j.py --flavor hybrid"
    )
    raise SystemExit(1)


# Hybrid DB
if not _db_has_data(HYBRID_DB_NAME):
    print("Hybrid DB 구축 시작...")
    parser = TokenTextSplitter(chunk_size=TOKEN_CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    nodes = parser.get_nodes_from_documents(documents)

    graph_store = Neo4jPropertyGraphStore(
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        url=NEO4J_URI,
        database=HYBRID_DB_NAME,
    )

    extractors = [
        # 다이어그램의 'Neo4j LLM Graph Builder 자동 추출' (node_schema/relation_types 자동)
        SimpleLLMPathExtractor(llm=llm, max_paths_per_chunk=20, num_workers=4),
        ImplicitPathExtractor(),
    ]

    index = PropertyGraphIndex(
        nodes=nodes,
        kg_extractors=extractors,
        property_graph_store=graph_store,
        llm=llm,
        embed_model=embed_model,
        embed_kg_nodes=True,         # ★ Hybrid RAG 핵심: 노드 + 임베딩 동시 저장
        show_progress=True,
    )
    print("Hybrid DB 구축 완료!")
else:
    print("이미 존재하는 그래프입니다. 건너뜁니다.")
    graph_store = Neo4jPropertyGraphStore(
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        url=NEO4J_URI,
        database=HYBRID_DB_NAME,
    )
    index = PropertyGraphIndex.from_existing(
        property_graph_store=graph_store,
        embed_model=embed_model,
        embed_kg_nodes=True,
    )
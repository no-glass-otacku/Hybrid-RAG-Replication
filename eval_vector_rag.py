"""
Vector RAG → RAGAS 성능 평가 파이프라인
=========================================
흐름: ChromaDB 로드 → 쿼리 엔진 구성 → ARAGOG 벤치마크 실행
      → RAGAS 6개 메트릭 평가 → 결과 CSV 저장

필요 패키지:
    pip install ragas datasets langchain-openai llama-index-llms-openai
                llama-index-embeddings-openai llama-index-vector-stores-chroma
"""

import os
import json
import pandas as pd
import chromadb

from llama_index.core import VectorStoreIndex, PromptTemplate, Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from langchain_ollama import OllamaEmbeddings as LangchainOllamaEmbeddings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

# RAGAS
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
        answer_relevancy,
    answer_correctness,
    answer_similarity,
)

# RAGAS가 내부적으로 사용할 평가 LLM/임베딩 (논문 기준: GPT-4o)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from utils import load_config

# ── 환경 설정 ──────────────────────────────────────────────────────────────────
load_config()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


# ── 파라미터 (논문 기준값) ─────────────────────────────────────────────────────
GENERATION_MODEL  = "gpt-4o-mini" #"gpt-3.5-turbo"   # 답변 생성 LLM
EVAL_MODEL        = "gpt-4o-mini" #"gpt-4o"          # RAGAS 평가 LLM
EMBED_MODEL       = "text-embedding-3-large"
CHROMA_PATH       = "./chroma_db"
COLLECTION_NAME   = "ai_arxiv_full"
SIMILARITY_TOP_K  = 4                 # 다이어그램 기준 top_k = 4
TEMPERATURE       = 0.0               # deterministic 생성

# 벤치마크 파일 경로 (question, ground_truth 컬럼 포함 CSV 또는 JSON)
BENCHMARK_PATH    = "eval_questions/benchmark.json"   # ← 실제 경로로 수정

# 결과 저장 경로
OUTPUT_CSV        = "results/vector_rag_ragas_results.csv"
os.makedirs("results", exist_ok=True)


# ── LLM / 임베딩 초기화 ────────────────────────────────────────────────────────
generation_llm = OpenAI(model=GENERATION_MODEL, temperature=TEMPERATURE)
# embed_model    = OpenAIEmbedding(model=EMBED_MODEL)
embed_model    = OllamaEmbedding("nomic-embed-text")
Settings.llm         = generation_llm
Settings.embed_model = embed_model

# RAGAS 평가용 LLM/임베딩 (LangChain 래퍼 필요)
ragas_llm   = LangchainLLMWrapper(ChatOpenAI(model=EVAL_MODEL, temperature=TEMPERATURE))
# ragas_embed = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=EMBED_MODEL))
ragas_embed = LangchainEmbeddingsWrapper(
    LangchainOllamaEmbeddings(model="nomic-embed-text")
)
# RAGAS 메트릭에 평가 LLM / 임베딩 주입
RAGAS_METRICS = [
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
    answer_correctness,
    answer_similarity,
]
for metric in RAGAS_METRICS:
    metric.llm        = ragas_llm
    metric.embeddings = ragas_embed


# ── VectorDB 로드 ──────────────────────────────────────────────────────────────
print(f"[1/4] ChromaDB 로드 중: {CHROMA_PATH} / {COLLECTION_NAME}")
chroma_client     = chromadb.PersistentClient(path=CHROMA_PATH)
chroma_collection = chroma_client.get_collection(COLLECTION_NAME)
vector_store      = ChromaVectorStore(chroma_collection=chroma_collection)
index             = VectorStoreIndex.from_vector_store(vector_store=vector_store)
print(f"      문서 수: {chroma_collection.count()}")


# ── 프롬프트 템플릿 ────────────────────────────────────────────────────────────
with open("resources/text_qa_template.txt", "r", encoding="utf-8") as f:
    text_qa_template = PromptTemplate(f.read())


# ── 쿼리 엔진 구성 ─────────────────────────────────────────────────────────────
print(f"[2/4] 쿼리 엔진 구성 (top_k={SIMILARITY_TOP_K})")
query_engine = index.as_query_engine(
    llm=generation_llm,
    embed_model=embed_model,
    text_qa_template=text_qa_template,
    similarity_top_k=SIMILARITY_TOP_K,
)


# ── 벤치마크 로드 ──────────────────────────────────────────────────────────────
print(f"[3/4] 벤치마크 로드 중: {BENCHMARK_PATH}")

ext = os.path.splitext(BENCHMARK_PATH)[-1].lower()
if ext == ".csv":
    benchmark_df = pd.read_csv(BENCHMARK_PATH)
elif ext in (".json", ".jsonl"):
    benchmark_df = pd.read_json(BENCHMARK_PATH)
else:
    raise ValueError(f"지원하지 않는 형식: {ext}  (csv / json / jsonl 만 가능)")

# 컬럼명 유연하게 처리
col_map = {}
for col in benchmark_df.columns:
    if col.lower() in ("questions", "query"):
        col_map["questions"] = col
    if col.lower() in ("ground_truths", "answer", "reference", "expected_answer"):
        col_map["ground_truths"] = col

assert "questions"     in col_map, "벤치마크 파일에 'question' 컬럼이 없습니다."
assert "ground_truths" in col_map, "벤치마크 파일에 'ground_truths' 컬럼이 없습니다."

questions  = benchmark_df[col_map["questions"]].tolist()
references = benchmark_df[col_map["ground_truths"]].tolist()
print(f"      QA 쌍: {len(questions)}개")


# ── 쿼리 실행 & 데이터 수집 ────────────────────────────────────────────────────
print("[4/4] 쿼리 실행 중...")

ragas_data = {
    "user_input": [],  # RAGAS 0.2+: 구 "question"
    "response": [],  # RAGAS 0.2+: 구 "answer"
    "retrieved_contexts": [],  # RAGAS 0.2+: 구 "contexts"  List[List[str]]
    "reference": [],  # RAGAS 0.2+: 구 "ground_truth"
}

for i, (question, reference) in enumerate(zip(questions, references), 1):
    print(f"  [{i:3d}/{len(questions)}] {question[:60]}...")

    response = query_engine.query(question)

    ragas_data["user_input"].append(question)
    ragas_data["response"].append(str(response))
    ragas_data["retrieved_contexts"].append(
        [node.node.get_content() for node in response.source_nodes]
    )
    ragas_data["reference"].append(reference)

# ── RAGAS 평가 ─────────────────────────────────────────────────────────────────
print("\n[RAGAS] 평가 실행 중 (6개 메트릭)...")
dataset = Dataset.from_dict(ragas_data)
result  = evaluate(dataset=dataset, metrics=RAGAS_METRICS)

scores_df = result.to_pandas()
scores_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
print(f"\n결과 저장 완료 → {OUTPUT_CSV}")


# ── 요약 출력 ──────────────────────────────────────────────────────────────────
metric_cols = [m.name for m in RAGAS_METRICS]
summary = scores_df[metric_cols].agg(["mean", "std", "min", "max"])

print("\n" + "="*60)
print("  Vector RAG RAGAS 평가 결과 요약")
print("="*60)
print(summary.round(4).to_string())
print("="*60)
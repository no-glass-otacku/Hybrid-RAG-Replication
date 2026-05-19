"""
Hybrid RAG → RAGAS 성능 평가 파이프라인
==========================================
흐름: Neo4j Hybrid DB 로드 → 노드유사도+1-hop retriever 구성 → ARAGOG 벤치마크 실행
      → RAGAS 6개 메트릭 평가 → 결과 CSV 저장

다이어그램의 'Hybrid RAG 검색' 정의에 충실:
    노드 유사도 계산  sim(v_q, N_i.embedding) on 전체 노드
       └─→ 상위 K 노드 선택 (top_k=4, 내림차순)
              └─→ 관계 탐색 get_relationships(N_i), hop_depth=1
                  └─→ 정보 통합 (decode): 텍스트 + 트리플 → retrieved_info

LlamaIndex 의 `VectorContextRetriever(path_depth=1, include_text=True)` 가
이 단계를 그대로 구현한다 — 별도의 LLM 키워드 매칭 retriever 는 사용하지 않는다
(다이어그램의 Hybrid RAG 정의는 노드 임베딩 + 관계 탐색 두 가지만 묶음).

필요 패키지:
    pip install ragas datasets langchain-openai llama-index-llms-openai
                llama-index-embeddings-openai llama-index-graph-stores-neo4j
"""

import os
import json
import pandas as pd

from llama_index.core import PropertyGraphIndex, PromptTemplate, Settings
from llama_index.core.indices.property_graph import VectorContextRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore

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

# RAGAS가 내부적으로 사용할 평가 LLM/임베딩 (다이어그램: GPT-4o)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from utils import load_config

# ── 환경 설정 ──────────────────────────────────────────────────────────────────
load_config()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


# ── 파라미터 (다이어그램 명시값) ───────────────────────────────────────────────
# [C] generation 공통값 — 세 RAG 모두 동일하게 설정
GENERATION_MODEL  = "gpt-4o-mini"   # 답변 생성 LLM (다이어그램 명시; 비용절감 시 gpt-4o-mini 가능)
EVAL_MODEL        = "gpt-4o-mini"          # RAGAS 평가 LLM (다이어그램 명시; RAGAS 공식 default)
EMBED_MODEL       = "text-embedding-3-large"  # [A] 인덱싱·검색 동일 모델
TEMPERATURE       = 0.0               # [C] deterministic 생성
MAX_TOKENS        = 1024              # [C] max_tokens 동일
SIMILARITY_TOP_K  = 4                 # [B] Vector ↔ Hybrid 동일 top_k
HOP_DEPTH         = 1                 # 다이어그램: 1-hop traversal

# Neo4j 접속 (build_hybrid_db.py 와 동일한 환경변수 사용)
NEO4J_URI         = os.getenv("NEO4J_URI_HYBRID", "bolt://localhost:7687")
NEO4J_USERNAME    = os.getenv("NEO4J_USERNAME_HYBRID", "4565a96f")
NEO4J_PASSWORD    = os.getenv("NEO4J_PASSWORD_HYBRID")
HYBRID_DB_LABEL   = "ai_arxiv_hybrid"  # Vector RAG 의 'ai_arxiv_full' 과 동일한 네이밍 정책
HYBRID_DB_NAME    = os.getenv("NEO4J_DATABASE_HYBRID", "4565a96f")

# 벤치마크 (논문: ARAGOG, arXiv AI 논문 16개 / QA쌍 107개)
BENCHMARK_PATH    = "eval_questions/benchmark.json"   # ← 실제 경로로 수정

# 결과 저장 경로
OUTPUT_CSV        = "results/hybrid_rag_ragas_results.csv"
os.makedirs("results", exist_ok=True)


# ── LLM / 임베딩 초기화 ────────────────────────────────────────────────────────
generation_llm = OpenAI(
    model=GENERATION_MODEL,
    temperature=TEMPERATURE,
    max_tokens=MAX_TOKENS,
)
embed_model = OpenAIEmbedding(model=EMBED_MODEL)

Settings.llm         = generation_llm
Settings.embed_model = embed_model

# RAGAS 평가용 LLM/임베딩 (LangChain 래퍼 필요)
ragas_llm   = LangchainLLMWrapper(ChatOpenAI(model=EVAL_MODEL, temperature=TEMPERATURE))
ragas_embed = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=EMBED_MODEL))

# RAGAS 메트릭에 평가 LLM / 임베딩 주입 (다이어그램의 6개 지표)
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


# ── Hybrid DB 로드 ─────────────────────────────────────────────────────────────
print(f"[1/4] Neo4j Hybrid DB 로드 중: {NEO4J_URI} / db={HYBRID_DB_NAME} ({HYBRID_DB_LABEL})")
graph_store = Neo4jPropertyGraphStore(
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    url=NEO4J_URI,
    database=HYBRID_DB_NAME,
)
index = PropertyGraphIndex.from_existing(
    property_graph_store=graph_store,
    embed_model=embed_model,
    embed_kg_nodes=True,    # Hybrid RAG: 노드 임베딩 사용
)


# ── 프롬프트 템플릿 ────────────────────────────────────────────────────────────
# [C] prompt_template — 세 RAG 모두 동일 형식
with open("resources/text_qa_template.txt", "r", encoding="utf-8") as f:
    text_qa_template = PromptTemplate(f.read())


# ── 쿼리 엔진 구성 (Hybrid RAG: 노드 유사도 + 1-hop 관계 탐색) ────────────────
print(f"[2/4] Hybrid 쿼리 엔진 구성 (retriever=VectorContext, "
      f"top_k={SIMILARITY_TOP_K}, path_depth={HOP_DEPTH})")
# 다이어그램의 'Hybrid RAG 검색' 4단계를 단일 retriever 가 모두 처리:
#   1) sim(v_q, N_i.embedding)  — 쿼리 임베딩과 모든 노드 임베딩 유사도
#   2) top_k=4 (내림차순)
#   3) get_relationships(N_i), hop_depth=1
#   4) decode(N_i.emb) + (Ni,Rij,Nj)  — 텍스트 + 트리플 묶어서 반환
vector_retriever = VectorContextRetriever(
    graph_store=graph_store,
    embed_model=embed_model,
    similarity_top_k=SIMILARITY_TOP_K,
    path_depth=HOP_DEPTH,         # 1-hop 관계 탐색
    include_text=True,            # 매칭 노드의 source chunk 텍스트 포함 → '텍스트 + 트리플'
)

retriever = index.as_retriever(
    sub_retrievers=[vector_retriever],
    similarity_top_k=SIMILARITY_TOP_K,
)
query_engine = RetrieverQueryEngine.from_args(
    retriever=retriever,
    llm=generation_llm,
    text_qa_template=text_qa_template,
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
    "user_input": [],          # RAGAS 0.2+: 구 "question"
    "response": [],            # RAGAS 0.2+: 구 "answer"
    "retrieved_contexts": [],  # RAGAS 0.2+: 구 "contexts"  List[List[str]]
    "reference": [],           # RAGAS 0.2+: 구 "ground_truth"
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
print("  Hybrid RAG RAGAS 평가 결과 요약")
print("="*60)
print(summary.round(4).to_string())
print("="*60)
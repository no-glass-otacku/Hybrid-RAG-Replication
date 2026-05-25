"""
Graph RAG → RAGAS 성능 평가 파이프라인
=========================================
흐름: Neo4j Graph DB 로드 → Graph 전용 retriever 구성 → ARAGOG 벤치마크 실행
      → RAGAS 6개 메트릭 평가 → 결과 CSV 저장

다이어그램의 'Graph RAG 검색' 정의에 충실:
    Cypher 쿼리 변환 (query_llm=GPT-3.5-Turbo)
       └─→ 그래프 쿼리 실행 (Neo4j / Cypher)
              └─→ 노드·관계 검색 (hop_depth=1)
                  결과: (Ni, Rij, Nj) 트리플 → 프롬프트 컨텍스트로 전달

LlamaIndex 의 `TextToCypherRetriever` 가 이 단계를 그대로 구현한다.

필요 패키지:
    pip install ragas datasets langchain-openai llama-index-llms-openai
                llama-index-embeddings-openai llama-index-graph-stores-neo4j
"""

import os
import re
import json
import pandas as pd
# Cypher 문법에러가 자꾸 나서 에러 검증 과정 추가
from neo4j.exceptions import CypherSyntaxError
from llama_index.core import PropertyGraphIndex, PromptTemplate, Settings
from llama_index.core.indices.property_graph import TextToCypherRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.ollama import OllamaEmbedding
from langchain_ollama import OllamaEmbeddings as LangchainOllamaEmbeddings
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

from utils import load_config, get_neo4j_creds

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

# Neo4j 접속 — Graph 전용 인스턴스 (NEO4J_*_GRAPH 우선, 없으면 공통 NEO4J_* 폴백)
_creds            = get_neo4j_creds("graph")
NEO4J_URI         = _creds.uri
NEO4J_USERNAME    = _creds.username
NEO4J_PASSWORD    = _creds.password
GRAPH_DB_LABEL    = "ai_arxiv_graph"   # Vector RAG 의 'ai_arxiv_full' 과 동일한 네이밍 정책
GRAPH_DB_NAME     = _creds.database

# 벤치마크 (논문: ARAGOG, arXiv AI 논문 16개 / QA쌍 107개)
BENCHMARK_PATH    = "eval_questions/benchmark.json"   # ← 실제 경로로 수정

# 결과 저장 경로
OUTPUT_CSV        = "results/graph_rag_ragas_results.csv"
os.makedirs("results", exist_ok=True)


# ── LLM / 임베딩 초기화 ────────────────────────────────────────────────────────
generation_llm = OpenAI(
    model=GENERATION_MODEL,
    temperature=TEMPERATURE,
    max_tokens=MAX_TOKENS,
)
# embed_model = OpenAIEmbedding(model=EMBED_MODEL)
embed_model = LangchainEmbeddingsWrapper(
    LangchainOllamaEmbeddings(model="nomic-embed-text")
)
Settings.llm         = generation_llm
Settings.embed_model = embed_model

# RAGAS 평가용 LLM/임베딩 (LangChain 래퍼 필요)
ragas_llm   = LangchainLLMWrapper(ChatOpenAI(model=EVAL_MODEL, temperature=TEMPERATURE))
# ragas_embed = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=EMBED_MODEL))
ragas_embed = LangchainEmbeddingsWrapper(
    LangchainOllamaEmbeddings(model="nomic-embed-text")
)

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


# ── Graph DB 로드 ──────────────────────────────────────────────────────────────
print(f"[1/4] Neo4j Graph DB 로드 중: {NEO4J_URI} / db={GRAPH_DB_NAME} ({GRAPH_DB_LABEL})")
graph_store = Neo4jPropertyGraphStore(
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    url=NEO4J_URI,
    database=GRAPH_DB_NAME,
)
index = PropertyGraphIndex.from_existing(
    property_graph_store=graph_store,
    # embed_model=embed_model,
    embed_kg_nodes=False,    # Graph RAG: 노드 임베딩 없음
)


# ── 프롬프트 템플릿 ────────────────────────────────────────────────────────────
# [C] prompt_template — 세 RAG 모두 동일 형식
with open("resources/text_qa_template.txt", "r", encoding="utf-8") as f:
    text_qa_template = PromptTemplate(f.read())


# ── 쿼리 엔진 구성 (Graph RAG: Cypher 쿼리 변환 + 1-hop 검색) ─────────────────
print(f"[2/4] Graph 쿼리 엔진 구성 (retriever=TextToCypher, hop_depth={HOP_DEPTH})")
# 다이어그램의 'Cypher 쿼리 변환 → 그래프 쿼리 실행 → 노드·관계 검색' 단계.
# query_llm 으로 자연어 q → Cypher Q_q 변환 후 Neo4j 에서 직접 실행하여
# (Ni, Rij, Nj) 트리플 집합을 그대로 컨텍스트로 사용한다.

# (1) gpt-3.5-turbo 가 종종 ```cypher ... ``` 마크다운으로 감싸서 반환하는데
#     Neo4j 파서는 그걸 못 벗기고 SyntaxError 를 낸다. 코드펜스를 제거하는
#     validator 를 cypher_validator 훅으로 끼워 넣는다.
# ── Cypher 정제 / 검증 / 복구 ────────────────────────────────────────────────

FORBIDDEN_PATTERN = re.compile(
    r"\b("
    r"CREATE|MERGE|SET|DELETE|DETACH\s+DELETE|DROP|REMOVE|"
    r"LOAD\s+CSV|CALL\s+dbms|CALL\s+apoc|UNION|"
    r"SELECT|FROM|JOIN|GROUP\s+BY|HAVING"
    r")\b",
    re.IGNORECASE,
)

def clean_cypher(raw: str) -> str:
    """
    LLM이 생성한 Cypher에서 실행에 방해되는 표현을 제거한다.
    """
    q = raw.strip()

    # ```cypher ... ``` 제거
    q = re.sub(r"^```(?:cypher|sql)?\s*", "", q, flags=re.IGNORECASE)
    q = re.sub(r"\s*```$", "", q)

    # accidental prefix 제거
    for prefix in ["Cypher query:", "Query:", "Cypher:"]:
        if prefix.lower() in q.lower():
            q = re.split(prefix, q, flags=re.IGNORECASE)[-1].strip()

    # 첫 번째 statement만 사용
    q = q.split(";")[0].strip()

    return q


def is_safe_read_query(query: str) -> tuple[bool, str | None]:
    """
    read-only Cypher인지 1차 검사한다.
    """
    upper = query.upper().strip()

    if not upper.startswith(("MATCH", "OPTIONAL MATCH")):
        return False, "Query must start with MATCH or OPTIONAL MATCH."

    if FORBIDDEN_PATTERN.search(query):
        return False, "Query contains forbidden Cypher or SQL keyword."

    if not re.search(r"\bRETURN\b", upper):
        return False, "Query must contain RETURN."

    if not re.search(r"\bLIMIT\s+10\s*$", upper):
        return False, "Query must end with LIMIT 10."

    return True, None


def explain_cypher(query: str) -> tuple[bool, str | None]:
    """
    Neo4j에 EXPLAIN을 날려 실제 문법 검증을 수행한다.
    """
    try:
        graph_store.structured_query("EXPLAIN " + query)
        return True, None
    except Exception as e:
        return False, str(e)


CYPHER_REPAIR_TEMPLATE = PromptTemplate(
    "You are an expert Neo4j Cypher 5 syntax fixer.\n"
    "Fix the invalid Cypher query using the given schema and error message.\n\n"

    "Schema:\n"
    "{schema}\n\n"

    "Invalid Cypher query:\n"
    "{bad_query}\n\n"

    "Error or validation reason:\n"
    "{error}\n\n"

    "STRICT RULES:\n"
    "- Output ONLY the corrected raw Cypher query.\n"
    "- No markdown code fences.\n"
    "- No explanations, comments, or prose.\n"
    "- Single read-only statement only.\n"
    "- The query must start with MATCH or OPTIONAL MATCH.\n"
    "- Use only labels, relationship types, and properties from the schema.\n"
    "- Do NOT use UNION.\n"
    "- Do NOT use CREATE, MERGE, SET, DELETE, DETACH DELETE, DROP, REMOVE, LOAD CSV, CALL dbms, or CALL apoc.\n"
    "- Do NOT use SQL syntax: SELECT, FROM, JOIN, GROUP BY, HAVING.\n"
    "- If aggregation is needed, use WITH or RETURN aggregation. Never use GROUP BY.\n"
    "- WHERE may appear only immediately after MATCH, OPTIONAL MATCH, or WITH.\n"
    "- ORDER BY may appear only after RETURN or WITH, and must appear before LIMIT.\n"
    "- Always end with LIMIT 10.\n"
    "- Do NOT end with a semicolon.\n\n"

    "Corrected Cypher query:"
)


def repair_cypher(query: str, error: str) -> str:
    """
    안전성 검사 또는 EXPLAIN 검증에 실패한 Cypher를 한 번만 복구한다.
    """
    repaired = generation_llm.predict(
        CYPHER_REPAIR_TEMPLATE,
        schema=graph_store.get_schema_str(),
        bad_query=query,
        error=error,
    )
    return clean_cypher(repaired)


def safe_cypher_validator(cypher_query: str) -> str:
    """
    TextToCypherRetriever에 연결할 실제 validator.
    1. 정제
    2. read-only / SQL 문법 차단
    3. EXPLAIN 문법 검증
    4. 실패 시 1회 repair
    5. repair 결과 재검증
    """
    query = clean_cypher(cypher_query)

    safe, reason = is_safe_read_query(query)

    if not safe:
        query = repair_cypher(query, reason or "Unsafe query generated.")

    safe, reason = is_safe_read_query(query)
    if not safe:
        raise ValueError(f"Unsafe Cypher after repair: {reason}\nQuery:\n{query}")

    ok, error = explain_cypher(query)

    if not ok:
        query = repair_cypher(query, error or "Cypher syntax error.")

    safe, reason = is_safe_read_query(query)
    if not safe:
        raise ValueError(f"Unsafe Cypher after syntax repair: {reason}\nQuery:\n{query}")

    ok, error = explain_cypher(query)
    if not ok:
        raise ValueError(f"Cypher EXPLAIN failed after repair:\n{error}\n\nQuery:\n{query}")

    print(f"        [generated cypher] {query}")

    return query
TEXT_TO_CYPHER_TEMPLATE = PromptTemplate(
    "You are an expert Neo4j Cypher 5 query writer.\n"
    "Generate exactly ONE read-only Cypher query that answers the question using the given schema.\n\n"

    "Schema:\n"
    "{schema}\n\n"

    "STRICT OUTPUT RULES:\n"
    "- Output ONLY the raw Cypher query.\n"
    "- No markdown code fences.\n"
    "- No explanations, comments, or prose.\n"
    "- Do NOT prefix the answer with 'Cypher', 'Query:', or any natural language.\n"
    "- Do NOT end with a semicolon.\n\n"

    "STRICT CYPHER RULES:\n"
    "- The query must be valid Neo4j Cypher 5.\n"
    "- The query must start with MATCH or OPTIONAL MATCH.\n"
    "- Use only labels, relationship types, and properties that appear in the schema.\n"
    "- Do NOT invent labels, relationship types, or properties.\n"
    "- Do NOT use UNION.\n"
    "- Do NOT use CREATE, MERGE, SET, DELETE, DETACH DELETE, DROP, REMOVE, LOAD CSV, CALL dbms, or CALL apoc.\n"
    "- Do NOT use SQL syntax: SELECT, FROM, JOIN, GROUP BY, HAVING.\n"
    "- If aggregation is needed, use Cypher aggregation with WITH or RETURN. Never write GROUP BY.\n"
    "- WHERE may appear only immediately after MATCH, OPTIONAL MATCH, or WITH.\n"
    "- ORDER BY may appear only after RETURN or WITH, and must appear before LIMIT.\n"
    "- Prefer simple MATCH / OPTIONAL MATCH / WHERE / WITH / RETURN / ORDER BY / LIMIT queries.\n"
    "- Return scalar fields or map projections, not raw nodes or relationships.\n"
    "- Always end with LIMIT 10.\n\n"

    "Question:\n"
    "{question}\n\n"

    "Cypher query:"
)
cypher_retriever = TextToCypherRetriever(
    graph_store=graph_store,
    llm=generation_llm,                          # query_llm = generation_llm (gpt-3.5-turbo)
    text_to_cypher_template=TEXT_TO_CYPHER_TEMPLATE,
    cypher_validator=safe_cypher_validator,              # 마크다운 펜스 제거
    include_text=True,
)

retriever = index.as_retriever(
    sub_retrievers=[cypher_retriever],
    similarity_top_k=SIMILARITY_TOP_K,   # [B] 다이어그램 top_k 일치 유지 (Cypher 결과 절단용)
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

n_failed = 0
for i, (question, reference) in enumerate(zip(questions, references), 1):
    print(f"  [{i:3d}/{len(questions)}] {question[:60]}...")

    # LLM 이 라벨 / 관계 타입을 환각하거나, 마크다운 정제 후에도 미세한 syntax 오류가
    # 남아있는 경우가 종종 있다. 한 질문이 전체 평가를 죽이지 않도록 빈 컨텍스트로 폴백.
    try:
        response = query_engine.query(question)
        retrieved_contexts = [n.node.get_content() for n in response.source_nodes]
        answer_text = str(response)
    except Exception as e:
        n_failed += 1
        print(f"        [retrieval failed → empty ctx] {type(e).__name__}: {str(e)[:120]}")
        retrieved_contexts = []
        answer_text = "(retrieval failed — no context returned by Cypher)"

    ragas_data["user_input"].append(question)
    ragas_data["response"].append(answer_text)
    ragas_data["retrieved_contexts"].append(retrieved_contexts)
    ragas_data["reference"].append(reference)

if n_failed:
    print(f"\n  주의: {n_failed}/{len(questions)} 질문에서 retrieval 이 실패했습니다 "
          f"(빈 컨텍스트로 평가됨 — RAGAS retrieval 메트릭이 0 으로 점수 매겨짐)")


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
print("  Graph RAG RAGAS 평가 결과 요약")
print("="*60)
print(summary.round(4).to_string())
print("="*60)
from datasets import load_dataset
import pandas as pd
import os
import openai

from llama_index.core import Document, PropertyGraphIndex
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.indices.property_graph import (
    SimpleLLMPathExtractor,
    ImplicitPathExtractor,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore

# for local llm, ollama
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding


from utils import load_config

# 1. Config
load_config()
openai.api_key = os.getenv("OPENAI_API_KEY")

TOKEN_CHUNK_SIZE = 1024 #2024
CHUNK_OVERLAP = 204

NEO4J_URI = os.getenv("NEO4J_URI_GRAPH", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME_GRAPH", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD_GRAPH")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE_GRAPH", "neo4j")

# for llm configuration
OLLAMA_MODEL_GRAPH = "qwen2.5:7b" #"qwen2.5:14b"
OLLAMA_MODEL_EMBEDDING="nomic-embed-text"

# 2. Load data
dataset = load_dataset("jamescalam/ai-arxiv")
df = pd.DataFrame(dataset["train"])

required_paper_titles = [
    "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
    "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter",
    "HellaSwag: Can a Machine Really Finish Your Sentence?",
    "LLaMA: Open and Efficient Foundation Language Models",
    "Measuring Massive Multitask Language Understanding",
    "CodeNet: A Large-Scale AI for Code Dataset for Learning a Diversity of Coding Tasks",
    "Task2Vec: Task Embedding for Meta-Learning",
    "GLM-130B: An Open Bilingual Pre-trained Model",
    "SuperGLUE: A Stickier Benchmark for General-Purpose Language Understanding Systems",
    "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism",
    "PAL: Program-aided Language Models",
    "RoBERTa: A Robustly Optimized BERT Pretraining Approach",
    "DetectGPT: Zero-Shot Machine-Generated Text Detection using Probability Curvature",
]

required_papers = df[df["title"].isin(required_paper_titles)]
remaining_papers = df[~df["title"].isin(required_paper_titles)].sample(
    n=40,
    random_state=123,
)

final_df = pd.concat([required_papers, remaining_papers], ignore_index=True)
documents = [Document(text=content) for content in final_df["content"]]


# 3. Chunking
parser = TokenTextSplitter(
    chunk_size=TOKEN_CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)
nodes = parser.get_nodes_from_documents(documents)



# 4. Models with OpenAI
llm = OpenAI(model="gpt-4o-mini", temperature=0.0)
embed_model = OpenAIEmbedding(model="text-embedding-3-large")

# 4. Models with Ollama
# (Graph triple extractor)
# llm = Ollama(
#     model=OLLAMA_MODEL_GRAPH,
#     request_timeout=600.0,
# )
#
# # Embedding (완전 로컬)
# embed_model = OllamaEmbedding(
#     model_name=OLLAMA_MODEL_EMBEDDING
# )

# 5. Neo4j graph store
graph_store = Neo4jPropertyGraphStore(
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    url=NEO4J_URI,
    database=NEO4J_DATABASE,
)


# 6. Graph extraction
extractors = [
    SimpleLLMPathExtractor(
        llm=llm,
        max_paths_per_chunk=20,
        num_workers=1,
    ),
    ImplicitPathExtractor(),
]


# 7. Build Graph RAG index
index = PropertyGraphIndex(
    nodes=nodes,
    kg_extractors=extractors,
    property_graph_store=graph_store,
    llm=llm,
    embed_model=embed_model,
    embed_kg_nodes=False,   # Graph RAG 비교군: 노드 임베딩 저장 안 함
    show_progress=True,
)

print("Graph DB 구축 완료")
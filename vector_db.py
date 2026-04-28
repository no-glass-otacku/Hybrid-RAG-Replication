# Importing necessary libraries for loading datasets, data manipulation, document processing, vector storage, and embeddings.
from datasets import load_dataset
import pandas as pd
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
import chromadb
from llama_index.core.node_parser import TokenTextSplitter
from utils import chunked_iterable, load_config
from llama_index.vector_stores.chroma import ChromaVectorStore
import openai
import os

# Hardcoded values for easy adjustment
TOKEN_CHUNK_SIZE = 1024
CHUNK_OVERLAP = 0

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
# documents = [Document(text=content) for content in required_papers['content']]

# Setup the embedding model
embed_model = OpenAIEmbedding(model="text-embedding-3-large")

chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Classic vector DB
existing_collections = [c.name for c in chroma_client.list_collections()]

if "ai_arxiv_full" not in existing_collections:
    print("Vector DB 구축 시작...")
    parser = TokenTextSplitter(chunk_size=TOKEN_CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    nodes = parser.get_nodes_from_documents(documents)

    chroma_collection = chroma_client.create_collection(
        name="ai_arxiv_full",
        metadata={"hnsw:space": "cosine"}
    )
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(
        nodes, storage_context=storage_context,
        embed_model=embed_model,
        use_async=True
    )
    print("Vector DB 구축 완료!")
else:
    print("이미 존재하는 컬렉션입니다. 건너뜁니다.")
    chroma_collection = chroma_client.get_collection("ai_arxiv_full")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=embed_model
    )
# rag_pipeline.py
import os
import time
import logging
from typing import List, Dict

from datasets import load_dataset
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm

from langchain_community.llms.huggingface_hub import HuggingFaceHub
from langchain_community.embeddings import HuggingFaceEmbeddings  # API-based embeddings
from langchain_community.vectorstores import Pinecone as LC_Pinecone

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_llm() -> HuggingFaceHub:
    """Return the LLM via the Hugging Face Inference API."""
    return HuggingFaceHub(
        repo_id="meta-llama/Llama-2-7b-chat-hf",  # Use your desired Llama 2 variant
        model_kwargs={
            "temperature": 0.7,
            "max_new_tokens": 150,
            "top_p": 0.9
        }
        # Optionally, you can pass the token explicitly:
        # huggingfacehub_api_token="your_token_here"
    )

def create_pinecone_index(pc: Pinecone, index_name: str, dimension: int) -> object:
    """Create a Pinecone index if it doesn't exist."""
    index_names = [idx['name'] for idx in pc.list_indexes()]
    if index_name not in index_names:
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        timeout = 60  # seconds
        start_time = time.time()
        while not pc.describe_index(index_name).status['ready']:
            if time.time() - start_time > timeout:
                raise TimeoutError("Timed out waiting for Pinecone index to be ready.")
            time.sleep(1)
    return pc.Index(index_name)

def process_dataset_and_upsert(index: object, embeddings: HuggingFaceEmbeddings, data, batch_size: int = 100):
    """Process the dataset and upsert embeddings into Pinecone."""
    for i in tqdm(range(0, len(data), batch_size), desc="Upserting embeddings"):
        i_end = min(len(data), i + batch_size)
        batch = data.iloc[i:i_end]
        ids = (batch['doi'].astype(str) + " - " + batch['chunk-id'].astype(str)).tolist()
        texts = batch['chunk'].tolist()
        embeds = embeddings.embed_documents(texts)
        meta_data = batch[['chunk', 'source', 'title']].to_dict(orient='records')
        index.upsert(vectors=list(zip(ids, embeds, meta_data)))

def augmented_prompt(query: str, vectorstore: LC_Pinecone, k: int = 3) -> str:
    """Construct an augmented prompt with context from the vectorstore."""
    results = vectorstore.similarity_search(query, k=k)
    context = "\n".join([res.page_content for res in results])
    return f"""Using the following context, answer the query below.

Contexts:
{context}

Query:
{query}"""

def load_data():
    """Load dataset and return as a pandas DataFrame."""
    dataset = load_dataset("jamescalam/llama-2-arxiv-papers-chunked", split="train")
    return dataset.to_pandas()

def build_vectorstore() -> (LC_Pinecone, Pinecone, str):
    """
    Build (or retrieve) the Pinecone index and upsert the dataset embeddings.
    Returns the LangChain vectorstore, the Pinecone client instance, and the index name.
    """
    data = load_data()
    pinecone_api = os.getenv('pinecone_api')
    if not pinecone_api:
        raise ValueError("pinecone_api not set in environment variables.")
    pc = Pinecone(api_key=pinecone_api)
    index_name = "llama-2-rag"
    index = create_pinecone_index(pc, index_name, dimension=384)
    logger.info(f"Pinecone index stats: {index.describe_index_stats()}")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    process_dataset_and_upsert(index, embeddings, data, batch_size=100)
    logger.info(f"Index stats after upsert: {index.describe_index_stats()}")
    vectorstore = LC_Pinecone(index, embeddings.embed_query, text_field='chunk')
    return vectorstore, pc, index_name

# rag_pipeline.py
import os
import time
import logging
import pandas as pd
from typing import List, Dict

from datasets import load_dataset
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm

from langchain_huggingface.llms import HuggingFaceEndpoint
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_pinecone.vectorstores import Pinecone as LC_Pinecone
from langchain_core.prompts import PromptTemplate

import spacy
from transformers import AutoTokenizer
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_llm() -> HuggingFaceEndpoint:
    # repo_id = "meta-llama/Llama-2-70b-chat-hf"
    repo_id = "meta-llama/Llama-3.3-70B-Instruct"
    max_new_tokens = 8192
    return HuggingFaceEndpoint(
        repo_id=repo_id,
        max_new_tokens=max_new_tokens,
        top_k=10,
        top_p=0.95,
        temperature=0.5,
        repetition_penalty=1.03,
        task="text-generation"
    )


def create_pinecone_index(pc: Pinecone, index_name: str, dimension: int) -> object:
    index_names = [idx['name'] for idx in pc.list_indexes()]

    if index_name not in index_names:
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        timeout = 60
        start_time = time.time()
        while not pc.describe_index(index_name).status['ready']:
            if time.time() - start_time > timeout:
                raise TimeoutError(
                    "Timed out waiting for Pinecone index to be ready.")
            time.sleep(1)
    return pc.Index(index_name)


nlp = spacy.load("en_core_web_sm")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


def dynamic_chunking(text, max_tokens=512, overlap=50):
    doc = nlp(text)
    chunks = []
    current_chunk = []
    token_count = 0

    for sent in doc.sents:
        sent_length = len(tokenizer.tokenize(sent.text))
        if token_count + sent_length <= max_tokens:
            current_chunk.append(sent.text)
            token_count += sent_length
        else:
            # If adding this sentence would exceed max_tokens, start a new chunk
            if current_chunk:
                # Add overlap if it's not the first chunk
                if chunks:
                    overlap_text = ' '.join(current_chunk[-overlap:])
                    chunks.append(' '.join(current_chunk) + ' ' + overlap_text)
                else:
                    chunks.append(' '.join(current_chunk))
            current_chunk = [sent.text]
            token_count = sent_length

    # Handle the last chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks


def process_dataset_and_upsert(index: object, embeddings: GPT4AllEmbeddings, data, batch_size: int = 100):
    """Process the dataset and upsert embeddings into Pinecone."""
    new_data = []
    for index, row in tqdm(data.iterrows(), desc="Dynamically chunking data"):
        original_summary = row['summary']
        chunks = dynamic_chunking(original_summary)

        for chunk in chunks:
            new_row = row.copy()
            new_row['summary'] = chunk  # Replace the summary with the chunk
            new_row['chunk-id'] = row['chunk-id'] + \
                f"-{chunks.index(chunk)}"  # Update chunk-id for uniqueness
            new_data.append(new_row)

# Convert list of dictionaries to DataFrame
    dynamically_chunked_data = pd.DataFrame(new_data)

# Now, update the embedding and upsert process with the new data
    for i in tqdm(range(0, len(dynamically_chunked_data), batch_size), desc="Upserting embeddings"):
        i_end = min(len(dynamically_chunked_data), i + batch_size)
        batch = dynamically_chunked_data.iloc[i:i_end]
        ids = (batch['doi'].astype(str) + " - " +
               batch['chunk-id'].astype(str)).tolist()
        texts = batch['summary'].tolist()
        embeds = embeddings.embed_documents(texts)
        meta_data = batch[['chunk', 'source', 'title',
                           'summary']].to_dict(orient='records')
        index.upsert(vectors=list(zip(ids, embeds, meta_data)))


def augmented_prompt(query: str, vectorstore: LC_Pinecone, k: int = 3) -> str:
    """Construct an augmented prompt with context from the vectorstore."""
    results = vectorstore.similarity_search(query, k=k)
    context = "\n".join([res.page_content for res in results])
    return f"""Using the following contexts, answer below the query.

Contexts:
{context}

query:
{query}
"""


def load_data():
    """Load dataset and return as a pandas DataFrame."""
    dataset = load_dataset(
        "jamescalam/llama-2-arxiv-papers-chunked", split="train")
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
    stats = index.describe_index_stats()
    model_name = "all-MiniLM-L6-v2.gguf2.f16.gguf"
    gpt4all_kwargs = {'allow_download': 'True'}
    embeddings = GPT4AllEmbeddings(
        model_name=model_name,
        gpt4all_kwargs=gpt4all_kwargs,

    )
    # embeddings = HuggingFaceEmbeddings(
    #     model_name="sentence-transformers/all-mpnet-base-v2")
    if stats.get("total_vector_count", 0) == 0:
        process_dataset_and_upsert(index, embeddings, data, batch_size=100)
        logger.info(
            f"Index stats after upsert: {index.describe_index_stats()}")
    else:
        logger.info("Data already upserted. Skipping re-upsert.")
    # process_dataset_and_upsert(index, embeddings, data, batch_size=100)
    logger.info(f"Index stats after upsert: {index.describe_index_stats()}")
    vectorstore = LC_Pinecone(index, embeddings, text_key='summary')
    return vectorstore, pc, index_name


if __name__ == "__main__":
    query = "Which are the different types of CNN?"
    vectorstore, pc, index_name = build_vectorstore()
    # pc.delete_index(index_name)
    aug_prompt = augmented_prompt(query=query, vectorstore=vectorstore)
    print(aug_prompt, len(aug_prompt))
    llm = load_llm()

    response = llm.invoke(aug_prompt)
    print(response)

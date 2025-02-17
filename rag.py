import os
import time
import logging
from typing import List, Dict

from langchain.schema import SystemMessage, AIMessage, HumanMessage
from langchain_community.llms.huggingface_hub import HuggingFaceHub
from datasets import load_dataset
from pinecone import Pinecone, ServerlessSpec
from langchain_community.embeddings import HuggingFaceEmbeddings  # API-based embeddings
from langchain_community.vectorstores import Pinecone as LC_Pinecone
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_llm() -> HuggingFaceHub:
    """Set up and return the LLM via the Hugging Face Inference API."""
    return HuggingFaceHub(
        repo_id="meta-llama/Llama-2-7b-chat-hf",  # Specify your desired Llama 2 variant
        model_kwargs={
            "temperature": 0.7,
            "max_new_tokens": 150,
            "top_p": 0.9
        }
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
        # Wait for the index to be ready (with a maximum timeout)
        timeout = 60  # seconds
        start_time = time.time()
        while not pc.describe_index(index_name).status['ready']:
            if time.time() - start_time > timeout:
                raise TimeoutError("Timed out waiting for Pinecone index to be ready.")
            time.sleep(1)
    return pc.Index(index_name)

def process_dataset_and_upsert(index: object, embeddings: HuggingFaceEmbeddings, data, batch_size: int = 100):
    """Process the dataset and upsert embeddings into Pinecone."""
    for i in tqdm(range(0, len(data), batch_size)):
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
    context = '\n'.join([res.page_content for res in results])
    return f"""Using the following context answer the query.

Contexts: {context}.

Query: {query}"""

def main():
    # Ensure your Hugging Face API token is set in the environment variable HUGGINGFACEHUB_API_TOKEN.
    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not hf_token:
        raise ValueError("HUGGINGFACEHUB_API_TOKEN not set in environment variables.")
    
    # Initialize the LLM via the Hugging Face Inference API.
    llm = load_llm()
    
    # Test a simple conversation.
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="Hai AI, how are you?"),
        AIMessage(content="I am great, How are you?"),
        HumanMessage(content="I'd like to understand about string theory.")
    ]
    # Combine messages into a single prompt for simplicity.
    conversation = "\n".join([f"{msg.__class__.__name__}: {msg.content}" for msg in messages])
    response = llm(conversation)
    logger.info(f"Chat response:\n{response}")

    # Load dataset and convert to a pandas DataFrame.
    dataset = load_dataset("jamescalam/llama-2-arxiv-papers-chunked", split='train')
    data = dataset.to_pandas()

    # Create Pinecone index.
    pinecone_api = os.getenv('pinecone_api')
    if not pinecone_api:
        raise ValueError("pinecone_api not set in environment variables.")
    pc = Pinecone(api_key=pinecone_api)
    index_name = "llama-2-rag"
    index = create_pinecone_index(pc, index_name, dimension=384)
    logger.info(f"Index stats: {index.describe_index_stats()}")

    # Set up API-based embeddings (using a SentenceTransformer model for example).
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    
    # Process the dataset and upsert embeddings.
    process_dataset_and_upsert(index, embeddings, data, batch_size=100)
    logger.info(f"Index stats after upsert: {index.describe_index_stats()}")

    # Set up LangChain vectorstore for querying.
    vectorstore = LC_Pinecone(index, embeddings.embed_query, text_field='chunk')
    query = "Can you tell me about the llama2?"
    prompt_text = augmented_prompt(query, vectorstore)
    logger.info(f"Augmented prompt:\n{prompt_text}")

    # Use the LLM to answer the augmented prompt.
    answer = llm(prompt_text)
    logger.info(f"LLM answer:\n{answer}")

    # Cleanup: Delete the index (if this is a temporary index).
    pc.delete_index(index_name)

if __name__ == '__main__':
    main()

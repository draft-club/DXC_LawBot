import logging
import torch
from llama_index.embeddings.voyageai import VoyageEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from constants import VOYAGE_API_KEY, OPENAI_API_KEY

# Clear CUDA cache
torch.cuda.empty_cache()

# Set up logging
logging.basicConfig(level=logging.INFO)

# Initialize embedding model
def initialize_embedding_model(model_name, voyage_api_key):
    embed_model = VoyageEmbedding(model_name=model_name, voyage_api_key=voyage_api_key)
    sample_text = "Sample text to determine embedding dimension"
    sample_embedding = embed_model.get_text_embedding(sample_text)
    embedding_dimension = len(sample_embedding)
    logging.info(f"Embedding dimension is {embedding_dimension}")
    return embed_model, embedding_dimension

# Initialize the embedding model and get the embedding dimension
voyage_embed_model, voyage_embedding_dimension = initialize_embedding_model("voyage-law-2", VOYAGE_API_KEY)

# Initialize LLMs
ollama_llm = Ollama(
    model="llama3.1",
    max_length=4096,
    temperature=0.7,
    top_p=0.9,
    device_map="auto",
    server_url="http://localhost:11434",
    request_timeout=4600.0,
)

if not OPENAI_API_KEY:
    raise ValueError("No API key found for OpenAI. Please set the OPENAI_API_KEY in the .env file.")
openai_llm = OpenAI(
    api_key=OPENAI_API_KEY,
    model="gpt-3.5-turbo",
    max_tokens=4096,
    temperature=0.7,
    top_p=0.9,
)

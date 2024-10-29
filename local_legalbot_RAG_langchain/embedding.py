from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_aws import BedrockEmbeddings


def get_embedding_function(embedding_type="bedrock"):
    """
    Returns the specified embedding function.

    Args:
        embedding_type (str): The type of embedding to use ("bedrock" or "ollama").

    Returns:
        embeddings: The embedding function object.
    """
    if embedding_type == "bedrock":
        embeddings = BedrockEmbeddings(
            credentials_profile_name="default", region_name="us-east-1"
        )
    elif embedding_type == "ollama":
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
    else:
        raise ValueError("Invalid embedding type. Choose 'bedrock' or 'ollama'.")

    return embeddings

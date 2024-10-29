from langchain_community.embeddings.ollama import OllamaEmbeddings


def get_embedding_function():
    """
    Returns the Ollama embedding function.

    Returns:
        embeddings: The Ollama embedding function object.
    """
    embeddings = OllamaEmbeddings(model="mxbai-embed-large") #opensource embedding Model
    return embeddings

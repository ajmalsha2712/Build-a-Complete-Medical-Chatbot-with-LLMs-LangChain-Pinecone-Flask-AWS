import os


def _req(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return val


def openai_api_key() -> str:
    return _req("OPENAI_API_KEY")


def pinecone_api_key() -> str:
    return _req("PINECONE_API_KEY")


def pinecone_index_name() -> str:
    return os.environ.get("PINECONE_INDEX_NAME", "medibot")


def pinecone_cloud() -> str:
    return os.environ.get("PINECONE_CLOUD", "aws")


def pinecone_region() -> str:
    return os.environ.get("PINECONE_REGION", "us-east-1")


def embedding_model() -> str:
    return os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")


def chat_model() -> str:
    return os.environ.get("CHAT_MODEL", "gpt-4o-mini")


def chunk_size() -> int:
    return int(os.environ.get("CHUNK_SIZE", "900"))


def chunk_overlap() -> int:
    return int(os.environ.get("CHUNK_OVERLAP", "150"))


def pinecone_dimension() -> int:
    # text-embedding-3-small => 1536
    return int(os.environ.get("PINECONE_DIMENSION", "1536"))


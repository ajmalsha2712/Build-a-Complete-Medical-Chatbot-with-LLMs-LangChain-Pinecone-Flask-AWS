from __future__ import annotations

from pinecone import Metric, Pinecone, ServerlessSpec, VectorType

from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from src import config


def ensure_pinecone_index() -> None:
    pc = Pinecone(api_key=config.pinecone_api_key())
    name = config.pinecone_index_name()

    existing = {idx["name"] for idx in pc.list_indexes()}
    if name in existing:
        return

    pc.create_index(
        name=name,
        dimension=config.pinecone_dimension(),
        metric=Metric.COSINE,
        spec=ServerlessSpec(cloud=config.pinecone_cloud(), region=config.pinecone_region()),
        vector_type=VectorType.DENSE,
    )


def get_vectorstore() -> PineconeVectorStore:
    ensure_pinecone_index()

    embeddings = OpenAIEmbeddings(
        api_key=config.openai_api_key(),
        model=config.embedding_model(),
    )

    return PineconeVectorStore(
        index_name=config.pinecone_index_name(),
        embedding=embeddings,
    )


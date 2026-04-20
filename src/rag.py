from __future__ import annotations

from typing import Dict, List

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

from src import config
from src.prompt import SYSTEM_PROMPT
from src.vectorstore import get_vectorstore

_HISTORIES: Dict[str, InMemoryChatMessageHistory] = {}


def _get_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in _HISTORIES:
        _HISTORIES[session_id] = InMemoryChatMessageHistory()
    return _HISTORIES[session_id]


def retrieve_docs(question: str) -> List[Document]:
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    return retriever.invoke(question)


def _format_docs(docs: List[Document]) -> str:
    parts = []
    for d in docs:
        src = d.metadata.get("source")
        page = d.metadata.get("page")
        parts.append(f"[Source: {src} p.{page}] {d.page_content}")
    return "\n\n".join(parts)


def build_rag_chain() -> RunnableWithMessageHistory:
    llm = ChatOpenAI(
        api_key=config.openai_api_key(),
        model=config.chat_model(),
        temperature=0.2,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("history"),
            (
                "human",
                "Question: {input}\n\n"
                "Retrieved context:\n{context}\n\n"
                "Answer with medical safety in mind.",
            ),
        ]
    )

    def _retrieve(inputs: Dict[str, str]) -> Dict[str, str]:
        docs = retrieve_docs(inputs["input"])
        return {"input": inputs["input"], "context": _format_docs(docs)}

    chain = RunnableLambda(_retrieve) | prompt | llm

    return RunnableWithMessageHistory(
        chain,
        lambda session_id: _get_history(session_id),
        input_messages_key="input",
        history_messages_key="history",
    )


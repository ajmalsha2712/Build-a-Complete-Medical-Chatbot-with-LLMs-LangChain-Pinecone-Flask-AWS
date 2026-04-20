import os

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src import config
from src.vectorstore import get_vectorstore


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    load_dotenv(dotenv_path=os.path.join(base_dir, ".env"), override=False)

    pdf_path = os.environ.get("PDF_PATH", "data/Medical_book.pdf")
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found at: {pdf_path}")

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size(),
        chunk_overlap=config.chunk_overlap(),
    )
    chunks = splitter.split_documents(docs)

    # Add stable metadata
    for i, d in enumerate(chunks):
        d.metadata["source"] = os.path.basename(pdf_path)
        d.metadata["chunk"] = i

    vs = get_vectorstore()
    vs.add_documents(chunks)

    print(f"Loaded {len(docs)} pages, upserted {len(chunks)} chunks into Pinecone index '{config.pinecone_index_name()}'.")


if __name__ == "__main__":
    main()


# MediBot – Medical RAG Chatbot

This project ingests `data/Medical_book.pdf` into Pinecone and serves a Flask web UI + JSON API for chatting with citations.

## Setup

Create a virtualenv and install dependencies:

```bash
pip install -r requirements.txt
```

If you want a local embedding fallback (bigger install), you can additionally install `sentence-transformers` (this will pull PyTorch).

Create your env file:

```bash
copy .env.example .env
```

Fill in:
- `OPENAI_API_KEY`
- `PINECONE_API_KEY`

## Ingest the PDF into Pinecone

```bash
python ingest.py
```

## Run the chatbot

```bash
python app.py
```

Open `http://localhost:8080`.

## API

- `POST /api/chat` with JSON `{ "message": "..." }`
- `GET /health`


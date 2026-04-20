import os
import uuid
from typing import Any, Dict, List

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, session

from src.rag import build_rag_chain

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(_BASE_DIR, ".env"), override=False)


def create_app() -> Flask:
    base_dir = _BASE_DIR
    app = Flask(
        __name__,
        template_folder=os.path.join(base_dir, "templates"),
        static_folder=os.path.join(base_dir, "static"),
    )
    app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me")

    chain = None

    def get_chain():
        nonlocal chain
        if chain is None:
            chain = build_rag_chain()
        return chain

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.post("/api/chat")
    def chat():
        payload = request.get_json(silent=True) or {}
        message = (payload.get("message") or "").strip()
        if not message:
            return jsonify({"error": "Missing 'message'"}), 400

        if "chat_id" not in session:
            session["chat_id"] = str(uuid.uuid4())
        chat_id = session["chat_id"]

        try:
            ai_msg = get_chain().invoke(
                {"input": message},
                config={"configurable": {"session_id": chat_id}},
            )
        except RuntimeError as e:
            return jsonify({"error": str(e)}), 500

        answer = getattr(ai_msg, "content", str(ai_msg))

        # Provide citations by retrieving again (fast + keeps chain simpler)
        from src.rag import retrieve_docs

        sources: List[Dict[str, Any]] = []
        for doc in retrieve_docs(message):
            md = dict(doc.metadata or {})
            sources.append({"source": md.get("source"), "page": md.get("page")})

        return jsonify({"answer": answer, "sources": sources})

    @app.get("/health")
    def health():
        return jsonify({"ok": True})

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8080")), debug=True)


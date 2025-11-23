"""
Vectorize sentences with ChromaDB and send retrieved context to llama3.1 (Ollama).

Prerequisites:
  - Ollama installed and running locally: https://ollama.com
  - The model "llama3.1" pulled: `ollama pull llama3.1`

Run example:
  python -m flora.vectorize
"""

import subprocess
import shutil
import textwrap
import chromadb
from chromadb.config import Settings

def add_sentences_to_chromadb(sentences, collection_name="sentences_collection"):
    client = chromadb.Client(Settings(
        anonymized_telemetry=False,
    ))

    collection = client.get_or_create_collection(name=collection_name)

    ids = [f"sentence-{i}" for i in range(len(sentences))]

    collection.add(
        documents=sentences,
        ids=ids,
    )

    return collection


def _ensure_ollama_available():
    """Return None if available, else a human-friendly error string."""
    if shutil.which("ollama") is None:
        return (
            "Ollama CLI not found. Please install from https://ollama.com and ensure "
            '"ollama" is on your PATH.'
        )
    return None


def call_llama_with_context(question: str, retrieved_docs: list[str]) -> str:
    """Call llama3.1 via Ollama, providing retrieved_docs as context.

    Args:
        question: The user question/query.
        retrieved_docs: List of strings (documents) retrieved from ChromaDB.

    Returns:
        The model's response as a string. If Ollama is unavailable or an error occurs,
        a helpful message is returned instead of raising.
    """
    missing = _ensure_ollama_available()
    if missing:
        return missing

    context_block = "\n\n".join(f"- {d}" for d in retrieved_docs)
    prompt = textwrap.dedent(
        f"""
        You are a helpful assistant. Use only the provided context to answer the user's
        question. If the answer is not in the context, say you don't know.

        Context:
        {context_block}

        Question: {question}
        Answer:
        """
    ).strip()

    try:
        # Run: ollama run llama3.1
        proc = subprocess.run(
            ["ollama", "run", "llama3.1"],
            input=prompt,
            text=True,
            capture_output=True,
            check=False,
        )
    except Exception as e:
        return f"Failed to invoke Ollama: {e}"

    if proc.returncode != 0:
        # Include stderr to help diagnose
        return f"Ollama returned non-zero exit code {proc.returncode}: {proc.stderr.strip()}"

    return proc.stdout.strip()

if __name__ == "__main__":
    sentences = [
        "How are you?",
        "What did you do today?",
        "What did you have for dinner?",
        "What's on your agenda?"
    ]

    collection = add_sentences_to_chromadb(sentences)

    # Example query
    user_question = "What did you have for dinner?"

    results = collection.query(
        query_texts=[user_question],
        n_results=3,
    )

    # Extract top documents from Chroma query results
    # results['documents'] is a list aligned with query_texts; take the first
    retrieved = results.get("documents", [[]])
    top_docs = retrieved[0] if retrieved and isinstance(retrieved[0], list) else []

    print("Retrieved context:")
    for i, d in enumerate(top_docs, 1):
        print(f"  {i}. {d}")

    print("\nAsking llama3.1 with retrieved context...\n")
    answer = call_llama_with_context(user_question, top_docs)
    print("Answer:\n" + answer)
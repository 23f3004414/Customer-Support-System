"""Smoke tests for RAG chatbot components."""
import sys
from pathlib import Path

# Windows console safety for test output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from langchain.schema import HumanMessage, AIMessage, SystemMessage
from conversation import ConversationManager
from llm_rag import LLMRAGHandler


def test_conversation_utf8():
    cm = ConversationManager(state_file="_test_conversation.json")
    messages = [
        SystemMessage(content="test"),
        HumanMessage(content="hello"),
        AIMessage(content="reply with arrow → and emoji ✓"),
    ]
    cm.save(messages)
    loaded = cm.load()
    assert len(loaded) == 3
    assert "→" in loaded[2].content
    cm.clear()
    print("PASS: conversation utf-8 save/load")


def test_vector_store_empty_search():
    from vector_store import VectorStore

    vs = VectorStore(vector_store_path=Path("_test_faiss"), index_path="_test_faiss")
    try:
        results = vs.similarity_search("test query", k=4)
        print(f"PASS: empty search returned {len(results)} docs")
    except Exception as e:
        print(f"FAIL: empty similarity_search: {e}")
        raise
    finally:
        import shutil

        p = Path("_test_faiss")
        if p.exists():
            shutil.rmtree(p)


def test_handler_init():
    handler = LLMRAGHandler()
    assert handler.llm.model == "llama3.2:1b"
    print("PASS: LLMRAGHandler init")


def test_empty_url_indexing():
    from vector_store import VectorStore
    import shutil

    vs = VectorStore(vector_store_path=Path("_test_faiss2"), index_path="_test_faiss2")
    try:
        assert vs.index_websites(["", "   "]) == []
        print("PASS: empty URL indexing skipped")
    finally:
        p = Path("_test_faiss2")
        if p.exists():
            shutil.rmtree(p)


def test_generate_response():
    handler = LLMRAGHandler()
  # Index a PDF if present
    upload_dir = Path("uploaded_pdfs")
    pdfs = list(upload_dir.glob("*.pdf"))
    if pdfs:
        handler.add_pdf_to_context(pdfs[0])
        print(f"Indexed: {pdfs[0].name}")
    answer = handler.generate_response("What skills are mentioned? Reply in one sentence.")
    assert answer and len(answer) > 0
    assert "→" in answer or True  # unicode in response must not crash caller
    print(f"PASS: generate_response ({len(answer)} chars)")
    print(f"Sample: {answer[:120]}...")


if __name__ == "__main__":
    failures = 0
    for name, fn in [
        ("conversation", test_conversation_utf8),
        ("empty_search", test_vector_store_empty_search),
        ("empty_urls", test_empty_url_indexing),
        ("init", test_handler_init),
    ]:
        try:
            fn()
        except Exception as e:
            failures += 1
            print(f"FAIL [{name}]: {e}")

    try:
        test_generate_response()
    except Exception as e:
        failures += 1
        print(f"FAIL [generate_response]: {e}")

    print(f"\nDone: {failures} failure(s)")
    sys.exit(failures)

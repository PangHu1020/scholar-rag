"""Test agent end-to-end: build graph and run queries."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_openai import ChatOpenAI
from config import Config
from rag.retrieval import Retriever
from rag.citation import CitationExtractor
from agent.graph import build_graph
from agent.checkpointer import create_memory_checkpointer


def create_retriever_tool(retriever: Retriever):
    class RetrieverTool:
        def invoke(self, query: str):
            return retriever.retrieve(
                query=query,
                k=Config.TOP_K,
                use_hyde=False,
                rerank=True,
                expand_parent=True,
                rrf_k=Config.RRF_K,
                fetch_k=Config.FETCH_K,
            )
    return RetrieverTool()


def main():
    print("=" * 70)
    print("  Agent End-to-End Test")
    print("=" * 70)

    print("\n[1] Initializing LLM...")
    t0 = time.time()
    llm = ChatOpenAI(
        base_url=Config.LLM_BASE_URL,
        model=Config.LLM_MODEL,
        api_key=Config.LLM_API_KEY,
        temperature=Config.LLM_TEMPERATURE,
        max_tokens=Config.LLM_MAX_TOKENS,
    )
    resp = llm.invoke("Hi")
    print(f"  LLM ready ({time.time() - t0:.1f}s): {resp.content[:50]}")

    print("\n[2] Initializing Retriever...")
    t0 = time.time()
    retriever = Retriever(
        embedding_model=Config.EMBEDDING_MODEL,
        reranker_model=Config.RERANKER_MODEL,
        milvus_uri=Config.MILVUS_URI,
        collection_name=Config.COLLECTION_NAME,
        enable_cache=Config.ENABLE_CACHE,
    )
    retriever_tool = create_retriever_tool(retriever)
    print(f"  Retriever ready ({time.time() - t0:.1f}s)")

    print("\n[3] Building graph...")
    checkpointer = create_memory_checkpointer()
    graph = build_graph(
        llm=llm,
        retriever=retriever_tool,
        citation_extractor=CitationExtractor,
        max_retries=Config.MAX_RETRIES,
        checkpointer=checkpointer,
    )
    print(f"  Graph nodes: {list(graph.get_graph().nodes.keys())}")

    config = {"configurable": {"thread_id": "test-session-1"}}

    queries = [
        "What is DualPath and how does it improve LLM inference?",
        "What are the experimental results on DeepSeek 660B?",
    ]

    for i, query in enumerate(queries, 1):
        print(f"\n{'=' * 70}")
        print(f"  Turn {i}: {query}")
        print("=" * 70)

        t0 = time.time()
        result = graph.invoke({"query": query}, config=config)
        elapsed = time.time() - t0

        print(f"\n  [Answer] ({elapsed:.1f}s)")
        answer = result.get("answer", "NO ANSWER")
        for line in answer.split("\n"):
            print(f"    {line}")

        citations = result.get("citations", [])
        if citations:
            print(f"\n  [Citations] ({len(citations)} sources)")
            for ci, c in enumerate(citations[:3], 1):
                print(f"    {ci}. {CitationExtractor.format_citation(c)}")

        sub_answers = result.get("sub_answers", [])
        print(f"\n  [Sub-queries] ({len(sub_answers)} answered)")
        for sa in sub_answers:
            print(f"    - {sa['query']}: {sa['answer'][:80]}...")

        msgs = result.get("messages", [])
        print(f"\n  [Messages] {len(msgs)} in history")
        summary = result.get("summary", "")
        if summary:
            print(f"  [Summary] {summary[:100]}...")

    print(f"\n{'=' * 70}")
    print("  Test completed.")
    print("=" * 70)


if __name__ == "__main__":
    main()

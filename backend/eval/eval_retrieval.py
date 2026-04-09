"""Retrieval evaluation: dataset-agnostic core functions.

Test case formats:
    1. Chunk ID matching (legacy, not recommended):
       {"query": str, "relevant_ids": list[str]}
    
    2. LLM-based answer matching (recommended for custom datasets):
       {"query": str, "reference_answer": str}
       Use with hit_fn=is_hit_answer for chunk-agnostic evaluation.
    
    3. Page-based matching (MMDocIR):
       {"query": str, "relevant_pages": list[int], "paper_id": str, ...}
       Use with hit_fn=is_hit_page from mmdocir_adapter.

For custom hit functions, signature: hit_fn(doc: Document, case: dict) -> bool
"""

from typing import Callable, Optional
from langchain_core.documents import Document
from rag.retrieval import Retriever


def is_hit_answer(doc: Document, case: dict, llm=None) -> bool:
    """LLM-based hit function: check if retrieved chunk contains info to answer the query.
    
    Args:
        doc: Retrieved document with page_content and metadata.
        case: Test case with "query" and "reference_answer".
        llm: LangChain LLM for evaluation (defaults to Config LLM).
    
    Returns:
        True if the chunk contains information needed to answer the query.
    """
    from langchain_openai import ChatOpenAI
    from config import Config
    
    _llm = llm or ChatOpenAI(
        base_url=Config.LLM_BASE_URL,
        model=Config.LLM_MODEL,
        api_key=Config.LLM_API_KEY,
        temperature=0,
    )
    
    query = case.get("query", "")
    reference = case.get("reference_answer", "")
    chunk_text = doc.page_content
    
    prompt = f"""Given a query and a reference answer, determine if the provided text chunk contains sufficient information to answer the query.

Query: {query}

Reference Answer: {reference}

Text Chunk:
{chunk_text}

Does this chunk contain information that would help answer the query? Consider:
1. Does it contain key facts mentioned in the reference answer?
2. Does it provide relevant context for answering the query?
3. Could someone use this chunk to formulate a correct answer?

Answer only "yes" or "no"."""
    
    response = _llm.invoke(prompt).content.strip().lower()
    return "yes" in response


def calculate_metrics(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> dict:
    """Recall@k, Precision@k, MRR, MAP from chunk_id lists."""
    if not relevant_ids:
        return {"recall": 0.0, "precision": 0.0, "mrr": 0.0, "ap": 0.0}

    retrieved_set = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    hits = len(retrieved_set & relevant_set)

    recall = hits / len(relevant_set)
    precision = hits / k if k > 0 else 0.0

    mrr = 0.0
    for i, rid in enumerate(retrieved_ids[:k], 1):
        if rid in relevant_set:
            mrr = 1.0 / i
            break

    ap, hits_at_k = 0.0, 0
    for i, rid in enumerate(retrieved_ids[:k], 1):
        if rid in relevant_set:
            hits_at_k += 1
            ap += hits_at_k / i
    ap = ap / len(relevant_set) if relevant_set else 0.0

    return {"recall": recall, "precision": precision, "mrr": mrr, "ap": ap}


def calculate_metrics_from_hits(hits: list[bool], num_relevant: int, k: int) -> dict:
    """Recall@k, Precision@k, MRR, MAP from a boolean hit list."""
    if num_relevant == 0:
        return {"recall": 0.0, "precision": 0.0, "mrr": 0.0, "ap": 0.0}

    hits_k = hits[:k]
    total_hits = sum(hits_k)

    recall = min(total_hits / num_relevant, 1.0)
    precision = total_hits / k if k > 0 else 0.0

    mrr = 0.0
    for i, h in enumerate(hits_k, 1):
        if h:
            mrr = 1.0 / i
            break

    ap, running = 0.0, 0
    for i, h in enumerate(hits_k, 1):
        if h:
            running += 1
            ap += running / i
    ap = ap / num_relevant

    return {"recall": recall, "precision": precision, "mrr": mrr, "ap": ap}


def evaluate_retrieval(
    retriever: Retriever,
    test_cases: list[dict],
    k: int = 5,
    fetch_k: int = 20,
    hit_fn: Optional[Callable[[dict, dict], bool]] = None,
    llm = None,
    verbose: bool = False,
) -> dict:
    """Evaluate retrieval on test cases.

    Args:
        retriever: Retriever instance.
        test_cases: List of dicts. Must contain "query" and either:
            - "relevant_ids": list[str]  for chunk_id matching
            - any fields consumed by hit_fn
        k: Top-k for metrics.
        fetch_k: Candidates before reranking.
        hit_fn: Optional callable(doc: Document, case: dict) -> bool.
                When provided, used instead of chunk_id matching.
        llm: LangChain LLM for hit_fn evaluation (if hit_fn needs it).
        verbose: Print per-query details.

    Returns:
        Dict with averaged Recall@k, Precision@k, MRR, MAP.
    """
    all_metrics = []

    for i, case in enumerate(test_cases, 1):
        query = case["query"]
        results: list[Document] = retriever.retrieve(
            query, k=k, fetch_k=fetch_k, rerank=True, expand_parent=True
        )

        if hit_fn is not None:
            if hit_fn == is_hit_answer:
                hits = [hit_fn(doc, case, llm=llm) for doc in results]
            else:
                hits = [hit_fn(doc, case) for doc in results]
            if "relevant_pages" in case or "relevant_layouts" in case:
                num_relevant = len(case.get("relevant_pages", case.get("relevant_layouts", [])))
                num_relevant = max(num_relevant, 1)
            else:
                # For answer-based evaluation without ground truth count,
                # assume we need at least k/2 relevant docs (or min 2)
                num_relevant = max(k // 2, 2)
            metrics = calculate_metrics_from_hits(hits, num_relevant, k)
        else:
            relevant_ids = case.get("relevant_ids", [])
            if not relevant_ids:
                continue
            retrieved_ids = [doc.metadata.get("chunk_id", "") for doc in results]
            metrics = calculate_metrics(retrieved_ids, relevant_ids, k)

        all_metrics.append(metrics)

        if verbose:
            print(f"\n[{i}] {query[:80]}")
            if hit_fn is not None:
                print(f"  hits={hits[:k]}  recall={metrics['recall']:.2%}  mrr={metrics['mrr']:.4f}")
            else:
                print(f"  recall={metrics['recall']:.2%}  precision={metrics['precision']:.2%}")

    n = len(all_metrics)
    if n == 0:
        return {"recall@k": 0.0, "precision@k": 0.0, "mrr": 0.0, "map": 0.0, "num_queries": 0}

    def _avg(vals): return sum(vals) / len(vals) if vals else 0.0
    return {
        f"recall@{k}": _avg([m["recall"] for m in all_metrics]),
        f"precision@{k}": _avg([m["precision"] for m in all_metrics]),
        "mrr": _avg([m["mrr"] for m in all_metrics]),
        "map": _avg([m["ap"] for m in all_metrics]),
        "num_queries": n,
    }

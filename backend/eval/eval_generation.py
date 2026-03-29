"""Evaluate agent generation quality using RAGAS metrics."""

import sys
import time
import warnings
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore", category=DeprecationWarning)

from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    FactualCorrectness,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_openai import ChatOpenAI
from config import Config
from rag.retrieval import Retriever
from rag.citation import CitationExtractor
from agent.graph import build_graph

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


EVAL_CASES = [
    {
        "query": "What is DualPath and what problem does it solve?",
        "reference": (
            "DualPath is an agentic LLM inference framework that addresses the "
            "storage bandwidth bottleneck in PD-disaggregated architectures by "
            "introducing a dual-path KV-Cache loading mechanism, which distributes "
            "network load across both prefill and decode engines."
        ),
    },
    {
        "query": "How does DualPath improve LLM inference throughput?",
        "reference": (
            "DualPath improves throughput by introducing a storage-to-decode loading "
            "path alongside the traditional storage-to-prefill path, using a global "
            "workload-aware scheduler to dynamically balance load, achieving up to "
            "1.87x offline throughput improvement and 1.96x higher agent runs per second."
        ),
    },
    {
        "query": "What is the dual-path KV-Cache loading mechanism?",
        "reference": (
            "The dual-path mechanism loads KV-Cache via two paths: the conventional "
            "storage-to-prefill path and a novel storage-to-decode path where data is "
            "loaded into decoding engines first, then transferred to prefill engines "
            "via RDMA over the compute network."
        ),
    },
    {
        "query": "What are the experimental results of DualPath on DeepSeek 660B?",
        "reference": (
            "On DeepSeek 660B, DualPath achieves up to 1.87x throughput improvement "
            "in offline inference and 2.25x higher actions per second in online serving, "
            "with significant TTFT reduction while maintaining stable TPOT."
        ),
    },
    {
        "query": "How does the DualPath scheduler balance load across prefill and decode engines?",
        "reference": (
            "DualPath uses a global workload-aware scheduler that dynamically selects "
            "between the storage-to-prefill and storage-to-decode paths based on "
            "real-time load conditions, preventing network congestion and avoiding "
            "interference with latency-critical model execution."
        ),
    },
]


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


def collect_samples(graph, eval_cases: list[dict]) -> list[SingleTurnSample]:
    samples = []
    for i, case in enumerate(eval_cases, 1):
        query = case["query"]
        print(f"\n[{i}/{len(eval_cases)}] {query}")

        t0 = time.time()
        result = graph.invoke({
            "query": query,
            "messages": [],
            "summary": "",
            "documents": [],
            "sub_queries": [],
            "sub_answers": [],
            "answer": "",
            "citations": [],
        })
        elapsed = time.time() - t0

        answer = result.get("answer", "")
        sub_answers = result.get("sub_answers", [])
        contexts = []
        for sa in sub_answers:
            if sa.get("answer"):
                contexts.append(sa["answer"])

        print(f"  Time: {elapsed:.1f}s | Sub-queries: {len(sub_answers)} | Contexts: {len(contexts)}")
        print(f"  Answer preview: {answer[:120]}...")

        sample = SingleTurnSample(
            user_input=query,
            response=answer,
            retrieved_contexts=contexts if contexts else [answer],
            reference=case.get("reference"),
        )
        samples.append(sample)

    return samples


def main():
    print("=" * 70)
    print("  RAGAS Generation Evaluation")
    print("=" * 70)

    print("\n[1] Initializing components...")
    llm = ChatOpenAI(
        base_url=Config.LLM_BASE_URL,
        model=Config.LLM_MODEL,
        api_key=Config.LLM_API_KEY,
        temperature=Config.LLM_TEMPERATURE,
        max_tokens=Config.LLM_MAX_TOKENS,
    )

    retriever = Retriever(
        embedding_model=Config.EMBEDDING_MODEL,
        reranker_model=Config.RERANKER_MODEL,
        milvus_uri=Config.MILVUS_URI,
        collection_name=Config.COLLECTION_NAME,
        enable_cache=Config.ENABLE_CACHE,
    )
    retriever_tool = create_retriever_tool(retriever)

    graph = build_graph(
        llm=llm,
        retriever=retriever_tool,
        citation_extractor=CitationExtractor,
        max_retries=Config.MAX_RETRIES,
    )

    print("\n[2] Running agent on evaluation queries...")
    samples = collect_samples(graph, EVAL_CASES)

    print(f"\n\n[3] Evaluating with RAGAS (evaluator LLM: {Config.LLM_MODEL})...")
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(
        base_url=Config.LLM_BASE_URL,
        model=Config.LLM_MODEL,
        api_key=Config.LLM_API_KEY,
        temperature=0,
    ))
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)
    )

    metrics = [
        Faithfulness(),
        AnswerRelevancy(),
        ContextPrecision(),
        FactualCorrectness(),
    ]

    dataset = EvaluationDataset(samples=samples)
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        show_progress=True,
    )

    print(f"\n{'=' * 70}")
    print("  RAGAS Evaluation Results")
    print("=" * 70)

    df = result.to_pandas()
    metric_cols = [c for c in df.columns if c not in ("user_input", "response", "retrieved_contexts", "reference", "reference_contexts")]

    for col in metric_cols:
        vals = df[col].dropna()
        if len(vals) > 0:
            print(f"  {col:30s}  avg={vals.mean():.4f}  min={vals.min():.4f}  max={vals.max():.4f}")

    print(f"\n  Samples evaluated: {len(df)}")
    print("=" * 70)

    output_path = Path(__file__).parent / "ragas_results.csv"
    df.to_csv(output_path, index=False)
    print(f"\n  Detailed results saved to: {output_path}")

    print("\n  Per-query scores:")
    for _, row in df.iterrows():
        q = row["user_input"][:60]
        scores = "  ".join(f"{c}={row[c]:.2f}" for c in metric_cols if not str(row.get(c)) == "nan")
        print(f"    {q:60s}  {scores}")


if __name__ == "__main__":
    main()

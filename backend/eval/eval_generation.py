"""Generation evaluation: dataset-agnostic RAGAS-based functions."""

import time
import warnings
import logging

warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.getLogger("langchain_milvus").setLevel(logging.ERROR)

from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
from ragas import evaluate as ragas_evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, FactualCorrectness
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from config import Config
from rag.factory import EmbeddingService

logger = logging.getLogger(__name__)

DEFAULT_METRICS = [Faithfulness(), AnswerRelevancy(), FactualCorrectness()]


def collect_samples(graph, eval_cases: list[dict], verbose: bool = True) -> list[SingleTurnSample]:
    """Run the agent graph on eval_cases and collect RAGAS samples.

    Each case must have "query"; optionally "reference" or "reference_answer" for FactualCorrectness.
    """
    import asyncio

    async def _run_all():
        samples = []
        for i, case in enumerate(eval_cases, 1):
            query = case["query"]
            reference = case.get("reference") or case.get("reference_answer")
            if verbose:
                print(f"\n[{i}/{len(eval_cases)}] {query}")

            t0 = time.time()
            result = await graph.ainvoke({
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

            # Collect retrieved contexts from each sub-agent's documents
            contexts = []
            for sa in sub_answers:
                for doc in sa.get("documents", []):
                    if isinstance(doc, str) and doc.strip():
                        contexts.append(doc)

            if verbose:
                print(f"  {elapsed:.1f}s | sub-queries={len(sub_answers)} | contexts={len(contexts)}")
                print(f"  {answer[:120]}...")

            samples.append(SingleTurnSample(
                user_input=query,
                response=answer,
                retrieved_contexts=contexts or [""],
                reference=reference,
            ))
        return samples

    return asyncio.run(_run_all())


def evaluate_generation(
    samples: list[SingleTurnSample],
    llm=None,
    metrics=None,
) -> dict:
    """Run RAGAS evaluation on collected samples.

    Args:
        samples: From collect_samples().
        llm: LangChain LLM for RAGAS evaluation (defaults to Config LLM).
        metrics: RAGAS metric list (defaults to DEFAULT_METRICS).

    Returns:
        Dict of {metric_name: avg_score} plus per-sample DataFrame under "dataframe".
    """
    from langchain_openai import ChatOpenAI

    _llm = llm or ChatOpenAI(
        base_url=Config.LLM_BASE_URL,
        model=Config.LLM_MODEL,
        api_key=Config.LLM_API_KEY,
        temperature=0,
    )
    _metrics = metrics or DEFAULT_METRICS

    evaluator_llm = LangchainLLMWrapper(_llm)
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        EmbeddingService.get_embeddings(Config.EMBEDDING_MODEL)
    )

    dataset = EvaluationDataset(samples=samples)
    result = ragas_evaluate(
        dataset=dataset,
        metrics=_metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        show_progress=True,
    )

    df = result.to_pandas()
    skip_cols = {"user_input", "response", "retrieved_contexts", "reference", "reference_contexts"}
    metric_cols = [c for c in df.columns if c not in skip_cols]

    summary = {"dataframe": df}
    for col in metric_cols:
        vals = df[col].dropna()
        if len(vals) > 0:
            summary[col] = float(vals.mean())

    return summary

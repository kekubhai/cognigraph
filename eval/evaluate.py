import json
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eval.eval_dataset import TEST_QUERIES
from eval.metrics import evaluate_answer
from langgraph_flow import run_flow
from init import get_embeddings


def run_eval(verbose: bool = True) -> list[dict]:
    embedder = get_embeddings()
    results = []

    for i, item in enumerate(TEST_QUERIES):
        if verbose:
            print(f"[{i+1}/{len(TEST_QUERIES)}] {item['query']}")

        flow_result = run_flow(item["query"])
        generated_answer = flow_result.get("summary", "") or ""
        retrieved_chunks = flow_result.get("documents", []) or []

        metrics = evaluate_answer(
            query=item["query"],
            generated_answer=generated_answer,
            expected_answer=item["expected_answer"],
            expected_topics=item["expected_topics"],
            retrieved_chunks=retrieved_chunks,
            embedder=embedder,
        )

        if verbose:
            _print_metrics(generated_answer, metrics)

        results.append(metrics)

    return results


def _print_metrics(answer: str, metrics: dict):
    print(f"  Answer: {answer[:80]}{'...' if len(answer) > 80 else ''}")
    print(f"  Semantic sim:  {metrics['semantic_similarity']:.3f}")
    print(f"  Token F1:      {metrics['token_f1']:.3f}")
    print(f"  Topics:        {metrics['topic_coverage']:.3f}")
    print(f"  Faithfulness:  {metrics['faithfulness']:.3f}")
    print(f"  Context rel:   {metrics['context_relevance']:.3f}")
    print(f"  Chunks:        {metrics['num_chunks']}")
    print()


def summarize(results: list[dict]) -> dict:
    keys = ["semantic_similarity", "token_f1", "topic_coverage", "faithfulness", "context_relevance"]
    avg = {}
    for k in keys:
        vals = [r[k] for r in results]
        avg[k] = round(sum(vals) / len(vals), 4)
    avg["total_queries"] = len(results)
    avg["avg_chunks"] = round(sum(r["num_chunks"] for r in results) / len(results), 1)
    avg["avg_answer_length"] = round(sum(r["answer_length"] for r in results) / len(results), 1)
    return avg


if __name__ == "__main__":
    results = run_eval(verbose=True)
    summary = summarize(results)

    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k:25s} {v:.4f}")
        else:
            print(f"  {k:25s} {v}")

    filename = f"eval_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, "w") as f:
        json.dump({"results": results, "summary": summary}, f, indent=2)
    print(f"\nReport saved to {filename}")

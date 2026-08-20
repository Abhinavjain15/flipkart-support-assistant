"""
Retrieval evaluation: Precision@3 and Recall@3, computed at the document level.
Each retrieved chunk is mapped back to its parent doc and deduplicated before
scoring against the Task 1 answer key (knowledge_base/policies.py).
"""

from embed_index import search
from knowledge_base.policies import RETRIEVAL_EVAL_QUERIES

K = 3


def evaluate_query(query: str, relevant_docs: set, k: int = K) -> dict:
    results = search(query, k=k)
    retrieved_docs_ordered = [r["parent_doc"] for r in results]
    retrieved_docs_dedup = list(
        dict.fromkeys(retrieved_docs_ordered)
    )  # dedupe, preserve order

    hits = [d for d in retrieved_docs_dedup if d in relevant_docs]
    num_hits = len(hits)

    precision_at_k = (
        num_hits / len(retrieved_docs_dedup) if retrieved_docs_dedup else 0.0
    )
    recall_at_k = num_hits / len(relevant_docs) if relevant_docs else 0.0

    return {
        "query": query,
        "retrieved_docs": retrieved_docs_dedup,
        "relevant_docs": relevant_docs,
        "hits": hits,
        "precision_at_k": precision_at_k,
        "recall_at_k": recall_at_k,
    }


if __name__ == "__main__":
    all_results = []
    for item in RETRIEVAL_EVAL_QUERIES:
        r = evaluate_query(item["query"], item["relevant_docs"])
        all_results.append(r)

        print(f"Query: {r['query']}")
        print(f"  Retrieved (deduped, top-{K}): {r['retrieved_docs']}")
        print(f"  Relevant (answer key):        {sorted(r['relevant_docs'])}")
        print(f"  Hits: {r['hits']}")
        print(
            f"  Precision@{K} = {len(r['hits'])}/{len(r['retrieved_docs'])} = {r['precision_at_k']:.4f}"
        )
        print(
            f"  Recall@{K}    = {len(r['hits'])}/{len(r['relevant_docs'])} = {r['recall_at_k']:.4f}"
        )
        print()

    avg_precision = sum(r["precision_at_k"] for r in all_results) / len(all_results)
    avg_recall = sum(r["recall_at_k"] for r in all_results) / len(all_results)

    print("=" * 60)
    print(
        f"Average Precision@{K} across {len(all_results)} queries: {avg_precision:.4f}"
    )
    print(f"Average Recall@{K} across {len(all_results)} queries:    {avg_recall:.4f}")

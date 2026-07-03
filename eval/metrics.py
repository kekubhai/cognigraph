import re
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import confusion_matrix, classification_report

def classification_report_by_threshold(results: list[dict], threshold: float = 0.5):
    y_true = [1] * len(results)  # all expected to be good
    y_pred = [1 if r["token_f1"] >= threshold else 0 for r in results]
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, target_names=["bad", "good"]))
def semantic_similarity(answer: str, expected: str, embedder) -> float:
    emb1 = embedder.embed_query(answer)
    emb2 = embedder.embed_query(expected)
    sim = cosine_similarity([emb1], [emb2])[0][0]
    return round(float(sim), 4)


def token_f1(answer: str, expected: str) -> float:
    a_tokens = set(re.findall(r'\w+', answer.lower()))
    e_tokens = set(re.findall(r'\w+', expected.lower()))
    if not a_tokens or not e_tokens:
        return 0.0
    intersection = a_tokens & e_tokens
    precision = len(intersection) / len(a_tokens)
    recall = len(intersection) / len(e_tokens)
    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 4)


def topic_coverage(answer: str, expected_topics: list[str]) -> float:
    if not expected_topics:
        return 1.0
    answer_lower = answer.lower()
    matched = sum(1 for t in expected_topics if t.lower() in answer_lower)
    return round(matched / len(expected_topics), 4)


def faithfulness(answer: str, chunks: list[str]) -> float:
    if not chunks or not answer:
        return 0.0

    combined_context = " ".join(chunks).lower()
    sentences = re.split(r'[.!?]+', answer)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]

    if not sentences:
        return 0.0

    supported = 0
    for sentence in sentences:
        words = [w for w in re.findall(r'\w+', sentence.lower()) if len(w) > 3]
        if not words:
            continue
        matches = sum(1 for w in words if w in combined_context)
        if matches / len(words) >= 0.5:
            supported += 1

    return round(supported / len(sentences), 4)


def context_relevance(query: str, chunks: list[str], embedder) -> float:
    if not chunks:
        return 0.0
    query_emb = embedder.embed_query(query)
    chunk_embs = [embedder.embed_query(c) for c in chunks]
    sims = cosine_similarity([query_emb], chunk_embs)[0]
    return round(float(np.mean(sims)), 4)


def evaluate_answer(
    query: str,
    generated_answer: str,
    expected_answer: str,
    expected_topics: list[str],
    retrieved_chunks: list[str],
    embedder,
) -> dict:
    return {
        "query": query,
        "semantic_similarity": semantic_similarity(generated_answer, expected_answer, embedder),
        "token_f1": token_f1(generated_answer, expected_answer),
        "topic_coverage": topic_coverage(generated_answer, expected_topics),
        "faithfulness": faithfulness(generated_answer, retrieved_chunks),
        "context_relevance": context_relevance(query, retrieved_chunks, embedder),
        "num_chunks": len(retrieved_chunks),
        "answer_length": len(generated_answer.split()),

    }

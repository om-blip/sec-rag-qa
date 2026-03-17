# evaluate.py
# Custom RAG evaluator using Groq as the judge LLM.
# Scores each answer on Faithfulness, Relevancy, Correctness,
# Precision@K and Recall@K.
# Run: python app/evaluate.py <collection_id>

import os
import sys
import json
import csv

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from groq import Groq
from dotenv import load_dotenv
from app.retriever import retrieve_chunks
from app.chain import ask

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ── Test cases ────────────────────────────────────────────────
# Ground truth Q&A pairs based on Schneider Electric 2023 report.
# Add or edit these to match whatever PDF you're evaluating.
TEST_CASES = [
    {
        "question": "What was the revenue in 2023?",
        "ground_truth": "Revenue in 2023 was 35,902 million euros"
    },
    {
        "question": "What were the main risk categories?",
        "ground_truth": "The main risks are Economic, Customer, Environmental, and Other risks"
    },
    {
        "question": "What was the revenue growth percentage in 2023?",
        "ground_truth": "Revenue grew 12.7% organically and 5.1% on a reported basis"
    },
    {
        "question": "What was the net debt at end of 2023?",
        "ground_truth": "Net debt was 9,367 million euros at December 31, 2023"
    },
    {
        "question": "What dividend was proposed for 2023?",
        "ground_truth": "A dividend of 3.50 euros per share was proposed"
    },
]


# ── Generation metrics ────────────────────────────────────────

def score_answer(question: str, answer: str, context: str, ground_truth: str) -> dict:
    """
    Ask Groq to score a generated answer on 3 metrics.
    - faithfulness: is the answer grounded in the retrieved context?
    - relevancy:    does the answer actually address the question?
    - correctness:  how close is the answer to the ground truth?
    Returns a dict with float scores 0.0 to 1.0.
    """
    prompt = f"""You are evaluating a RAG system. Score the answer on 3 metrics.
Return ONLY a JSON object, nothing else. No explanation outside the JSON.

Question: {question}
Ground Truth: {ground_truth}
Retrieved Context: {context[:1500]}
Generated Answer: {answer}

Score each metric from 0.0 to 1.0:
- faithfulness: Is the answer grounded in the context? (1.0 = fully grounded, 0.0 = hallucinated)
- relevancy: Does the answer actually address the question? (1.0 = directly answers, 0.0 = off-topic)
- correctness: How close is the answer to the ground truth? (1.0 = matches, 0.0 = wrong)

Return exactly this JSON:
{{"faithfulness": 0.0, "relevancy": 0.0, "correctness": 0.0, "reason": "one line explanation"}}"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=200
    )

    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        print(f"  Warning: could not parse scores — defaulting to 0.5")
        return {
            "faithfulness": 0.5,
            "relevancy": 0.5,
            "correctness": 0.5,
            "reason": "parse error"
        }


# ── Retrieval metrics ─────────────────────────────────────────

def get_relevant_pages(question: str, ground_truth: str) -> list:
    """
    Ask Groq which page numbers SHOULD contain the answer to this question.
    This is our ground truth for retrieval evaluation.
    Returns a list of page number integers.
    """
    prompt = f"""Given this question and its known answer, which page numbers from 
a financial report would most likely contain this information?
Return ONLY a JSON array of integers e.g. [2, 3, 15]. Nothing else.

Question: {question}
Known answer: {ground_truth}

JSON array:"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=50
    )

    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        pages = json.loads(raw)
        return [int(p) for p in pages]
    except Exception:
        return []


def precision_at_k(retrieved_pages: list, relevant_pages: list) -> float:
    """
    Of the K chunks retrieved, what fraction are relevant?
    precision@k = |retrieved ∩ relevant| / k

    Example: retrieved=[2,5,8,23,51], relevant=[2,23]
    hits=2, precision@5 = 2/5 = 0.40
    """
    if not retrieved_pages:
        return 0.0
    hits = len(set(retrieved_pages) & set(relevant_pages))
    return round(hits / len(retrieved_pages), 3)


def recall_at_k(retrieved_pages: list, relevant_pages: list) -> float:
    """
    Of all relevant chunks that exist, what fraction did we retrieve?
    recall@k = |retrieved ∩ relevant| / |relevant|

    Example: retrieved=[2,5,8,23,51], relevant=[2,23,66]
    hits=2, recall@5 = 2/3 = 0.67
    """
    if not relevant_pages:
        return 0.0
    hits = len(set(retrieved_pages) & set(relevant_pages))
    return round(hits / len(relevant_pages), 3)


# ── Main evaluation loop ──────────────────────────────────────

def run_evaluation(collection_id: str):
    print(f"\nRunning RAG evaluation — collection: {collection_id}")
    print(f"Test cases: {len(TEST_CASES)}")
    print("=" * 60)

    all_scores = []

    for i, tc in enumerate(TEST_CASES):
        print(f"\n[{i+1}/{len(TEST_CASES)}] {tc['question']}")

        # Step 1: retrieve chunks for this question
        chunks = retrieve_chunks(tc["question"], collection_id, top_k=5)
        context = " ".join([c["text"] for c in chunks])
        retrieved_pages = [c["page"] for c in chunks]

        # Step 2: get answer from pipeline
        result = ask(tc["question"], collection_id)
        answer = result["answer"]
        print(f"  Answer: {answer[:120]}...")

        # Step 3: score generation quality
        gen_scores = score_answer(
            tc["question"], answer, context, tc["ground_truth"]
        )

        # Step 4: score retrieval quality
        relevant_pages = get_relevant_pages(tc["question"], tc["ground_truth"])
        p_at_k = precision_at_k(retrieved_pages, relevant_pages)
        r_at_k = recall_at_k(retrieved_pages, relevant_pages)

        scores = {
            **gen_scores,
            "precision_at_k": p_at_k,
            "recall_at_k": r_at_k,
            "retrieved_pages": retrieved_pages,
            "relevant_pages": relevant_pages
        }
        all_scores.append(scores)

        # Per-question output
        print(f"  Faithfulness:  {gen_scores['faithfulness']:.2f}  "
              f"Relevancy: {gen_scores['relevancy']:.2f}  "
              f"Correctness: {gen_scores['correctness']:.2f}")
        print(f"  Precision@5:   {p_at_k:.2f}  "
              f"Recall@5:  {r_at_k:.2f}")
        print(f"  Retrieved pages: {retrieved_pages}")
        print(f"  Expected pages:  {relevant_pages}")
        print(f"  Reason: {gen_scores.get('reason', '-')}")

    # ── Aggregate scores ──────────────────────────────────────
    metrics = ["faithfulness", "relevancy", "correctness", "precision_at_k", "recall_at_k"]
    avgs = {m: round(sum(s[m] for s in all_scores) / len(all_scores), 3) for m in metrics}
    overall = round(sum(avgs.values()) / len(avgs), 3)

    print("\n" + "=" * 60)
    print("FINAL EVALUATION RESULTS")
    print("=" * 60)
    print("\n--- Generation Metrics ---")
    print(f"  Faithfulness:   {avgs['faithfulness']:.3f}  (answer grounded in context?)")
    print(f"  Relevancy:      {avgs['relevancy']:.3f}  (answer addresses the question?)")
    print(f"  Correctness:    {avgs['correctness']:.3f}  (answer matches ground truth?)")
    print("\n--- Retrieval Metrics ---")
    print(f"  Precision@5:    {avgs['precision_at_k']:.3f}  (retrieved chunks actually useful?)")
    print(f"  Recall@5:       {avgs['recall_at_k']:.3f}  (found all relevant chunks?)")
    print(f"\n  Overall mean:   {overall:.3f}")
    print("=" * 60)
    print("\nScale: < 0.6 = poor  |  0.6-0.8 = good  |  0.8+ = strong")

    # ── Save results to CSV ───────────────────────────────────
    csv_path = "evaluation_results.csv"
    with open(csv_path, "w", newline="") as f:
        fieldnames = ["question", "ground_truth"] + metrics + ["reason", "retrieved_pages", "relevant_pages"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for tc, sc in zip(TEST_CASES, all_scores):
            writer.writerow({
                "question": tc["question"],
                "ground_truth": tc["ground_truth"],
                "faithfulness": sc["faithfulness"],
                "relevancy": sc["relevancy"],
                "correctness": sc["correctness"],
                "precision_at_k": sc["precision_at_k"],
                "recall_at_k": sc["recall_at_k"],
                "reason": sc.get("reason", ""),
                "retrieved_pages": str(sc["retrieved_pages"]),
                "relevant_pages": str(sc["relevant_pages"])
            })

    print(f"\nFull results saved to {csv_path}")
    print("Add this CSV to your GitHub repo and reference it in your README.")

    return avgs


# ── Entry point ───────────────────────────────────────────────
if __name__ == "__main__":
    collection_id = sys.argv[1] if len(sys.argv) > 1 else "a6caf8154556"
    run_evaluation(collection_id)
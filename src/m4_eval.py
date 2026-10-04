from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    import math
    from config import OPENAI_API_KEY, OPENAI_BASE_URL, LLM_MODEL, EMBEDDING_MODEL

    metric_names = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
    fallback = {**dict.fromkeys(metric_names, 0.0), "per_question": []}
    if len({len(questions), len(answers), len(contexts), len(ground_truths)}) != 1:
        raise ValueError("Evaluation inputs must have equal lengths")
    if not questions or not OPENAI_API_KEY:
        return {**fallback, "status": "skipped",
                "error": "Empty dataset or OPENAI_API_KEY is not configured"}
    try:
        from datasets import Dataset
        from langchain_openai import ChatOpenAI
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from ragas import evaluate
        from ragas.metrics import faithfulness, AnswerRelevancy, context_precision, context_recall
        from ragas.run_config import RunConfig
        import asyncio

        class ThrottledChatOpenAI(ChatOpenAI):
            async def _agenerate(self, *args, **kwargs):
                await asyncio.sleep(2)  # Groq: 30 RPM → 2s delay
                return await super()._agenerate(*args, **kwargs)
                
            def _generate(self, *args, **kwargs):
                import time
                time.sleep(2)
                return super()._generate(*args, **kwargs)

        dataset = Dataset.from_dict({
            "question": questions, "answer": answers,
            "contexts": contexts, "ground_truth": ground_truths,
        })
        result = evaluate(
            # One generated question per answer; use the same setting for both pipelines.
            dataset, metrics=[faithfulness, AnswerRelevancy(strictness=1), context_precision, context_recall],
            llm=ThrottledChatOpenAI(
                model=LLM_MODEL, api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL,
                temperature=0, max_tokens=4096, timeout=300, max_retries=10,
            ),
            embeddings=HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True, "batch_size": 8},
            ),
            run_config=RunConfig(timeout=300, max_retries=5, max_workers=1),
            raise_exceptions=False,
        )
        df = result.to_pandas()
        if len(df) != len(questions):
            raise ValueError("RAGAS returned an unexpected number of rows")
        per_question = []
        for i, (_, row) in enumerate(df.iterrows()):
            # Thay NaN bằng 0.0 thay vì raise lỗi (xảy ra khi câu trả lời là "Không tìm thấy.")
            raw_scores = {name: float(row[name]) for name in metric_names}
            scores = {name: (v if math.isfinite(v) else 0.0) for name, v in raw_scores.items()}
            if any(not math.isfinite(v) for v in raw_scores.values()):
                print(f"  ⚠️  Câu {i+1}: Một số metrics trả về NaN → dùng 0.0 (câu trả lời không có statements).")
            per_question.append(EvalResult(
                question=questions[i], answer=answers[i], contexts=list(contexts[i]),
                ground_truth=ground_truths[i], **scores,
            ))
        aggregate = {name: sum(getattr(row, name) for row in per_question) / len(per_question)
                     for name in metric_names}
        return {**aggregate, "per_question": per_question, "status": "success"}
    except Exception as exc:
        error = str(exc).replace(OPENAI_API_KEY, "[redacted]")
        print(f"  ⚠️ RAGAS evaluation failed: {type(exc).__name__}: {error}")
        return {**fallback, "status": "failed", "error": f"{type(exc).__name__}: {error}"}


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    import math

    if bottom_n <= 0:
        return []
    diagnostic_tree = {
        "faithfulness": ("LLM hallucinating", "Tighten prompt, lower temperature"),
        "context_recall": ("Missing relevant chunks", "Improve chunking or add BM25"),
        "context_precision": ("Too many irrelevant chunks", "Add reranking or metadata filter"),
        "answer_relevancy": ("Answer doesn't match question", "Improve prompt template"),
    }
    failures = []
    for result in eval_results:
        scores = {name: float(getattr(result, name)) for name in diagnostic_tree}
        if not all(math.isfinite(score) for score in scores.values()):
            continue
        worst_metric = min(scores, key=scores.get)
        diagnosis, suggested_fix = diagnostic_tree[worst_metric]
        failures.append({
            "question": result.question, "worst_metric": worst_metric,
            "score": scores[worst_metric], "average_score": sum(scores.values()) / len(scores),
            "diagnosis": diagnosis, "suggested_fix": suggested_fix,
        })
    return sorted(failures, key=lambda row: row["average_score"])[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "reports/ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")

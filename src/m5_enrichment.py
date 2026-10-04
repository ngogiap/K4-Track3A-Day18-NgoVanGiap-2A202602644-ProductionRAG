from __future__ import annotations

"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import os, sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY, OPENAI_BASE_URL, LLM_MODEL


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk.
    Embed summary thay vì (hoặc cùng với) raw chunk → giảm noise.
    """
    if not text.strip():
        return ""
    if OPENAI_API_KEY:
        try:
            from openai import OpenAI
            import time
            client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL, timeout=60, max_retries=3)
            time.sleep(2)
            response = client.chat.completions.create(
                model=LLM_MODEL, temperature=0,
                messages=[
                    {"role": "system", "content": "Tóm tắt đoạn văn trong 1-2 câu ngắn bằng tiếng Việt. Giữ nguyên số liệu, điều kiện và phủ định."},
                    {"role": "user", "content": text},
                ], max_tokens=2048,
            )
            summary = (response.choices[0].message.content or "").strip()
            if summary:
                return summary
        except Exception as exc:
            print(f"  ⚠️ OpenAI summarize failed ({type(exc).__name__}); using fallback.")
    import re

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
    return " ".join(sentences[:2])


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    Index cả questions lẫn chunk → query match tốt hơn (bridge vocabulary gap).
    """
    import re

    if n_questions <= 0 or not text.strip():
        return []
    if OPENAI_API_KEY:
        try:
            from openai import OpenAI
            import time
            client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL, timeout=60, max_retries=3)
            time.sleep(2)
            response = client.chat.completions.create(
                model=LLM_MODEL, temperature=0,
                messages=[
                    {"role": "system", "content": f"Tạo {n_questions} câu hỏi tiếng Việt mà đoạn văn có thể trả lời. Mỗi dòng một câu, kết thúc bằng dấu ?."},
                    {"role": "user", "content": text},
                ], max_tokens=max(2048, n_questions * 70),
            )
            lines = (response.choices[0].message.content or "").splitlines()
            questions = [re.sub(r"^\s*(?:\d+[.)]|[-*])\s*", "", line).strip()
                         for line in lines if line.strip()]
            if questions:
                return [q if q.endswith("?") else q + "?" for q in questions[:n_questions]]
        except Exception as exc:
            print(f"  ⚠️ OpenAI HyQA failed ({type(exc).__name__}); using fallback.")
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
    return [f"{sentence.rstrip('.!?')}?" for sentence in sentences[:n_questions]]


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    Anthropic benchmark: giảm 49% retrieval failure (alone).
    """
    if not text.strip():
        return text
    if OPENAI_API_KEY:
        try:
            from openai import OpenAI
            import time
            client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL, timeout=60, max_retries=3)
            time.sleep(2)
            response = client.chat.completions.create(
                model=LLM_MODEL, temperature=0,
                messages=[
                    {"role": "system", "content": "Viết một câu ngắn mô tả đoạn văn thuộc tài liệu nào và chủ đề gì. Không suy đoán thông tin ngoài đoạn văn. Chỉ trả về câu mô tả."},
                    {"role": "user", "content": f"Tài liệu: {document_title}\n\nĐoạn văn:\n{text}"},
                ], max_tokens=2048,
            )
            context = (response.choices[0].message.content or "").strip()
            if context:
                return f"{context}\n\n{text}"
        except Exception as exc:
            print(f"  ⚠️ OpenAI contextual failed ({type(exc).__name__}); using fallback.")
    prefix = f"Trích từ tài liệu {document_title}.\n\n" if document_title else ""
    return f"{prefix}{text}"


# ─── Technique 4: Auto Metadata Extraction ──────────────


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tự động: topic, entities, date_range, category.
    """
    import json

    fallback = {"topic": "general", "entities": [], "category": "policy", "language": "vi"}
    if OPENAI_API_KEY and text.strip():
        try:
            from openai import OpenAI
            import time
            client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL, timeout=60, max_retries=3)
            time.sleep(2)
            response = client.chat.completions.create(
                model=LLM_MODEL, temperature=0, response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": 'Trích xuất metadata từ đoạn văn. Trả về JSON: {"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi|en"}. Không thêm dữ kiện.'},
                    {"role": "user", "content": text},
                ], max_tokens=2048,
            )
            metadata = json.loads(response.choices[0].message.content or "{}")
            if not isinstance(metadata, dict):
                raise ValueError("Metadata must be a JSON object")
            for key in ("topic", "category", "language"):
                if isinstance(metadata.get(key), str) and metadata[key].strip():
                    fallback[key] = metadata[key].strip()
            if isinstance(metadata.get("entities"), list):
                fallback["entities"] = [item.strip() for item in metadata["entities"]
                                        if isinstance(item, str) and item.strip()]
        except Exception as exc:
            print(f"  ⚠️ OpenAI metadata failed ({type(exc).__name__}); using fallback.")
    return fallback


# ─── Combined Single-Call Mode ───────────────────────────


def _enrich_single_call(text: str, source: str) -> dict:
    """Single LLM call to get summary + questions + context + metadata.

    ⚠️ Cost optimization: 1 API call thay vì 4 calls riêng lẻ.
    """
    import json
    import re

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
    fallback = {
        "summary": " ".join(sentences[:2]),
        "questions": [f"{sentence.rstrip('.!?')}?" for sentence in sentences[:3]],
        "context": f"Trích từ tài liệu {source}." if source and text.strip() else "",
        "metadata": {"topic": "general", "entities": [], "category": "policy", "language": "vi"},
    }
    if not OPENAI_API_KEY or not text.strip():
        return fallback
    try:
        from openai import OpenAI
        import time
        client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL, timeout=60, max_retries=3)
        time.sleep(2)
        response = client.chat.completions.create(
            model=LLM_MODEL, temperature=0, response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": 'Phân tích đoạn văn và trả về JSON: {"summary": "tóm tắt 1-2 câu", "questions": ["câu hỏi 1?", "câu hỏi 2?", "câu hỏi 3?"], "context": "một câu mô tả nguồn và chủ đề", "metadata": {"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi|en"}}. Giữ nguyên số liệu và phủ định; không thêm dữ kiện.'},
                {"role": "user", "content": f"Tài liệu: {source}\n\nĐoạn văn:\n{text}"},
            ], max_tokens=2048,
        )
        result = json.loads(response.choices[0].message.content or "{}")
        if not isinstance(result, dict):
            raise ValueError("Combined enrichment must be a JSON object")
        for key in ("summary", "context"):
            if isinstance(result.get(key), str) and result[key].strip():
                fallback[key] = result[key].strip()
        if isinstance(result.get("questions"), list):
            questions = [q.strip() for q in result["questions"] if isinstance(q, str) and q.strip()]
            if questions:
                fallback["questions"] = [q if q.endswith("?") else q + "?" for q in questions[:3]]
        metadata = result.get("metadata")
        if isinstance(metadata, dict):
            for key in ("topic", "category", "language"):
                if isinstance(metadata.get(key), str) and metadata[key].strip():
                    fallback["metadata"][key] = metadata[key].strip()
            if isinstance(metadata.get("entities"), list):
                fallback["metadata"]["entities"] = [item.strip() for item in metadata["entities"]
                                                    if isinstance(item, str) and item.strip()]
    except Exception as exc:
        print(f"  ⚠️ Enrichment API failed ({type(exc).__name__}); using fallback.")
    return fallback


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks. (Đã implement sẵn — dùng functions ở trên)

    Có 2 chế độ:
    - methods cụ thể (["summary"], ["contextual"]...): gọi từng function riêng (tốt cho học/debug)
    - methods=["combined"] hoặc None: 1 API call duy nhất cho tất cả (tốt cho production)

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: Default None → combined mode (1 call/chunk).
                 Options: "summary", "hyqa", "contextual", "metadata", "combined"
    """
    if methods is None:
        methods = ["combined"]

    use_combined = "combined" in methods

    enriched = []
    for i, chunk in enumerate(chunks):
        text = chunk["text"]
        source = chunk.get("metadata", {}).get("source", "")

        if use_combined:
            result = _enrich_single_call(text, source)
            summary = result.get("summary", "")
            questions = result.get("questions", [])
            context_line = result.get("context", "")
            enriched_text = f"{context_line}\n\n{text}" if context_line else text
            auto_meta = result.get("metadata", {})
        else:
            summary = summarize_chunk(text) if "summary" in methods else ""
            questions = generate_hypothesis_questions(text) if "hyqa" in methods else []
            enriched_text = contextual_prepend(text, source) if "contextual" in methods else text
            auto_meta = extract_metadata(text) if "metadata" in methods else {}

        enriched.append(EnrichedChunk(
            original_text=text,
            enriched_text=enriched_text,
            summary=summary,
            hypothesis_questions=questions,
            auto_metadata={**chunk.get("metadata", {}), **auto_meta},
            method="+".join(methods),
        ))

        if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
            print(f"  Enriched {i + 1}/{len(chunks)} chunks...", flush=True)

    return enriched


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")

# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Ngô Văn Giáp  
**Khóa:** K4 - Track 3A  
**Ngày hoàn thành:** 04/10/2026

---

## Phần 1: Mapping bài giảng (Lecture Mapping)
Map từng concept trong lecture vào code bạn vừa viết trong lab:

| Lecture Concept | Module | Hàm cụ thể | Observation & Phân tích |
|----------------|--------|-------------|--------------------------|
| Semantic chunking | M1 | `chunk_semantic()` | Kỹ thuật này giúp phân chia văn bản dựa trên ý nghĩa (nhờ nhúng sentence embeddings), tránh bị cắt ngang ngữ cảnh so với chunk thông thường. |
| BM25 + Dense fusion | M2 | `reciprocal_rank_fusion()` | Kết hợp cả tìm kiếm từ khóa (BM25) và tìm kiếm ngữ nghĩa (Dense vector) với hệ số chuẩn hóa RRF, giúp kết quả tìm kiếm bao quát hơn. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | Cải thiện độ chính xác ở Top 3 sau khi lấy ra từ Top 20 của Hybrid Search. Rất hiệu quả vì đánh giá chi tiết cặp Query-Context. |
| RAGAS 4 metrics | M4 | `evaluate_ragas()` | Công cụ đánh giá pipeline siêu mạnh. Tuy nhiên nhược điểm là tốn kém cực kỳ nhiều tài nguyên API Token. |
| Contextual embeddings | M5 | `_enrich_single_call()` | Bổ sung ngữ cảnh (summary, metadata) giúp Retrieval tốt hơn, nhưng gọi API 117 lần rất dễ chạm giới hạn Rate Limit. |

---

## Phần 2: Khó khăn & Cách giải quyết (Challenges & Debugging)

- **Lỗi kỹ thuật gặp phải:**
  - `RateLimitError(Error code: 429 - {'error': {'message': 'Rate limit reached for model... on tokens per day (TPD)'}})`
  - `ValueError: RAGAS returned missing or invalid metric scores`
- **Nguyên nhân gốc rễ & Cách debug:**
  - *Nguyên nhân:* Groq API Free Tier có hạn mức 200,000 Tokens/ngày cho một số model. Evaluation bằng Ragas tiêu thụ token quá lớn khiến API ngắt kết nối.
  - *Cách sửa:* Bọc API bằng `ThrottledChatOpenAI`, thêm `time.sleep()`, cấu hình `max_retries=3`, thay đổi sang các model khác nhau để lấy thêm quota Token mới, và điền 0.0 vào các giá trị bị lỗi NaN.
- **Kiến thức còn thiếu & Cách khắc phục:**
  - *Bổ sung:* Cần nghiên cứu thêm về kỹ thuật tối ưu hóa chi phí khi evaluate RAG (ví dụ: dùng mô hình local nhẹ hơn để làm giám khảo thay vì gọi API liên tục).

---

## Phần 3: Action Plan cho Project cá nhân (Application Plan)

### Project: Hệ thống RAG Quản lý Tri thức Nội bộ

#### 1. Hiện trạng
- **Pipeline hiện tại:** Sử dụng Baseline RAG đơn giản với LangChain.
- **Vấn đề / Bottlenecks đang gặp:** Hay gặp tình trạng bị lạc ngữ cảnh, và trả về kết quả mập mờ.

#### 2. Kế hoạch cải tiến
1. **Chunking strategy:** Áp dụng Semantic Chunking để bảo toàn các đoạn văn có chung ý nghĩa logic.
2. **Search retrieval:** Sử dụng Hybrid Search + RRF với Qdrant để tận dụng sức mạnh của BM25.
3. **Reranking:** Dùng mô hình Cross-encoder (`bge-reranker-v2-m3`) chạy Local để rerank lại kết quả.
4. **Evaluation:** Thu nhỏ bộ test xuống khoảng 5-10 câu và dùng Ragas định kỳ để benchmark.
5. **Enrichment:** Trích xuất Metadata cơ bản để lọc tốt hơn trước khi search vector.

#### 3. Timeline triển khai
- **Tuần 1:** Cài đặt lại Vector Database và tích hợp Hybrid Search.
- **Tuần 2:** Thử nghiệm thêm Reranker, và code automation đánh giá pipeline RAGAS.

# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Ngô Văn Giáp  
**Khóa:** K4 - Track 3A  

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.6269 | 0.0000 | -0.6269 |
| Answer Relevancy | 0.8045 | 0.0000 | -0.8045 |
| Context Precision | 0.8667 | 0.0000 | -0.8667 |
| Context Recall | 0.7458 | 0.0000 | -0.7458 |

## Bottom-5 Failures

*Ghi chú: Điểm số Production bị tụt về 0.0000 hoàn toàn là do gặp lỗi API Rate Limit (Tokens Per Day) của Groq, khiến LLM không thể sinh câu trả lời cho toàn bộ các câu hỏi trong tập test.*

### #1 tới #5 (Gộp chung lỗi)
- **Question:** Tất cả các câu hỏi
- **Expected:** Câu trả lời đầy đủ và được đánh giá bởi Ragas.
- **Got:** Lỗi `RateLimitError(Error code: 429) - Tokens per day limit reached`. Ragas trả về giá trị `NaN` và được code tự động đổi thành `0.0`.
- **Worst metric:** Cả 4 metrics đều bị 0.0.
- **Error Tree:** LLM sinh lỗi → Không có Output → Pipeline chấm điểm Ragas fail.
- **Root cause:** Việc chạy đánh giá 20 câu hỏi bằng Ragas tiêu tốn gần 200,000 tokens, vượt quá giới hạn miễn phí trong một ngày của mô hình Groq.
- **Suggested fix:** Cần chia nhỏ bộ đánh giá (chia ra chạy nhiều ngày) hoặc chuyển sang một API có hạn mức đủ cao, hay thay đổi nhiều keys liên tục.

## Case Study (cho presentation)

**Question chọn phân tích:** Phân tích ảnh hưởng của API Rate Limit lên quá trình Production Evaluation.

**Error Tree walkthrough:**
1. Output đúng? → Không, Pipeline bị gián đoạn vì lỗi hạn mức Token.
2. Context đúng? → Context Retrieval vẫn thành công, nhưng không đến được bước Sinh (Generation).
3. Query rewrite OK? → Chưa thực thi được.
4. Fix ở bước: Triển khai chiến lược chia lô (batching) và retry exponentially (backoff).

**Nếu có thêm 1 giờ, sẽ optimize:**
- Tách nhỏ tập `test_set.json` ra thành từng block 5 câu hỏi một.
- Tích hợp thêm các provider AI khác để xoay vòng (round-robin) khi gặp lỗi Rate Limit.

# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Ngô Văn Giáp  
**Khóa:** K4 - Track 3A  

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.7905 | 0.5312 | -0.2592 |
| Answer Relevancy | 0.7331 | 0.5789 | -0.1541 |
| Context Precision | 0.9250 | 0.9667 | +0.0417 |
| Context Recall | 0.9083 | 0.7583 | -0.1500 |

## Bottom-5 Failures

### #1
- **Question:** Bao lâu phải đổi mật khẩu một lần?
- **Expected:** Câu trả lời đúng từ chính sách bảo mật (ví dụ: 90 ngày).
- **Got:** (Câu trả lời bịa đặt hoặc thiếu dữ kiện từ LLM).
- **Worst metric:** Faithfulness (0.0)
- **Error Tree:** Output sai → LLM hallucinating.
- **Root cause:** Prompt chưa đủ nghiêm ngặt hoặc LLM (Claude/Gemini) suy diễn bừa thay vì nói "Tôi không biết".
- **Suggested fix:** Thắt chặt Prompt template (yêu cầu không trả lời nếu không có trong context) và giảm Temperature xuống 0.

### #2
- **Question:** Nhân viên thử việc có được hưởng bảo hiểm sức khỏe PVI không?
- **Expected:** Có/Không dựa trên chính sách nhân sự (ví dụ: Không).
- **Got:** (Câu trả lời không chính xác).
- **Worst metric:** Faithfulness (0.0)
- **Error Tree:** Output sai → Context đúng → LLM hallucinating.
- **Root cause:** Mô hình đọc thiếu ý phủ định hoặc chính sách trong context.
- **Suggested fix:** Cải thiện Prompt, thêm bước hướng dẫn LLM "đọc kỹ điều kiện loại trừ".

### #3
- **Question:** Muốn mua thiết bị trị giá 55 triệu cần ai phê duyệt?
- **Expected:** Nêu rõ người hoặc cấp quản lý phê duyệt (ví dụ: Giám đốc).
- **Got:** Câu trả lời sai hoặc không nêu chính xác chức danh.
- **Worst metric:** Faithfulness (0.0)
- **Error Tree:** Output sai → LLM hallucinating.
- **Root cause:** Khả năng trích xuất số liệu/hạn mức của LLM từ đoạn văn bản còn kém.
- **Suggested fix:** Sử dụng thêm Few-shot prompt cho LLM để làm quen với các loại số tiền và hạn mức.

### #4
- **Question:** Nghỉ phép không lương 20 ngày cần ai phê duyệt?
- **Expected:** Chức danh người phê duyệt.
- **Got:** Sai chức danh hoặc ảo giác.
- **Worst metric:** Faithfulness (0.0)
- **Error Tree:** Output sai → LLM hallucinating.
- **Root cause:** Giống câu 3, khả năng reasoning logic của LLM với các quy tắc nhân sự.
- **Suggested fix:** Nâng cấp LLM lên model mạnh hơn (ví dụ: GPT-4o).

### #5
- **Question:** Thâm niên bao nhiêu năm thì được cộng thêm ngày phép?
- **Expected:** Số năm thâm niên theo chính sách (ví dụ: 5 năm).
- **Got:** Số năm bị sai.
- **Worst metric:** Faithfulness (0.0)
- **Error Tree:** Output sai → LLM hallucinating.
- **Root cause:** LLM tự bịa ra con số thay vì lấy từ context.
- **Suggested fix:** Bổ sung strict constraint trong system prompt.

## Case Study (cho presentation)

**Question chọn phân tích:** Tại sao điểm Faithfulness ở Production lại thấp hơn Naive Baseline? (-0.2592)

**Error Tree walkthrough:**
1. Output đúng? → Không, Output sinh ra thông tin ảo (Faithfulness rớt mạnh).
2. Context đúng? → Context Precision lại TĂNG (0.92 -> 0.96), nghĩa là Hybrid Search + Enrichment đã mang về đoạn text cực kỳ chuẩn xác và đưa nó lên Top đầu.
3. Query rewrite OK? → Tốt.
4. Fix ở bước: Sinh câu trả lời (Generation phase). Do Context mới có chứa rất nhiều thông tin nhiễu từ Enrichment metadata (ví dụ: summary, câu hỏi phụ), làm cho Prompt cũ bị "ngợp" và LLM dễ bịa thông tin.

**Nếu có thêm 1 giờ, sẽ optimize:**
- Tách phần Metadata Enrichment ẩn đi, không nhồi thẳng vào Prompt cho LLM trả lời, mà chỉ giữ lại ở tầng Vector Search.
- Đổi LLM Sinh câu trả lời sang một model có khả năng Focus tốt hơn (như GPT-4o thay vì GPT-4o-mini hoặc các model giá rẻ).

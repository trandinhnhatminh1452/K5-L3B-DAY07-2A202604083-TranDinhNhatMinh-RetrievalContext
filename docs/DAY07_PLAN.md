# Kế hoạch và Thiết kế Day 07 — Agentic RAG

Tài liệu thiết kế thống nhất cho hệ thống Agentic RAG của sinh viên Trần Đình Nhật Minh (MSSV: 2A202604083).

## 1. Mục tiêu và Kiến trúc

- **Corpus**: Kế thừa trọn vẹn từ Day 06 gồm 283 paragraph chunks (Strategy C, tối đa 1.000 ký tự/chunk, không overlap) từ tác phẩm *The Principles of Scientific Management* (Project Gutenberg #6435).
- **Vector Store & Database**: PostgreSQL với extension `pgvector`, bảng `rag_chunks`, kết nối ở chế độ `read-only`.
- **Embedding Model**: `intfloat/multilingual-e5-small` (revision `614241f622f53c4eeff9890bdc4f31cfecc418b3`), 384 chiều, vector chuẩn hóa. Prefix: `query: ` cho câu hỏi và `passage: ` cho tài liệu.
- **Agent Model**: `llama3.2:3b` chạy qua Ollama local HTTP API (`/api/chat`), định dạng Structured Outputs JSON Schema.

## 2. Các phương pháp Retrieval

1. **Dense Retrieval (`dense`)**:
   - Sử dụng model `intfloat/multilingual-e5-small` để embed câu hỏi với tiền tố `query: `.
   - Tính cosine distance qua toán tử pgvector `<=>` trên cơ sở dữ liệu PostgreSQL.
2. **Lexical Retrieval (`tfidf`)**:
   - Xây dựng chỉ mục TF-IDF nội bộ (chuẩn hóa Unicode NFC, casefold, tokenizer `\w+`, BM25/TF-IDF IDF weighting).
   - Truy vấn xếp hạng cosine similarity trên từ khóa chính xác.
3. **Hybrid Retrieval (`hybrid`)**:
   - Kết hợp hai bảng xếp hạng Dense và TF-IDF bằng thuật toán Reciprocal Rank Fusion (RRF, $k=60$):
     $$\text{RRF Score}(d) = \sum_{m \in \{\text{dense}, \text{tfidf}\}} \frac{1}{60 + \text{rank}_m(d)}$$
   - Lấy tối thiểu 20 ứng viên từ mỗi nhánh trước khi hợp nhất.

## 3. Các cấu hình thử nghiệm trên Dev

| ID | Phương pháp | Context Budget | Vai trò & Giả thuyết |
|---|---|---|---|
| **B0** | Dense | 800 ký tự | **Baseline**: Đo lường hiệu năng khởi đầu với ngân sách context hẹp. |
| **E1** | Dense | 4.000 ký tự | **Cải thiện 1**: Tăng context budget lên 4.000 ký tự để chứa vừa các chunk dài tới 1.000 ký tự của Strategy C, tránh việc cắt bỏ chunk bằng chứng. |
| **M1** | TF-IDF | 4.000 ký tự | **So sánh phương pháp**: Đánh giá hiệu quả của truy xuất từ khóa truyền thống so với dense retrieval. |
| **E2** | Hybrid RRF | 4.000 ký tự | **Cải thiện 2**: Kết hợp điểm mạnh ngữ nghĩa của Dense và khả năng khớp từ khóa chính xác của TF-IDF nhằm tăng độ bao phủ bằng chứng. |

## 4. Giao thức Agent & An toàn

- **Số lượt tìm kiếm tối đa**: 4 lượt tìm (`max_searches = 4`), mỗi lượt lấy `top_k = 3`.
- **Cơ chế chống lặp**: Vòng lặp agent phát hiện nếu model đưa ra truy vấn trùng hệt truy vấn đã tìm (`duplicate_query_stop`), agent lập tức dừng và chuyển sang trạng thái từ chối (`abstain`).
- **Hành động từ chối chủ động (`deliberate_abstention`)**: Model được phép gọi `abstain` kèm lý do rõ ràng khi nhận thấy context không đủ bằng chứng. Bắt buộc agent phải tìm kiếm ít nhất một lần trước khi từ chối.
- **Citation Rendering & Validation**:
  - Model phải liệt kê danh sách chunk ID trong trường `citations` và trích dẫn `[chunk-id]` trong văn bản.
  - Renderer tự động bổ sung tham chiếu inline ở cuối câu trả lời nếu model đã chọn ID hợp lệ từ context cuối nhưng chưa chèn inline.
  - Mọi citation đều được kiểm tra tính hợp lệ: ID phải tồn tại trong context cuối, không có citation bịa đặt.

## 5. Tiêu chí chọn cấu hình (`select`)

Thứ tự ưu tiên đánh giá trên tập dev:
1. Tỷ lệ câu được xử lý đúng (`correct_rate` trên toàn bộ 17 câu dev: câu có đáp án phải trả lời đúng đủ dẫn nguồn, câu thiếu bằng chứng phải từ chối đúng; lỗi runtime tính là 0).
2. Tỷ lệ context chứa đủ bằng chứng (`context_evidence_rate` trên 15 câu có đáp án).
3. Evidence Hit@3 của câu hỏi gốc (`hit3_rate`).
4. Độ trễ trung vị của agent (`median_agent_ms` thấp hơn).
5. Thứ tự ưu tiên giải hòa: B0 $\to$ E1 $\to$ M1 $\to$ E2.

# Báo cáo Day 7 — Agentic RAG

**Họ và tên:** Trần Đình Nhật Minh  
**Mã số sinh viên (MSSV):** 2A202604083  
**Lớp:** K5-L3B  
**Repository:** `K5-L3B-DAY07-2A202604083-TranDinhNhatMinh-RetrievalContext`  

---

## 1. Dữ liệu và benchmark

- **Nguồn, quyền sử dụng, số trang và số chunk**:
  - Dữ liệu kế thừa từ bài tập Day 06: Tác phẩm *The Principles of Scientific Management* của Frederick Winslow Taylor, Project Gutenberg #6435 (Public Domain).
  - Gồm 46 logical HTML units (được gán nhãn `page` theo đơn vị cấu trúc HTML).
  - Sử dụng cấu hình chunking **C** (Paragraph-based): Tổng cộng **283 chunks**, kích thước tối đa 1.000 ký tự, không overlap.
- **Vector store, embedding model, phiên bản và số chiều**:
  - Cơ sở dữ liệu: PostgreSQL với extension `pgvector`, bảng `rag_chunks`, strategy C. Kết nối transaction ở chế độ `read-only` (`default_transaction_read_only=on`).
  - Embedding model: `intfloat/multilingual-e5-small`, revision `614241f622f53c4eeff9890bdc4f31cfecc418b3`, kích thước **384 chiều**, vector chuẩn hóa độ dài đơn vị (unit norm). Tiền tố: `query: ` cho câu hỏi, `passage: ` cho tài liệu.
- **Agent model, cách chạy và phân chia dev/test**:
  - Model: `llama3.2:3b` (digest `a80c4f17acd5...`), chạy cục bộ qua Ollama HTTP API (`POST /api/chat`), Structured Outputs JSON Schema. (Ở bước thử nghiệm ban đầu, `qwen2.5:7b` gây tải RAM lớn trên máy, do đó học viên thống nhất chuyển sang `llama3.2:3b` và giữ nguyên model này trên toàn bộ các cấu hình so sánh).
  - Nhiệt độ $0$, seed $42$, context window $4.096$ token, tối đa $512$ token output.
  - Phân chia: **Dev** gồm 17 câu (15 câu có đáp án từ Day 06 + 2 câu unanswerable `ua-dev-01`, `ua-dev-02`); **Test** gồm 7 câu (5 câu có đáp án từ Day 06 + 2 câu unanswerable `ua-test-01`, `ua-test-02`). Tổng cộng **24 câu**. Toàn bộ đầu vào được đóng băng tại `benchmark/freeze.json` (SHA-256).
- **Cách xác định câu có/thiếu evidence và tiêu chí chấm**:
  - Câu có đáp án: Gold evidence xác định từ văn bản gốc. Riêng câu `q13` có bằng chứng trải dài qua hai paragraph chunks được bảo tồn nguyên vẹn bằng nhóm điều kiện logic `evidence_groups` dạng AND (yêu cầu context phải chứa đầy đủ mọi phần).
  - Câu thiếu bằng chứng: Kiểm tra toàn bộ 283 chunks của kho tài liệu để chứng minh sự vắng mặt của thông tin (ví dụ: số liệu năm 2025, áp dụng tại Việt Nam năm 2024, hoặc mức thưởng tiền mặt cố định).
  - Tiêu chí chấm:
    - `context_grade`: 0 = không có evidence; 1 = có nhưng thiếu một phần; 2 = đủ bằng chứng và nguồn.
    - `grounded_answer`: 0 = sai, bịa đặt hoặc từ chối câu có đáp án; 1 = đúng một phần; 2 = đúng và đủ dựa trên context.
    - `abstention_correct`: đúng nếu từ chối câu thiếu bằng chứng sau khi đã tra cứu.

---

## 2. Retrieval và thử nghiệm

### Bảng tổng hợp Retrieval trên tập Dev và Test

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|:---:|:---:|:---:|:---:|
| **B0** | Dense (e5-small), top-3, no filter, budget 800 ký tự | Baseline ban đầu | dev | 8/15 (53.3%) | 12/15 (80.0%) | 3/15 (20.0%) | 157.2 ms / 0 USD |
| **E1** | Dense (e5-small), top-3, no filter, budget 4.000 ký tự | Cải thiện ngân sách context để chứa đủ các chunk dài của Strategy C | dev | 8/15 (53.3%) | 12/15 (80.0%) | 8/15 (53.3%) | 253.4 ms / 0 USD |
| **M1** | TF-IDF (lexical), top-3, no filter, budget 4.000 ký tự | So sánh phương pháp từ khóa chính xác | dev | 8/15 (53.3%) | 10/15 (66.7%) | 8/15 (53.3%) | 3.7 ms / 0 USD |
| **E2** | Hybrid RRF ($k=60$), top-3, no filter, budget 4.000 ký tự | Kết hợp ngữ nghĩa Dense và khớp từ khóa TF-IDF | dev | 9/15 (60.0%) | 10/15 (66.7%) | 8/15 (53.3%) | 121.7 ms / 0 USD |
| **M1 (Chốt)** | TF-IDF (lexical), top-3, no filter, budget 4.000 ký tự | Cấu hình được chọn theo tiêu chí dev protocol | test | **5/5 (100.0%)** | **5/5 (100.0%)** | **4/5 (80.0%)** | **3.2 ms / 0 USD** |
| **B0** | Dense (e5-small), top-3, no filter, budget 800 ký tự | Baseline đối chứng độc lập trên test | test | 2/5 (40.0%) | 4/5 (80.0%) | 0/5 (0.0%) | 93.6 ms / 0 USD |

### Phân tích lỗi Dev, Nguyên nhân và Lý do chọn cấu hình

1. **Lỗi 1: Nghẽn ngân sách context ở Baseline B0 (Context Bottleneck)**:
   - *Hiện tượng*: Tại B0, dù Hit@3 đạt 80.0% (12/15 câu trên dev), nhưng `Context Evidence Rate` chỉ đạt 20.0% (3/15 câu trên dev) và 0.0% (0/5 câu trên test). Tỷ lệ từ chối nhầm (`false_abstention`) lên tới 80.0% (12/15 câu dev, 4/5 câu test bị agent từ chối trả lời).
   - *Nguyên nhân*: Các chunk của Strategy C có độ dài lên tới 1.000 ký tự. Ngân sách 800 ký tự chỉ đủ chứa tối đa 1 chunk duy nhất, dẫn đến việc các chunk chứa bằng chứng bị cắt bỏ khỏi prompt của agent.
   - *Giải pháp*: Cấu hình E1, M1, E2 tăng context budget lên 4.000 ký tự. Kết quả `Context Evidence Rate` nhảy vọt từ 20.0% lên 53.3% trên dev và đạt **80.0%** trên test; tỷ lệ câu trả lời đúng tăng rõ rệt.
2. **Lỗi 2: Lặp lại truy vấn giống hệt và ảo giác ở câu thiếu bằng chứng**:
   - *Hiện tượng*: Trong lượt chạy pilot ban đầu, agent có xu hướng phát sinh query giống hệt câu hỏi ban đầu khi context chưa đủ, dẫn đến vòng lặp vô ích. Ở câu `ua-dev-02`, B0 bị ảo giác sinh ra con số cố định "$500" không hề có trong tài liệu.
   - *Giải pháp*: Thêm guard `duplicate_query_stop` trong `starter/agentic.py` (nếu model lặp lại query giống hệt thì buộc dừng và chuyển sang abstain). Cải thiện prompt yêu cầu không suy diễn con số cố định. Kết quả ở E1, M1, E2, cả 2 câu `ua-dev-01` và `ua-dev-02` đều được từ chối chính xác (Abstention = 100%), và trên test cả 2 câu `ua-test-01`, `ua-test-02` đều được từ chối đúng 100%.
3. **Lý do chọn cấu hình M1**:
   - Theo tiêu chí định sẵn trong `benchmark/protocol.json`: M1 đạt tỷ lệ đúng tổng thể cao nhất (4/17 câu dev đúng hoàn toàn và có citation hợp lệ), điểm Grounded Answer trung bình cao nhất (1.07/2 so với 0.13 của B0 và 0.73 của E1), tỷ lệ từ chối đúng 100% (2/2), đồng thời có độ trễ truy vấn cực thấp (3.7 ms so với 121.7 ms của E2 và 253.4 ms của E1). Kết quả trên test một lần nữa khẳng định M1 vượt trội với Hit@1/Hit@3 đạt 100% và Grounded Answer đạt 0.80/2 (so với 0.00/2 của B0).

---

## 3. Agent và đánh giá

- **Tool, số lượt gọi, điều kiện dừng, cách chọn context và xử lý lỗi**:
  - Tool: Công cụ tìm kiếm tài liệu (`search`) tích hợp trong vòng lặp agent.
  - Số lượt gọi tối đa: 4 lượt (`max_searches = 4`), mỗi lượt lấy $k=3$ chunks.
  - Điều kiện dừng: Dừng khi model ra quyết định `answer`, gọi `abstain`, gặp truy vấn trùng lặp (`duplicate_query_stop`), hoặc chạm ngưỡng 4 lượt search.
  - Context packing: Thu thập các chunk nguyên vẹn theo thứ tự tìm thấy, loại bỏ chunk trùng lặp ID, cắt theo giới hạn ký tự `budget`.

### Bảng đánh giá Agent trên tập Dev và Test

| Cấu hình / split | Tool use | Số lượt gọi TB | Grounded answer (0/1/2, mean) | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm | Độ trễ agent trung vị | Token (Prompt / Completion) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **B0 / dev** | 17/17 (100%) | 1.06 | 0:14, 1:0, 2:1 (0.13) | 1/4 (25.0%) | 4/4 (100%) | 1/2 (50.0%) | 12/15 (80.0%) | 15.415 ms | 18.024p / 1.738c |
| **E1 / dev** | 17/17 (100%) | 1.00 | 0:9, 1:1, 2:5 (0.73) | 14/24 (58.3%) | 10/10 (100%) | 2/2 (100%) | 5/15 (33.3%) | 33.787 ms | 27.121p / 3.316c |
| **M1 / dev** | 17/17 (100%) | 1.00 | 0:6, 1:2, 2:7 (1.07) | 16/27 (59.3%) | 10/10 (100%) | 2/2 (100%) | 5/15 (33.3%) | 34.188 ms | 27.271p / 3.491c |
| **E2 / dev** | 17/17 (100%) | 1.00 | 0:7, 1:2, 2:6 (0.93) | 13/29 (44.8%) | 11/11 (100%) | 2/2 (100%) | 4/15 (26.7%) | 37.121 ms | 27.180p / 4.007c |
| **B0 / test** | 7/7 (100%) | 1.00 | 0:5, 1:0, 2:0 (0.00) | 1/1 (100%) | 1/1 (100%) | 2/2 (100%) | 4/5 (80.0%) | 21.587 ms | 7.421p / 780c |
| **M1 / test** | 7/7 (100%) | 1.00 | 0:3, 1:0, 2:2 (**0.80**) | 2/5 (40.0%) | 2/2 (100%) | 2/2 (100%) | 3/5 (60.0%) | 31.312 ms | 11.133p / 1.434c |

### Minh họa câu trả lời thực tế (Audit Cases)

| Query ID | Split / Config | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|---|
| **q03** | dev / B0 | Có đáp án | `search("deliberate slowing down of work...")` $\to$ chunk `2750ebfa` (p.3) | *"The deliberate slowing down of work in England is called 'hanging it out' and in Scotland, it is called 'ca canae'. [2750ebfa...]"* | **Grade 2**: Câu trả lời chính xác, trích dẫn đúng nguồn hỗ trợ. |
| **q10** | dev / M1 | Có đáp án | `search("law of heavy labor pig-iron example")` $\to$ chunk `84c109ab` (p.18) | *"A first-class workman can only be under load 43 per cent of the day. He must be entirely free from load during 57 per cent of the day. [84c109ab...] [7396b697...] [031cb491...]"* | **Grade 2**: Nêu chính xác hai tỷ lệ 43% và 57%, chunk 84c109ab hỗ trợ đầy đủ. |
| **ua-dev-01** | dev / M1 | Thiếu evidence | `search("percentage of U.S. manufacturing firms...")` $\to$ không có số liệu 2025 | *"The collection does not provide sufficient evidence to answer this question."* (action: abstain) | **Abstention đúng**: Agent không bịa đặt số liệu năm 2025, từ chối chính xác sau khi đã tìm kiếm. |
| **ua-dev-02** | dev / B0 | Thiếu evidence | `search("fixed dollar cash premium...")` $\to$ chunk `a5484bba` | *"Taylor prescribes a cash premium of $500 for every accepted worker improvement. [a5484bba...]"* | **Grade 0 (Ảo giác)**: Chunk a5484bba chỉ yêu cầu thưởng tiền mặt nhưng không có con số $500. Model tự bịa số liệu. |
| **q18** | test / M1 | Có đáp án | `search("Thompson hiring bicycle-ball inspectors quality")` $\to$ chunk `6f590e99` (p.26) | *"a low personal coefficient [6f590e99...] [37dc7f1e...]"* | **Grade 2**: Trả lời chính xác phẩm chất "a low personal coefficient", trích dẫn đúng chunk nguồn. |
| **q20** | test / M1 | Có đáp án | `search("bicycle-ball inspection worker falling behind help")` $\to$ chunk `359ab8b3` (p.27) | *"Management was supposed to help the worker by sending a teacher to each individual who was found to be falling behind to find what was wrong, to straighten her out, and to encourage and help her to catch up. [359ab8b3...]"* | **Grade 2**: Nêu đúng giải pháp gửi người hướng dẫn (teacher) giúp sửa sai và bắt kịp tiến độ. |
| **ua-test-01** | test / M1 | Thiếu evidence | `search("Vietnam adopted Taylor's scientific management 2024")` $\to$ không có dữ liệu | *"The collection does not provide sufficient evidence to answer this question."* (action: abstain) | **Abstention đúng**: Toàn bộ 283 chunk không có dữ liệu Việt Nam năm 2024, agent từ chối chuẩn xác. |

- **Đường dẫn artifact & kết quả**:
  - Dev traces & summaries: `results/agent/dev/{B0,E1,M1,E2}/`
  - Test traces & summaries: `results/agent/test/{B0,M1}/`
  - Đóng băng dữ liệu: `benchmark/freeze.json`
  - Lựa chọn cấu hình: `results/selection.json`
- **Đánh đổi và giới hạn**:
  - Tăng context budget từ 800 lên 4.000 ký tự giúp điểm số tăng vượt trội (từ 0.00 lên 0.80 trên test), nhưng làm tăng độ trễ xử lý của model từ ~21s lên ~31s do kích thước context gửi vào mô hình lớn hơn.
  - Mô hình nhỏ 3B cục bộ đôi khi vẫn có xu hướng trích dẫn thừa 1-2 chunk bối cảnh xung quanh ngoài chunk chứa bằng chứng trực tiếp, dẫn đến tỷ lệ citation hợp lệ tuyệt đối đạt ~40-60%.

---

## Thử thách nâng cao (nếu thực hiện)

- Tài liệu, nguồn, quyền sử dụng, số trang và phiên bản:
- Cách xử lý text, bảng, ảnh, biểu đồ; phạm vi index và phần chưa xử lý:
- Cách chia benchmark, schema evidence, model và cấu hình đã chốt trên dev:

| Nhóm test | Đạt / tổng | Tỷ lệ | Lỗi / nhận xét |
|---|---|---|---|
| Bảng | | | |
| Ảnh, sơ đồ | | | |
| Biểu đồ | | | |
| Multi-hop | | | |
| Text | | | |
| Thiếu bằng chứng | | | |
| **Toàn bộ** | | | |

- Trace, nhãn và evidence từng câu; thời gian, chi phí, lỗi hệ thống:
- Giới hạn, cách tái lập và đường dẫn benchmark:
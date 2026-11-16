# Dùng lại dữ liệu Day 6 trong Day 7

Phần kết nối được triển khai trong `starter/day6.py`. Chạy lệnh từ thư mục gốc **Day 7**. Hai repo hiện nằm cạnh nhau trong `Desktop\L3B`.

## Dữ liệu được dùng lại

- Corpus: *The Principles of Scientific Management*, Project Gutenberg #6435; 46 logical units. `page` là số unit HTML, không phải trang in.
- Cấu hình mặc định **C**, đã chọn theo dev Day 6: 283 paragraph chunks, tối đa 1.000 ký tự, không overlap. Bộ lọc chỉ đến từ câu hỏi/người dùng.
- PostgreSQL + pgvector: bảng `rag_chunks`, `strategy = 'C'`, cosine similarity. Kết nối đọc từ biến môi trường `DATABASE_URL`, hoặc `starter/.env` của Day 6. Không sao chép secret.
- Embedding: `intfloat/multilingual-e5-small`, revision `614241f622f53c4eeff9890bdc4f31cfecc418b3`, **384 chiều**, normalize; câu hỏi có tiền tố `query: `, tài liệu đã index có tiền tố `passage: `.
- Model lấy từ `starter/results/run_metadata.json`, không lấy `EMBEDDING_MODEL` trong `.env` vì file đó hiện ghi model khác.

Không chạy lại ingestion hoặc `python -m src.run` của Day 6 khi chỉ cần truy vấn. Adapter không có thao tác INSERT/UPDATE/DELETE và mở transaction read-only.

## 1. Dùng Python đã cài của Day 6

PowerShell, từ repo Day 7:

```powershell
$day6 = Resolve-Path '..\K5-L3B-DAY06-TranDinhNhatMinh-2A202604083-DataIngestion'
$py = Join-Path $day6 'starter\.venv\Scripts\python.exe'
& $py --version
```

Có thể tạo venv riêng ở Day 7 và cài `python -m pip install -r requirements.txt`. File requirements giữ phiên bản Sentence Transformers đã ghi trong lần chạy Day 6. Môi trường Day 6 hiện có Sentence Transformers 6.3.0, khác phiên bản 5.1.2 trong metadata; không tự hạ cấp môi trường cũ. Model/revision/prefix/normalization vẫn được cố định. Nếu so sánh benchmark giữa môi trường, ghi lại phiên bản thực tế.

## 2. Khởi động database cũ

```powershell
docker --context desktop-linux compose -p day6_student_starter --project-directory "$day6\starter" up -d --wait
```

Phải giữ project name `day6_student_starter` để dùng đúng volume `day6_student_starter_pg_data`. Repo đã chuyển từ Desktop vào Desktop\L3B, nên container cũ có thể còn bind mount sai đường dẫn `init.sql`; lệnh Compose trên cập nhật container và giữ volume. Không xóa volume.

## 3. Chuyển file dữ liệu

```powershell
& $py starter/day6.py prepare --day6-root "$day6" --strategy C
& $py starter/day6.py check
```

`prepare` chỉ đọc file và tạo:

| File trong `starter/data/day6/` | Nội dung |
|---|---|
| `chunks.json` | 283 chunks theo schema Day 7 |
| `questions_dev.json` | 15 câu dev có đáp án từ Day 6 |
| `questions_test.json` | 5 câu test có đáp án từ Day 6 |
| `manifest.json` | Nguồn, model/revision, strategy, checksum file và giới hạn |

`prepare` không xác nhận database; trường `database_verified: false` phản ánh điều đó. Chạy `check` để đối chiếu toàn bộ ID, nội dung, metadata và chiều vector với database đang kết nối. Kết quả kiểm tra in riêng, không sửa manifest hay database.

**ID:** file `chunks_C.jsonl` của Day 6 dùng ID dạng `paragraph-...`, trong khi database dùng SHA-256 của `[strategy, source, page, text, occurrence]`. Adapter tái tạo chính xác ID database và giữ ID export trong `metadata.day6_export_chunk_id`. Nhờ đó dense, TF-IDF và hybrid dùng chung citation ID. `source_id` là `gutenberg-6435`; `source` giữ đường dẫn HTML gốc.

## 4. Truy vấn thử

TF-IDF chạy offline, chỉ cần Python chuẩn:

```powershell
& $py starter/day6.py search --method tfidf --query "systematic management inefficiency"
```

Dense dùng vector đã lưu, chỉ tạo embedding cho câu hỏi. Lần đầu cho phép tải đúng revision model nếu cache cũ không còn:

```powershell
& $py starter/day6.py search --method dense --query "How does Taylor suggest remedying inefficiency?" --allow-download
```

Sau khi cache đủ, bỏ `--allow-download` để bắt buộc dùng model local. Không có fallback vector giả. Tùy chọn `--method hybrid` kết hợp dense và TF-IDF bằng Reciprocal Rank Fusion (`k=60`, ít nhất 20 candidates mỗi nhánh).

Từ code agent:

```python
# Chạy trong starter/, hoặc thêm starter/ vào sys.path như các test của repo.
from day6 import Day6Retriever
from agentic import run_agent

retriever = Day6Retriever(method="dense")  # hoặc tfidf / hybrid
retriever.check()
# controller phải là adapter gọi LLM thật:
result = run_agent(
    question, controller, retriever,
    top_k=3, budget=4000, max_searches=4,
)
```

`Day6Retriever.search(query, filters=..., top_k=...)` tương thích khung `agentic.py`. Chỉ truyền query và filter từ người dùng/model; không truyền gold hoặc expected_answer. Budget 4.000 ký tự là cấu hình khởi đầu cho chunk dài tới 1.000 ký tự, cần đo trên dev trước khi chọn.

## Kiểm tra và phần tiếp theo

```powershell
& $py -m unittest discover -s tests -v
```

Phần này chuẩn bị dữ liệu và retriever, chưa phải toàn bộ bài nộp Day 7:

- Benchmark Day 6 có **20 câu có đáp án**; Day 7 cần thêm **2 câu thiếu bằng chứng cho dev và 2 cho test**, tổng ít nhất 24.
- Gold q13 trải qua ranh giới paragraph chunks. Không sửa evidence để làm đẹp điểm; baseline mẫu chỉ nhận evidence nằm trọn trong một chunk nên không chạy nguyên xi với toàn bộ cấu hình C. Cần đánh giá evidence đa chunk riêng hoặc mở rộng evaluator.
- Nối controller với model thật, benchmark ba phương pháp và ít nhất hai cấu hình cải thiện trên dev, rồi cố định cấu hình trước khi chạy test.
- Không dùng test để điều chỉnh; nếu test Day 6 từng được dùng để chỉnh hệ thống thì phải tạo test mới. Chưa chạy benchmark test trong bước kết nối này.
- ID và số chiều đúng chưa chứng minh model đã tạo vector trong database: bảng Day 6 không lưu danh tính embedding model. Adapter dựa trên run metadata đi kèm và đối chiếu nội dung index; giữ metadata để truy vết.
- Dữ liệu sinh trong `starter/data/` được Git ignore; lệnh `prepare` là cách tái tạo. Không commit cache model hay `.env`.

## Kết quả xác minh trên máy

- PostgreSQL đã khởi động lại trên đúng volume Day 6.
- `check`: 283/283 chunks C khớp ID, text, metadata, source/page và vector 384 chiều.
- `check --with-model`: embedding lại một passage mẫu bằng model/revision đã ghi và so với vector cũ; cosine similarity **0.9999999403953677**, đạt ngưỡng 0.999. Đây là kiểm tra tương thích một vector mẫu, không phải benchmark chất lượng retrieval.
- `python -m unittest discover -s tests -v`: **22 tests pass**, bao gồm 7 tests mới cho adapter. Test cũ đã được sửa để đọc JSON bằng UTF-8 trên Windows.

Lệnh kiểm chứng model sau khi cache đã đủ:

```powershell
& $py starter/day6.py check --with-model
```

Khi tải model lần đầu bị kẹt ở Xet, thử đặt `$env:HF_HUB_DISABLE_XET = '1'` trước lệnh có `--allow-download`. Cache model nằm trong `.cache/huggingface/` của Day 7 khi cache Day 6 không còn; thư mục này được Git ignore.

Smoke test thật cho dense, TF-IDF và hybrid đã chạy trên cùng câu hỏi về Introduction; mỗi phương pháp trả 3 chunks đúng filter. Trace, kiểm chứng vector mẫu và phiên bản thư viện được lưu tại `starter/results/day6-reuse/smoke.json` (được Git ignore). Đây là kiểm tra tích hợp, chưa phải điểm benchmark Day 7.

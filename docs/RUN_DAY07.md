# Chạy bài Day 07

Chạy các lệnh PowerShell bên dưới từ thư mục Day 07. Mã nguồn dùng Python 3.11+; bộ điều khiển Ollama dùng HTTP của thư viện chuẩn, không cần SDK hay API key.

## Môi trường và dữ liệu

- Dữ liệu kế thừa: [DAY6_REUSE.md](DAY6_REUSE.md). Không chạy lại ingestion hay ghi đè vector store.
- PostgreSQL + pgvector: bảng `rag_chunks`, cấu hình C, 283 chunk, vector 384 chiều.
- Embedding: `intfloat/multilingual-e5-small`, revision `614241f622f53c4eeff9890bdc4f31cfecc418b3`, chuẩn hóa vector; tiền tố `query: ` / `passage: `.
- Cấu hình kết nối lấy từ biến `DATABASE_URL` hoặc `.env` Day 06. Không đưa `.env` hay chuỗi kết nối vào Git.
- Ollama phải được mở trước. Cả `qwen2.5:7b` và `llama3.2:3b` đã sinh JSON trong smoke test. Sau khi Qwen gây áp lực RAM lớn ở lượt thử dev, người học chọn dùng `llama3.2:3b` làm model chính thức và chạy lại toàn bộ so sánh. Không đổi model giữa các cấu hình so sánh.
- Có thể dùng venv Day 06 sẵn có; `requirements.txt` ghi dependency dự án. Môi trường đo thực tế có sentence-transformers 6.3.0, khác pin 5.1.2; xem metadata lần chạy và bước kiểm tra vector mẫu. Không coi hai môi trường là đã kiểm chứng tương đương nếu chưa kiểm tra.

```powershell
$day7Python = '..\K5-L3B-DAY06-TranDinhNhatMinh-2A202604083-DataIngestion\starter\.venv\Scripts\python.exe'
& $day7Python starter/day6.py prepare --strategy C
& $day7Python starter/day6.py check --with-model
& $day7Python src/day07.py check --model llama3.2:3b
& $day7Python -m unittest discover -s tests -v
```

`prepare` của adapter chỉ tái tạo bản chuyển đổi Day 06. Các nhãn bổ sung Day 07 được lưu riêng tại `benchmark/`, không bị lệnh đó ghi đè. Manifest bản chuyển đổi chứa đường dẫn Day 06 trên máy; tái tạo bằng lệnh trên ở máy mới trước khi tạo một bộ thí nghiệm mới. Không sửa đầu vào của bộ kết quả đã đóng băng để chạy tiếp.

## Chuẩn bị benchmark và chạy dev

```powershell
& $day7Python src/day07.py prepare
& $day7Python src/day07.py run --split dev --model llama3.2:3b
```

Lệnh chuẩn bị tạo 17 câu dev và 7 câu test, giữ nguyên bằng chứng gốc của q13 và thêm nhóm hai phần bắt buộc. Bốn câu thiếu bằng chứng có lý do kiểm tra toàn corpus. `benchmark/freeze.json` lưu SHA-256 của corpus, manifest, nhãn và giao thức thí nghiệm. Chạy lại `prepare` chỉ kiểm tra khóa; không tự tạo lại khóa khi dữ liệu đã đổi.

| ID | Phương pháp | Context | Mục đích |
| --- | --- | --- | --- |
| B0 | Dense | 800 ký tự | Baseline |
| E1 | Dense | 4.000 ký tự | Đo tăng ngân sách context |
| M1 | TF-IDF | 4.000 ký tự | So sánh phương pháp |
| E2 | Hybrid RRF | 4.000 ký tự | So với E1 để đo phương pháp |

Mỗi cấu hình cố định một phương pháp; model tự chọn query và hành động tiếp theo. Giới hạn chung: top-3, tối đa 4 lượt tìm; temperature 0, seed 42, context model 4096 token, output tối đa 512 token. Ngân sách 800/4.000 **ký tự của context đã đóng gói** khác với context window tính theo token của model. Ghép nguyên block theo thứ tự nhận, bỏ trùng ID và bỏ block không vừa ngân sách. Sau lỗi lặp query trong lượt thử dev, vòng lặp có chặn query trùng hệt để tránh tiếp tục gọi cùng truy vấn trên dữ liệu bất biến; guard này dùng chung cho toàn bộ lượt so sánh Llama.

Trace được lưu sau mỗi câu. Nếu phiên chạy bị gián đoạn, thêm `--resume` với cùng tham số và cùng mã nguồn. Không dùng output cũ với model hoặc pipeline khác.

```powershell
& $day7Python src/day07.py run --split dev --model llama3.2:3b --resume
```

`--retrieval-only` là chế độ chẩn đoán dev riêng tại `results/retrieval_only/`; không phải benchmark agent và không được dùng thay dữ liệu agent để chọn cấu hình.

## Kiểm tra nguồn và chọn cấu hình

```powershell
& $day7Python src/day07.py review-packet --split dev --config B0
# Đọc audit_packet.json; điền reviews.template.json thành reviews.json.
& $day7Python src/day07.py summarize --split dev --config B0 --reviews results/agent/dev/B0/reviews.json
```

Lặp lại với E1, M1, E2. Mỗi nhãn cần lý do dựa trên nguồn; từng citation cần nhãn hỗ trợ và lý do riêng. ID nằm trong context chưa chứng minh nguồn hỗ trợ câu trả lời. Ghi đúng người/công cụ chấm trong `reviewer`; chỉ đặt `student_verified: true` sau khi người học thực sự kiểm tra.

```powershell
& $day7Python src/day07.py select
```

Tiêu chí đã khai báo: tỷ lệ xử lý đúng trên toàn bộ dev (lỗi tính không đạt), tỷ lệ context đủ bằng chứng, Evidence Hit@3 query gốc, độ trễ agent trung vị thấp hơn; nếu vẫn bằng nhau thì thứ tự B0, E1, M1, E2. Cấu hình được chọn có thể là baseline. `results/selection.json` khóa cả hash kết quả dev, mã nguồn, dữ liệu và digest model.

Hit@1/Hit@3 một chunk đòi toàn bộ evidence gốc nằm trong chunk đúng nguồn/trang. q13 có thể không đạt các metric này dù các chunk cùng nhau đủ bằng chứng. `collective_hit3` và `context_evidence` tính đủ mọi phần của một nhóm; các nhóm thay thế nối bằng OR. Các câu thiếu bằng chứng không nằm trong mẫu số retrieval. Báo thêm điểm context và grounded answer 0/1/2 theo `BENCHMARK.md`.

## Test sau khi chọn bằng dev

Thay `<SELECTED>` bằng cấu hình trong `results/selection.json`:

```powershell
& $day7Python src/day07.py run --split test --configs B0 <SELECTED> --model llama3.2:3b
& $day7Python src/day07.py review-packet --split test --config B0
# Điền nhãn, tổng hợp B0 và cấu hình được chọn như bước dev.
```

Không chạy test nếu chưa có lựa chọn dev hợp lệ. Mã kiểm tra model/digest, hash dữ liệu, pipeline và kết quả dev. Đọc đủ 5 câu test có đáp án cho context và cả 7 câu cho câu trả lời, ở baseline và cấu hình được chọn. Không chọn lại sau khi xem test.

Trace chứa dữ liệu nguồn và nội dung prompt để kiểm toán, nhưng không chứa thông tin kết nối database. Khi chia sẻ, kiểm tra quyền của dữ liệu và chỉ gửi các artifact được phép.

## Giao thức Ollama

Controller gọi [`POST /api/chat`](https://docs.ollama.com/api/chat), dùng [`format` là JSON schema](https://docs.ollama.com/capabilities/structured-outputs), không stream. Các hành động là `search`, `answer`, `abstain`. Lỗi HTTP, timeout, JSON không hợp lệ hay output chạm giới hạn được ghi thành lỗi; không thay bằng đáp án giả.

Model trả nội dung và danh sách ID citation trong JSON. Renderer thêm các ID do model chọn vào cuối câu trả lời nếu chưa có inline, sau khi kiểm tra các ID thuộc context cuối. Renderer không chọn nguồn hoặc sửa nội dung khẳng định. Trace giữ nguyên phản hồi model và ghi riêng hành động sau định dạng. Kiểm tra nguồn hỗ trợ vẫn cần thực hiện khi chấm; định dạng hợp lệ không chứng minh câu trả lời đúng. Toàn bộ phép so sánh chính chạy thống nhất với cùng renderer.

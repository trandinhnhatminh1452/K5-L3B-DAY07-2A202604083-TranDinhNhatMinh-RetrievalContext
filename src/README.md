# Code

- `day07.py`: chuẩn bị và khóa nhãn; chạy dev/test; tổng hợp review và chọn cấu hình.
- `ollama_controller.py`: bộ điều khiển LLM Ollama thật qua HTTP và JSON schema.
- `evidence.py`: xác thực nhãn, các phương án OR và các phần bằng chứng bắt buộc AND.

Agent dùng adapter PostgreSQL/pgvector và ba phương pháp trong `starter/day6.py`, cùng vòng lặp `starter/agentic.py` đã bổ sung hành động từ chối. Xem [hướng dẫn chạy](../docs/RUN_DAY07.md).

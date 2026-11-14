# Day 06 · Data Ingestion Engineering

**Lab cá nhân** · **4 giờ** · **Dữ liệu tự chọn**

Biến tài liệu thành kho tri thức có thể truy xuất cho AI agent, rồi đo xem cách chuẩn bị dữ liệu nào hiệu quả nhất.

| 📄 Dữ liệu | 🧩 Chiến lược | 🎯 Benchmark |
|---|---|---|
| Tối thiểu **30 trang nội dung** | Tối thiểu **3 cấu hình** | Tối thiểu **20 câu hỏi** |

## Bắt đầu nhanh

Dùng [starter PostgreSQL + pgvector](starter/README.md) để dựng database và hoàn thiện các hàm TODO, hoặc tự chọn vector store tương đương. Starter chỉ là khung: **bạn tự hiện thực pipeline và benchmark**.

## 1. Xây pipeline

![Đọc tài liệu, chuẩn hóa, chia chunk, tạo vector, lưu và kiểm tra](assets/ingestion-pipeline.svg)

Chọn tài liệu được phép sử dụng; giữ **ID tài liệu, nguồn, trang/vị trí** và metadata phù hợp. Dùng embedding model thật qua API hoặc chạy local; lưu và truy xuất bằng pgvector, Qdrant, Chroma hoặc một store tương đương.

## 2. So sánh và đánh giá

![Ba chiến lược được đánh giá trên cùng corpus và benchmark dev/test](assets/benchmark-workflow.svg)

Dùng cùng dữ liệu, model và câu hỏi cho cả ba cấu hình. Tách **15 câu dev / 5 câu test** trước khi chạy; chọn cấu hình trên dev, rồi đánh giá test sau khi cố định. Mỗi câu hỏi cần có đoạn bằng chứng và trang nguồn; thiết kế bộ câu hỏi và đánh giá theo [BENCHMARK.md](BENCHMARK.md).

Đánh giá theo năm khía cạnh: **độ chính xác truy xuất**, **tính mạch lạc của chunk**, **tính hữu dụng của metadata**, **chất lượng grounding** và **tác động của chiến lược dữ liệu**. Giải thích lựa chọn bằng kết quả, ví dụ lỗi và các đánh đổi.

## 3. Chuẩn bị bài nộp

1. Chạy pipeline và lưu cấu hình, kết quả top-3, trace cùng nhãn đánh giá thủ công.
2. Điền [thông tin học viên](STUDENT.md), [nhật ký thí nghiệm](artifacts/experiment_log.csv) và bản sao [mẫu báo cáo cá nhân](report/individual/MSSV_HoTen.md).
3. Kiểm tra mã nguồn, cách chạy, manifest dữ liệu và checklist trong [SUBMISSION.md](SUBMISSION.md).

## Tài liệu hướng dẫn

| Bạn cần tìm | Tài liệu |
|---|---|
| Setup baseline PostgreSQL + pgvector | [starter/README.md](starter/README.md) |
| Các bước làm và cách tính chỉ số | [GUIDE.md](GUIDE.md) |
| Tiến độ trong 4 giờ | [CHECKPOINTS.md](CHECKPOINTS.md) |
| Năm tiêu chí benchmark và bộ câu hỏi | [BENCHMARK.md](BENCHMARK.md) |
| Tiêu chí chấm điểm | [RUBRIC.md](RUBRIC.md) |
| Quy tắc sử dụng dữ liệu, AI và API | [RULES.md](RULES.md) |
| Cách nộp bài | [SUBMISSION.md](SUBMISSION.md) |

[data/format-example.json](data/format-example.json) chỉ minh họa định dạng, không phải corpus đủ điều kiện. Phần xử lý và benchmark do bạn tự hiện thực; không bắt buộc xây chatbot.

---

## Bài làm — cách setup và chạy (hoàn thành mốc 4:00)

| Thành phần | Lựa chọn |
|---|---|
| Corpus | [docs/222_QLSXKS_L06_Nhom_2.pdf](docs/), báo cáo BTL của nhóm học viên — manifest: [data/manifest.json](data/manifest.json); 26 trang nội dung (yêu cầu đã xác nhận: 20–30 trang) |
| OCR | Tesseract 5.3.0 + `vie` trong Docker ([starter/ocr/Dockerfile](starter/ocr/Dockerfile)); 14/20 ảnh được nhận (độ tin cậy ≥80), kết quả [starter/data/ocr/ocr_results.json](starter/data/ocr/ocr_results.json) |
| Embedding | `google/embeddinggemma-2` rev `f9972ac`, local CPU, chỉ text, 768 chiều, float32, chuẩn hóa L2; prompt document `title: none \| text: `, query `task: search result \| query: `; chi phí 0 |
| Vector store | PostgreSQL 16.16 + pgvector 0.8.7 (Docker), bảng `rag_chunks`, cosine `<=>`, exact search |
| Benchmark | [benchmark/questions.json](benchmark/questions.json): 24 câu, 17 dev / 7 test, 5 câu filter `section` (3 dev, 2 test) |
| Chiến lược | A fixed 500/50 · B theo đoạn ≤800 · C = B + lọc `section` — chi tiết và kết quả [artifacts/experiment_log.csv](artifacts/experiment_log.csv) |

```bash
cd starter
# Windows: repo nằm ở đường dẫn sâu nên venv đặt ngoài repo để tránh lỗi MAX_PATH khi cài torch
py -3.13 -m venv C:/Users/<user>/.venvs/day06        # macOS/Linux: python3 -m venv .venv
PY=C:/Users/<user>/.venvs/day06/Scripts/python.exe   # hoặc .venv/bin/python
$PY -m pip install -r requirements.txt
$PY -m pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cpu
$PY -m pip install -r requirements-embedding.txt
cp .env.example .env
docker compose -p day6_student_starter up -d --wait
docker build -t day6-ocr:tesseract-vie ocr
$PY scripts/check_setup.py

$PY -m src.run ocr                    # OCR ảnh trong PDF (cache vào data/ocr/)
$PY -m src.run verify                 # đọc PDF + OCR, log trang, kiểm chứng evidence benchmark
$PY -m src.run ingest                 # chunk A/B, embed chunk mới/đổi, upsert, xóa chunk cũ
$PY -m src.run retrieve --split dev   # top-3 cho A, B, C
$PY -m src.run evaluate --split dev   # Hit@1, Evidence Hit@3 (n/N)
$PY -m src.run select                 # chọn chiến lược trên dev theo tiêu chí đã chốt
$PY -m src.run retrieve --split test  # chỉ chạy được sau khi đã select
$PY -m src.run evaluate --split test
$PY -m src.run audit-samples          # mẫu chấm coherence / metadata / grounding
$PY -m src.run audit-summary          # tổng hợp nhãn trong results/audit/*_labels.csv
```

### Kết quả

| Strategy | Split | Hit@1 | Evidence Hit@3 | Lấy đúng trang @3 | Filter tuân thủ | Chunks |
|---|---|---|---|---|---|---|
| A | dev | 8/17 | 10/17 | 17/17 | 0/3 | 112 |
| B | dev | 14/17 | 17/17 | 17/17 | 0/3 | 91 |
| C | dev | 15/17 | 17/17 | 17/17 | 3/3 | 91 (chung B) |
| A | test | 4/7 | 5/7 | 5/7 | 0/2 | |
| B | test | 4/7 | 5/7 | 6/7 | 0/2 | |
| **C (chọn trên dev)** | test | 4/7 | 5/7 | 7/7 | 2/2 | |

| Strategy | Coherence /2 (n=10) | Metadata đủ và đúng | Grounding 0/1/2 (n=7) | Grounding /2 |
|---|---|---|---|---|
| A | 1,30 | 20/21 | 2/0/5 | 1,43 |
| B | 2,00 | 21/21 | 1/1/5 | 1,57 |
| C | 2,00 (chung chunk B) | 21/21 | 0/1/6 | 1,86 |

Nhãn thủ công: bản nháp do Claude (AI) chấm theo thang trong GUIDE.md kèm lý do từng dòng, **học viên đã duyệt toàn bộ** (sửa grounding q20/C từ 1 thành 2). Báo cáo: [report/individual/2A202603987_KhuongHoangNguyen.md](report/individual/2A202603987_KhuongHoangNguyen.md).

Kết quả trong [starter/results/](starter/results/): `ingest_log.json`, `chunks_{A,B}.jsonl`, `ingest_report.json` (các lần chạy: nạp mới → nạp lại 0 embed → giả lập chunk cũ/đổi → nạp lại sau OCR, embed lại toàn bộ do metadata có thêm `content_source` → nạp lại 0 embed), `benchmark_verification.json`, `traces_{dev,test}.jsonl`, `eval_{dev,test}.json` + `_per_query.csv`, `selection.json`, `audit/`, `audit_summary.json`.

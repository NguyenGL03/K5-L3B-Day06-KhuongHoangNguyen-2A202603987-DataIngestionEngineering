# Báo cáo cá nhân — Day 6

Học viên: Khương Hoàng Nguyên · MSSV 2A202603987 · L3B · Thời điểm chạy: 2026-11-14 (UTC+7)

## Dữ liệu và pipeline

- **Chủ đề, nguồn/quyền, số trang và quy tắc tính trang:** Báo cáo bài tập lớn "Quản lý sản xuất cho kỹ sư" (HK222, ĐH Bách Khoa TP.HCM) về Công ty CP Nước khoáng Quảng Ninh, do nhóm của học viên thực hiện nên được phép sử dụng cho lab; file `docs/222_QLSXKS_L06_Nhom_2.pdf` (sha256 `a7746d53…0889`). 30 trang vật lý, **26 trang nội dung** (p4–p29). Loại p1 (bìa), p2 (mục lục), p3 (bảng phân công), p30 (tài liệu tham khảo); p1 và p3 có họ tên/MSSV nên không đưa vào index. Yêu cầu corpus đã được xác nhận lại là 20–30 trang. Chi tiết từng trang trong [data/manifest.json](../../data/manifest.json).
- **Chuẩn hóa, lỗi và giới hạn:**
  - Trích text bằng pypdf 6.21.4, chuẩn hóa Unicode NFC, bỏ ký tự ẩn và ligature, gộp khoảng trắng nhưng giữ xuống dòng (cần cho danh sách/heading).
  - Header/footer (dòng lặp ở ≥60% trang) được loại; chú thích chân trang ở p5, p19, p20, p21 được chuyển vào metadata, số chú thích dính cuối câu (vd. "tích cực.1") được bỏ.
  - Heading đánh số được nhận diện tuần tự (chương 1→8, mục con thuộc chương hiện tại) để gán `section`.
  - Log từng trang ở `starter/results/ingest_log.json`: 0 trang rỗng, 0 lỗi trích xuất, 3 trang ít text vì nội dung là ảnh (p6, p10, p17).
  - **OCR:** Tesseract 5.3.0 + `vie` trong Docker. Mỗi ảnh OCR 2 biến thể (xám ×2; nhị phân hóa), `--psm 4`, giữ bản có độ tin cậy trung bình cao hơn, nhận khi ≥80. Nhận 14/20 ảnh (bảng tài chính p5, bảng kế hoạch p6, quy trình p12, sơ đồ mặt bằng p13, chuỗi cung ứng p14, bảng lead time p17, Business Model Canvas p18, ma trận rủi ro p27), thêm 4.057 ký tự. Loại 6 ảnh: nhãn chai p9/p10 (độ tin cậy 12–78, chủ yếu nhiễu), cây sản phẩm và lịch theo ngày p17 (OCR không giữ được cấu trúc). Đã kiểm mẫu bằng mắt: bảng p5 đúng số; ma trận rủi ro p27 mất tên 7/9 dòng do ô nền màu; biểu đồ đường p6 chỉ lấy được tiêu đề; có lỗi dấu ("Thành phâm", "nhăn hiệu").
  - Giới hạn khác: chính tài liệu nguồn có câu thiếu dấu ("Ngay 25 thang 12 năm 2015…", p4); chunk không vượt trang nên đoạn văn kéo dài sang trang sau bị cắt (vd. p14→p15, p26→p27).
- **Embedding, vector store, lệnh chạy, chi phí:**
  - `google/embeddinggemma-2` revision `f9972ac22e69…`, local qua sentence-transformers 6.3.0 / transformers 5.23.0 / torch 2.15.0+cpu; chỉ nạp phần text, float32, 768 chiều, chuẩn hóa L2.
  - Document dùng prompt `title: none | text: `, query dùng `task: search result | query: `.
  - PostgreSQL 16.16 + pgvector 0.8.7 (Docker), một bảng `rag_chunks` khóa `(strategy, chunk_id)`, khoảng cách cosine `<=>`, **exact search** (không ANN index) nên filter không làm hụt top-3.
  - Lệnh chạy: `python -m src.run ocr | verify | ingest | retrieve | evaluate | select | audit-samples | audit-summary` (đầy đủ trong [README.md](../../README.md)).
  - Chi phí 0 USD (không gọi API). Thời gian trên CPU laptop:
    - tải model ~10–14 s; OCR 20 ảnh 24,5 s;
    - embed lần nạp đầy đủ cuối: A 85,7 s, B 88,2 s;
    - truy vấn: embed ~110 ms, search ~21 ms (median).
  - Nạp lại cùng phiên bản: 0 chunk embed lại, số dòng = số ID = số text phân biệt (không trùng). Giả lập dữ liệu đổi: chunk mồ côi bị xóa (1), chunk đổi hash được embed lại (1). Hash gồm model + text + metadata nên khi thêm trường `content_source` toàn bộ chunk được embed lại.
- **Metadata bắt buộc, trường lọc, nguồn giá trị, lý do:**
  - Bắt buộc: `document_id`, `source` (đường dẫn file), `page` (trang PDF vật lý).
  - Bổ sung: `printed_page` (số "Trang n/29" ở footer), `section` (slug chương, text), `section_no` (int), `section_title`, `subsection`, `content_source` (`pdf_text`/`ocr`/`mixed`), `char_start`/`char_end`.
  - `section` lấy nguyên văn từ heading đánh số trong PDF (không tự đặt), có 8 giá trị phân biệt, khớp mục lục.
  - Chọn `section` làm filter vì nhiều câu hỏi chỉ rõ nghĩa khi biết chương: "nguyên tắc" có ở chương 3 và 6, "ưu điểm" ở chương 2 và 4, "quy trình" ở chương 4 và 7.

## Cấu hình và benchmark

- **A / B / C:**
  - **A:** cửa sổ cố định 500 ký tự, overlap 50, cắt lùi về khoảng trắng gần nhất, không vượt trang; 112 chunk.
  - **B:** dựng lại đoạn từ dòng PDF (ngắt khi gặp heading, gạch đầu dòng, chú thích hình, dòng ngắn <80 ký tự hoặc dòng kết thúc bằng dấu câu), heading và khối OCR luôn mở chunk mới, gộp đoạn liên tiếp tới 800 ký tự, đoạn quá dài tách theo câu; 91 chunk.
  - **C:** dùng lại đúng index B (cùng chunk, cùng vector) và chỉ thêm `WHERE section = %s` cho câu có filter. **Khác biệt duy nhất B–C là filter SQL**; A–B chỉ khác chunker. Cùng model, store, metric, top-k = 3.
- **≥20 câu:**
  - 24 câu trong [benchmark/questions.json](../../benchmark/questions.json), chia trước khi chạy truy xuất: **17 dev / 7 test**.
  - Loại câu: trực tiếp, diễn đạt lại, điều kiện, số liệu, metadata_filter.
  - 5 câu filter `section`: q09, q13, q15 (dev); q20, q21 (test). Filter là ngữ cảnh giả định "người dùng đang đọc chương X", cung cấp trước truy xuất, không lấy từ gold.
  - Kiểm chứng evidence: `python -m src.run verify` kiểm tra từng `evidence_quote` là chuỗi con của text trang gold sau chuẩn hóa → 24/24; đồng thời kiểm giá trị filter tồn tại và đúng chương của trang gold.
  - Rủi ro rò rỉ: chỉ có 1 tài liệu nên không tách được theo tài liệu; q16 (test) và q23 (dev) cùng trang 23; q09 (dev) và q20 (test) cùng dạng câu "nguyên tắc".
- **Tiêu chí chọn (chốt trước khi xem test, ghi trong `questions.json`):** Evidence Hit@3 dev cao nhất → Hit@1 dev → grounding thủ công → ít chunk/thời gian hơn. Kết quả áp dụng ghi trong `starter/results/selection.json` (10:58:26), trước khi chạy test (lệnh `retrieve --split test` bị chặn nếu chưa có file này).

## Kết quả

| Strategy | Split | Hit@1 (n/N) | Evidence Hit@3 (n/N) | Chunk count | Thời gian | Chi phí |
|---|---|---|---|---|---|---|
| A | dev | 8/17 | 10/17 | 112 | embed 85,7 s (CPU) | 0 |
| B | dev | 14/17 | 17/17 | 91 | embed 88,2 s (CPU) | 0 |
| C | dev | 15/17 | 17/17 | 91 (chung B) | như B; search ~21 ms/câu | 0 |
| A | test | 4/7 | 5/7 | 112 | | 0 |
| B | test | 4/7 | 5/7 | 91 | | 0 |
| **C (chọn)** | test | 4/7 | 5/7 | 91 | | 0 |

Bổ sung (không thay định nghĩa chính): lấy đúng trang trong top-3 trên test là A 5/7, B 6/7, C 7/7; trên dev cả ba 17/17. Tổng token (tokenizer của model): A 15.996, B 14.091.

| Strategy | Coherence trung bình /2 (n) | Metadata đủ và đúng (n/N) | Filter tuân thủ (n/N hoặc N/A) | Grounding 0/1/2 (số câu) và trung bình /2 |
|---|---|---|---|---|
| A | 1,30 (10) — 0:2, 1:3, 2:5 | 20/21 | 0/2 | 2/0/5 — 1,43 |
| B | 2,00 (10) — 2:10 | 21/21 | 0/2 | 1/1/5 — 1,57 |
| C | 2,00 (10, cùng chunk B) | 21/21 | 2/2 | 0/1/6 — 1,86 |

Filter tuân thủ đo trên test (cả 3 top-3 thuộc đúng chương). Trên dev: A 0/3, B 0/3, C 3/3.

**Đường dẫn trace, nhãn thủ công và cách chọn mẫu** (trong `starter/results/`):
- Trace top-3 đầy đủ (text, score, metadata): `traces_dev.jsonl`, `traces_test.jsonl`.
- Chấm tự động: `eval_{dev,test}.json` và `eval_{dev,test}_per_query.csv`.
- Nhãn thủ công: `audit/coherence_labels.csv`, `audit/metadata_labels.csv`, `audit/grounding_labels.csv` (mẫu gốc ở `audit/*_samples.*`); tổng hợp ở `audit_summary.json`.
- Cách chọn mẫu:
  - **Coherence:** 10 vị trí cố định trước khi chấm = điểm giữa phần text PDF của 10 trang nội dung đầu (p4–p13); với mỗi index lấy chunk chứa vị trí đó.
  - **Metadata:** toàn bộ 7 câu test × top-3 × 3 chiến lược = 63 bản ghi, kiểm `source`, `page`, `printed_page`, `section`, `subsection` so với nguồn.
  - **Grounding:** 7 câu test × 3 chiến lược.
- Người chấm: Claude (AI) chấm nháp theo thang GUIDE.md kèm lý do từng dòng, **học viên (tác giả) duyệt toàn bộ** và sửa 1 nhãn (grounding q20/C: 1 → 2, vì câu dẫn không ảnh hưởng câu trả lời). Người chấm là tác giả nên có nguy cơ thiên lệch.

## Phân tích và kết luận

- **So sánh A–B (đổi chunker):**
  - Trên dev, B tăng Evidence Hit@3 từ 10/17 lên 17/17 và Hit@1 từ 8/17 lên 14/17. Ở cả 17 câu dev, A đều lấy đúng trang nhưng 7 câu (q01, q06, q07, q09, q13, q15, q23) có evidence bị cửa sổ 500 ký tự cắt đôi hoặc cắt mất câu dẫn.
  - Coherence 1,30 → 2,00. A có 2 vị trí điểm 0: p11 kết thúc ở "được áp dụng khi sản lượng lớn và ít", mất 4 điều kiện; p12 cắt giữa câu.
  - A sinh thêm các chunk đuôi rất ngắn ("dựng một quy trình… tránh gây", ~100 ký tự) và chúng chen vào top-3 (q16, q18, q20).
  - B ít chunk và ít token hơn (91 so với 112; 14.091 so với 15.996).
  - Trên test, A và B bằng nhau (5/7, 4/7), nên không kết luận B tốt hơn A trên mọi loại câu; khác biệt tập trung ở câu có evidence dài hoặc danh sách.
- **So sánh B–C (chỉ thêm filter):**
  - Filter sửa lỗi lấy nhầm chương: tuân thủ filter 0/2 → 2/2 trên test và 0/3 → 3/3 trên dev; lấy đúng trang test 6/7 → 7/7; grounding 1,57 → 1,86.
  - Dev Hit@1 14 → 15: q09 từ hạng 3 lên hạng 1, vì không lọc thì nguyên tắc quản lý chất lượng ở p22 đứng trên nguyên tắc bố trí mặt bằng ở p13.
  - Evidence Hit@3 test không đổi (5/7), vì hai câu filter trượt do cách chunk chứ không do truy xuất (xem ví dụ 2 bên dưới).
- **Hai ví dụ sai:**
  1. **q01 (dev, A):** top-3 = p4, p4, p29. Câu "…tiền thân là Xí nghiệp nước khoáng Quảng Ninh, được thành lập ngày 17/7/1989" bị tách: `…p004-fixed500o50-001` kết thúc ở "…được thành lập", `…-002` bắt đầu bằng "Xí nghiệp nước khoáng Quảng Ninh, được thành lập ngày 17/7/1989" nhưng mất chủ thể. Đúng trang nhưng không chunk nào chứa đủ evidence. B giữ cả đoạn (`p004-para800-001`) và hit ở hạng 1.
  2. **q21 (test, B và C):** "Những ưu điểm chính là gì?" (gold: ưu điểm của hợp đồng sản xuất, p14).
     - B (không lọc) trả p7, p22, p8. Kết quả p8 là "Ưu điểm của chiến lược cạnh tranh chi phí thấp", tức đúng dạng câu nhưng sai chủ đề.
     - C lọc đúng chương 4 nhưng danh sách 4 ưu điểm bị chunker B tách: câu dẫn và 3 ưu điểm đầu ở `p014-para800-001`, ưu điểm cuối đứng riêng ở `p014-para800-002` (49 ký tự). Top-3 của C có chunk 49 ký tự và chunk OCR nhiễu của BMC (p18) nhưng không có chunk chứa câu dẫn, nên grounding chỉ đạt 1.
     - q20/C có hiện tượng tương tự: đủ 7 nguyên tắc ở p22 nhưng câu dẫn nằm ở chunk khác, nên chấm tự động là trượt dù grounding đạt 2. Điều này cho thấy định nghĩa evidence-trong-một-chunk khắt khe hơn nhu cầu trả lời.
     - Ngoài ra ở q06 và q12 (dev), hạng 1 của B/C là chunk Kết luận p29 (văn bản tổng quát), gold đứng hạng 2.
- **Chiến lược chọn, đánh đổi, điều kiện áp dụng:**
  - Chọn **C** theo tiêu chí đã chốt trên dev (hòa B về Hit@3 17/17, hơn về Hit@1 15/17).
  - C chỉ có lợi khi ngữ cảnh chương được biết trước và đáng tin. Nếu filter sai, top-3 sẽ loại hẳn câu trả lời đúng; nếu không có filter, C quay về đúng hành vi B.
  - Chi phí thêm của C gần bằng 0 (một điều kiện SQL trên cột có index); metadata `section` đúng 100% trên mẫu kiểm tra.
  - Đánh đổi của B so với A: chunk dài hơn (median 558 so với 496 ký tự) và phụ thuộc heuristic dựng đoạn riêng cho PDF xuất từ LaTeX.
- **Hạn chế:**
  - Một tài liệu, 26 trang; test chỉ 7 câu (1 câu ≈ 14 điểm %), nên các chênh lệch trên test không có ý nghĩa thống kê và không suy rộng ngoài corpus.
  - Nhãn do tác giả duyệt (có AI chấm nháp), chưa có người chấm độc lập.
  - Dev đã được chạy truy xuất một lần trước khi thêm OCR (chỉ xem trang top-3, không chỉnh tham số theo đó). Khi viết q13/q21 đã biết cấu trúc chunk B ở trang 14/19 (chưa chạy truy xuất).
  - Thời gian embed đo trên CPU laptop dao động mạnh (cùng dữ liệu tương tự: 35 s ở lần chạy trước OCR, 86–88 s sau OCR), chỉ mang tính tham khảo.
  - OCR mất cấu trúc bảng màu (p27) và không đọc được biểu đồ đường.
  - Chunk không vượt trang.
- **Nếu làm lại hoặc mở rộng:**
  1. Chunker B giữ trọn danh sách cùng câu dẫn (cho phép vượt giới hạn tới hết danh sách, hoặc gộp đuôi <100 ký tự vào chunk trước). Thay đổi này nhắm thẳng vào lỗi q20/q21.
  2. Ghép đoạn qua ranh giới trang nhưng vẫn lưu trang bắt đầu/kết thúc.
  3. Đưa tiêu đề chương/mục vào phần `title:` của prompt document để chunk tự mang ngữ cảnh.
  4. Thử hybrid BM25 + dense cho câu số liệu và mã hiệu (QCVN, giấy phép).
  5. OCR có nhận diện bố cục bảng.
  6. Thêm tài liệu và câu hỏi test, nhờ người chấm thứ hai.

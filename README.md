# Trợ lý AI Rà soát Rủi ro Báo cáo Tài chính

> Đồ án I — Ngành Hệ thống thông tin quản lý, Đại học Bách Khoa Hà Nội
> Ứng dụng kiến trúc RAG (Retrieval-Augmented Generation) để hỗ trợ kiểm toán viên phát hiện sai lệch trọng yếu và rủi ro tuân thủ pháp lý trên báo cáo tài chính doanh nghiệp.

---

## Giới thiệu

Trong bối cảnh Việt Nam áp dụng Thông tư 99/2025/TT-BTC và lộ trình chuyển đổi sang chuẩn mực IFRS, việc rà soát rủi ro báo cáo tài chính (BCTC) ngày càng phức tạp, đòi hỏi kiểm toán viên cập nhật liên tục các quy định pháp lý. Các công cụ truyền thống như Excel chỉ kiểm tra được tính cân đối số liệu, không thể đọc hiểu ngữ cảnh hay phát hiện các kịch bản gian lận tinh vi.

Hệ thống là một chatbot dựa trên kiến trúc RAG, giúp kiểm toán viên:
- Đặt câu hỏi rà soát rủi ro dựa trên kho tri thức pháp lý (Thông tư 99/2025/TT-BTC, VSA 240, IAS 1, IAS 7, IFRS 15...)
- Tải lên một hoặc nhiều tệp báo cáo tài chính (PDF, DOCX, XLSX) để đối chiếu chéo số liệu giữa các cấu phần BCTC
- Duy trì hội thoại nhiều lượt với bộ nhớ phân tầng, hiểu các câu hỏi nối tiếp mơ hồ

**Phạm vi dữ liệu đầu vào**: bốn cấu phần BCTC bắt buộc — Báo cáo tình hình tài chính (B01-DN), Báo cáo kết quả hoạt động kinh doanh (B02-DN), Báo cáo lưu chuyển tiền tệ (B03-DN), Bản thuyết minh BCTC (B09-DN).

---

## Tác giả

Đoàn Tuấn Hùng — Đồ án cá nhân, phụ trách toàn bộ: nghiên cứu nghiệp vụ, xây dựng kho tri thức vector, RAG pipeline, Backend, Frontend và đánh giá hệ thống.

Giảng viên hướng dẫn: TS. Lê Hải Hà

---

## Kiến trúc hệ thống

```
┌───────────────────────────────────────────────────────────┐
│                FRONTEND (ReactJS + Tailwind)              │
│        Chat / Đính kèm tệp BCTC / Quản lý phiên           │
└──────────────────────────┬────────────────────────────────┘
                            │ REST API (JSON)
┌──────────────────────────▼──────────────────────────────────┐
│                    BACKEND (FastAPI)                        │
│   /api/v1/review │ /api/v1/review-file │ /api/v1/session    │
└───────┬───────────────────────────────────────┬─────────────┘
        │                                       │
┌───────▼─────────────────────────────┐   ┌─────▼─────────────────────┐
│        RAG Pipeline                 │   │   MongoDB Atlas           │
│                                     │   │   - financial_rules_vector│
│  1. Kiểm tra đầu vào (2 guard)      │   │   - conversation_sessions │
│     ├─ Phạm vi chủ đề (Gemini)      │   └───────────────────────────┘
│     └─ Độ dài câu hỏi (token)       │
│                                     │
│  2. Query Rewriting                 │
│     (viết lại câu hỏi nối tiếp      │
│      dựa trên bộ nhớ phân tầng)     │
│                                     │
│  3. HyDE                            │
│     (sinh tài liệu giả định,        │
│      trung bình vector truy vấn)    │
│                                     │
│  4. Hybrid Search                   │
│     ├─ Vector Search                │
│     │   (MongoDB Atlas,             │
│     │    Vietnamese-SBERT)          │
│     └─ BM25 Keyword Search          │
│         → Reciprocal Rank Fusion    │
│           (k=60, + source boost)    │
│                                     │
│  5. Reranker (Gemini, listwise)     │
│                                     │
│  6. Generator: Gemini 3.1 Flash-Lite│
│     (phân tích rủi ro + Markdown)   │
└─────────────────────────────────────┘
```

> **Hybrid Search**: kết hợp Vector Search (MongoDB Atlas, vector 768 chiều, mô hình `keepitreal/vietnamese-sbert`) và BM25 keyword search, hợp nhất bằng Reciprocal Rank Fusion. BM25 bắt được các thuật ngữ kỹ thuật đặc thù như "TK 511", "VSA 240" mà semantic search dễ bỏ sót.
>
> **HyDE (Hypothetical Document Embeddings)**: trước khi truy xuất, hệ thống sinh một đoạn văn bản pháp lý giả định (~200–300 từ) bằng Gemini, lấy trung bình vector của đoạn này với vector câu hỏi gốc để cải thiện chất lượng vector truy vấn — đặc biệt hữu ích với câu hỏi ngắn hoặc thiếu ngữ cảnh.
>
> **Bộ nhớ hội thoại phân tầng**: short-term giữ nguyên văn 8 cặp lượt gần nhất; long-term nén các lượt cũ hơn thành tóm tắt cộng dồn (rolling summary, tối đa 300 từ) qua Gemini; flagged accounts trích xuất tự động các mã tài khoản/điều luật đã được đề cập trong phiên để hỗ trợ viết lại câu hỏi ở các lượt sau. Mỗi phiên có TTL 24 giờ trên MongoDB.

---

## Tech Stack

| Thành phần | Công nghệ |
|---|---|
| Frontend | ReactJS + Vite + Tailwind CSS |
| Backend | FastAPI (Python) |
| LLM | Gemini 3.1 Flash-Lite (Google) |
| Embedding | `keepitreal/vietnamese-sbert` (Sentence-BERT, chạy offline, 768 chiều) |
| Vector DB | MongoDB Atlas Vector Search |
| Keyword Search | BM25 (rank_bm25), corpus load vào RAM khi khởi động server |
| Hợp nhất kết quả | Reciprocal Rank Fusion (k=60) + source boost theo Gemini |
| Reranker | Gemini listwise reranker  |
| Bộ nhớ hội thoại | Tiered memory (short-term + long-term summary + flagged accounts), lưu MongoDB |
| Đánh giá | RAGAS |
| Trích xuất tệp đính kèm | pypdf (PDF), python-docx (DOCX), pandas (XLSX/XLS đa sheet) |

---

## Kho tri thức pháp lý

Hệ thống truy xuất tri thức trong phạm vi các văn bản pháp lý và nghiệp vụ cốt lõi sau:

- Toàn văn chế độ kế toán doanh nghiệp theo **Thông tư 99/2025/TT-BTC**
- Chuẩn mực kế toán quốc tế nền tảng: **IAS 1, IAS 7, IFRS 15** (phục vụ lộ trình hội nhập IFRS)
- Chuẩn mực kiểm toán Việt Nam về nhận diện gian lận: **VSA 240** (cùng VSA 315, VSA 330)
- Bộ kịch bản rủi ro kế toán biên soạn sẵn, phục vụ bài toán đối chiếu chéo số liệu giữa các cấu phần BCTC

Mỗi chunk tri thức được gắn nhãn `metadata.category` (ví dụ `phap_ly_goc`, `chuan_muc_kiem_toan`, `chuan_muc_quoc_te`, `kich_ban_kiem_thu`, `huong_dan_nghiep_vu`) để phục vụ lọc và phân tích nguồn gốc tri thức.

---

## Cấu trúc thư mục

```
financial-audit/
│
├── financial-audit-api/
│   ├── main.py                       # FastAPI app: BM25 corpus, RAG pipeline, endpoints
│   ├── core/
│   │   └── config.py                 # Cấu hình: MONGO_URI, DB_NAME, GEMINI_API_KEY...
│   ├── services/
│   │   ├── embedding.py              # Mô hình nhúng Vietnamese-SBERT (offline)
│   │   ├── gemini_service.py         # HyDE, detect_source, rerank, generate_risk_analysis,
│   │   │                             #   rewrite_query_with_history, is_query_in_scope
│   │   └── conversation_memory.py    # Bộ nhớ phân tầng: short-term, long-term, flagged accounts
│   ├── upload_knowledge.py           # Công cụ nạp tri thức (DOCX/PDF/TXT/URL) lên MongoDB Atlas
│   ├── evaluate.ipynb                # Notebook đánh giá hệ thống bằng RAGAS
│   ├── ragas_50_result.csv           # Kết quả đánh giá định lượng trên bộ test set
│   └── requirements.txt
│
├── financial-audit-ui/
│   ├── src/
│   │   ├── App.jsx                   # Giao diện chat chính
│   │   └── main.jsx
│   ├── public/
│   ├── package.json
│   ├── tailwind.config.js
│   └── vite.config.js
│
└── README.md
```

---

## Pipeline RAG — chi tiết các giai đoạn

Pipeline xử lý một lượt truy vấn gồm các giai đoạn nối tiếp, thiết kế theo nguyên tắc fail-fast (các bước kiểm tra/lọc đặt sớm nhất để tránh tốn tài nguyên cho truy vấn không hợp lệ):

1. **Kiểm tra đầu vào** — hai guard độc lập đặt trước Query Rewriting:
   - Guard phạm vi chủ đề: Gemini phân loại câu hỏi (và trích đoạn tệp đính kèm nếu có) có thuộc phạm vi kế toán/kiểm toán/rủi ro BCTC hay không.
   - Guard độ dài: yêu cầu tối thiểu 3 token có nghĩa, bắt các câu quá ngắn mà Gemini có thể bỏ sót.

2. **Query Rewriting** — viết lại câu hỏi nối tiếp mơ hồ ("thế còn khoản mục đó thì sao") thành câu hỏi độc lập, đầy đủ ngữ cảnh, dựa trên lịch sử phiên. Câu viết lại chỉ dùng cho truy xuất, không dùng cho prompt sinh câu trả lời cuối.

3. **HyDE** — sinh đoạn văn bản pháp lý giả định, trung bình vector với câu hỏi gốc để tăng chất lượng vector truy vấn.

4. **Hybrid Search & RRF** — Vector Search (MongoDB Atlas, numCandidates=600) và BM25 (ngưỡng điểm tối thiểu 2,0) chạy song song, hợp nhất bằng RRF (k=60). Một chunk được giữ nếu điểm vector ≥ 0,50 **hoặc** được BM25 xếp hạng hợp lệ. Source boost được áp dụng sau RRF dựa trên nguồn văn bản mà Gemini xác định là liên quan nhất đến câu hỏi.

5. **Reranking** — toàn bộ candidate sau RRF được đưa vào một lần gọi Gemini duy nhất (listwise reranking), trả về thứ tự liên quan giảm dần. Top-10 chunk sau rerank được đưa vào prompt sinh câu trả lời.

6. **Sinh phân tích rủi ro** — các chunk đã lọc, lịch sử phiên và câu hỏi gốc được ghép thành prompt gửi tới Gemini 3.1 Flash-Lite. Khi không tìm thấy chunk phù hợp, hệ thống trả về thông báo từ chối thay vì để mô hình tự suy diễn — đây là cơ chế kiểm soát hallucination chủ động.

7. **Lưu hội thoại** — lượt hội thoại được ghi vào MongoDB qua module bộ nhớ phân tầng.

---

## Endpoints chính

| Method | Endpoint | Mô tả |
|---|---|---|
| `POST` | `/api/v1/session/create` | Tạo phiên hội thoại mới |
| `GET` | `/api/v1/session/{session_id}` | Lấy trạng thái phiên (lịch sử, tóm tắt, flagged accounts) |
| `POST` | `/api/v1/review` | Hỏi đáp rà soát rủi ro bằng văn bản |
| `POST` | `/api/v1/review-file` | Rà soát một hoặc nhiều tệp BCTC đính kèm (PDF/DOCX/XLSX/XLS) |
| `POST` | `/api/v1/admin/rebuild-bm25` | Xây dựng lại chỉ mục BM25 sau khi nạp tri thức mới |

---

## Setup môi trường

### Yêu cầu
- Python 3.10+
- Node.js (cho frontend)
- Tài khoản MongoDB Atlas (Vector Search) và Gemini API Key

### 1. Clone repo
```bash
git clone https://github.com/<username>/<repo-name>.git
cd <repo-name>
```

### 2. Cài đặt Backend
```bash
cd financial-audit-api
python -m venv venv

# Windows (PowerShell)
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Cấu hình biến môi trường

Tạo file cấu hình trong `core/config.py` (hoặc `.env` nếu đã chuyển sang biến môi trường) với các giá trị:

```python
MONGO_URI = "mongodb+srv://..."
DB_NAME = "do_an_chatbot_bctc"
COLLECTION_NAME = "financial_rules_vector"
VECTOR_INDEX_NAME = "default"
GEMINI_API_KEY = "..."
```

> **Lưu ý bảo mật**: không commit các giá trị key thật lên Git. Nên đưa các giá trị nhạy cảm vào `.env` và đọc qua `os.getenv()`.

### 4. Nạp kho tri thức

```bash
python upload_knowledge.py
```

Công cụ hỗ trợ chọn file (.docx/.pdf/.txt) qua hộp thoại hoặc nhập URL web để crawl, phân đoạn bằng `RecursiveCharacterTextSplitter` (chunk size 1200, overlap 200) và nhúng vector bằng `keepitreal/vietnamese-sbert` trước khi lưu vào MongoDB Atlas.

### 5. Chạy Backend

```bash
uvicorn main:app --reload
```

- Swagger docs: `http://localhost:8000/docs`

### 6. Cài đặt & chạy Frontend

```bash
cd ../financial-audit-ui
npm install
npm run dev
```

---

## Đánh giá hệ thống

Hệ thống được đánh giá định lượng bằng framework **RAGAS** trên ba chỉ số:
- **Context Precision** — độ chính xác của ngữ cảnh truy xuất
- **Faithfulness** — độ trung thực của câu trả lời so với văn bản gốc
- **Answer Relevancy** — độ liên quan của câu trả lời so với câu hỏi
- **Context Recall** — độ bao phủ của ngữ cảnh truy xuất

Chi tiết quy trình và kết quả đánh giá nằm trong `evaluate.ipynb` và `ragas_50_result.csv`.

---

## Hạn chế và hướng phát triển

- Hệ thống hiện chưa triển khai cơ chế trích dẫn nguồn có cấu trúc (ví dụ số điều, số trang cụ thể đi kèm câu trả lời).
- Bộ test set đánh giá RAGAS còn ở quy mô nhỏ, cần mở rộng để đánh giá toàn diện hơn trước khi ứng dụng thực tế.
- Phạm vi kho tri thức hiện giới hạn ở các văn bản pháp lý cốt lõi (Thông tư 99/2025/TT-BTC, IAS 1/7, IFRS 15, VSA 240); có thể mở rộng thêm các chuẩn mực khác trong tương lai.

---

## Liên hệ

> Đoàn Tuấn Hùng — doantuanhung1210@gmail.com

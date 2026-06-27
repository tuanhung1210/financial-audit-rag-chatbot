import os
os.environ["HF_HOME"] = r"D:\hf_cache"
os.environ["TRANSFORMERS_CACHE"] = r"D:\hf_cache"

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pymongo import MongoClient
from bson import ObjectId
from typing import Optional, List, Dict, Any
import pypdf
import docx
import pandas as pd
import io
import json
import re
import time
from rank_bm25 import BM25Okapi

from core.config import settings
from services.embedding import get_text_embedding
from services.gemini_service import (
    generate_risk_analysis,
    rewrite_query_with_history,
    is_query_in_scope,
    generate_hyde,
    detect_source,
    rerank,
)
from services.conversation_memory import (
    create_session,
    get_session,
    append_turn,
    build_history_context,
    build_history_context_from_list,
)

# ─────────────────────────────────────────────────────────────
app = FastAPI(title="Hệ thống API rà soát rủi ro báo cáo tài chính")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

mongo_client = MongoClient(settings.MONGO_URI, tls=True, tlsAllowInvalidCertificates=True)
db         = mongo_client[settings.DB_NAME]
collection = db[settings.COLLECTION_NAME]

# ─────────────────────────────────────────────────────────────
#  BM25 — cache corpus trong RAM
# ─────────────────────────────────────────────────────────────
_bm25_index:     Optional[BM25Okapi] = None
_bm25_doc_ids:   List[str]           = []
_bm25_doc_texts: Dict[str, str]      = {}


def _tokenize_vi(text: str) -> List[str]:
    return re.findall(r"[\wÀ-ỹ]+", text.lower(), flags=re.UNICODE)


def build_bm25_corpus() -> None:
    global _bm25_index, _bm25_doc_ids, _bm25_doc_texts

    print("[BM25] Đang tải corpus từ MongoDB...")
    doc_ids:          List[str]       = []
    doc_texts:        Dict[str, str]  = {}
    tokenized_corpus: List[List[str]] = []

    for doc in collection.find({}, {"_id": 1, "text_content": 1}):
        doc_id = str(doc["_id"])
        text   = (doc.get("text_content") or "").strip()
        if not text:
            continue
        doc_ids.append(doc_id)
        doc_texts[doc_id] = text
        tokenized_corpus.append(_tokenize_vi(text))

    _bm25_index    = BM25Okapi(tokenized_corpus) if tokenized_corpus else None
    _bm25_doc_ids  = doc_ids
    _bm25_doc_texts = doc_texts
    print(f"[BM25] Hoàn tất: {len(doc_ids):,} chunk đã lập chỉ mục.")


def bm25_search(query: str, top_k: int = 30, min_score: float = 2.0) -> List[str]:
    """
    min_score nâng từ 1.0 (bản cũ) lên 2.0 để đồng bộ notebook.
    Giảm false-positive cho các câu hỏi ngắn / chung chung.
    """
    if _bm25_index is None or not _bm25_doc_ids:
        return []
    tokens = _tokenize_vi(query)
    if not tokens:
        return []
    scores = _bm25_index.get_scores(tokens)
    ranked = sorted(zip(_bm25_doc_ids, scores), key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, score in ranked[:top_k] if score >= min_score]


# ─────────────────────────────────────────────────────────────
#  RRF — với boost cho detected_sources
# ─────────────────────────────────────────────────────────────
def reciprocal_rank_fusion(
    vector_ranked_ids: List[str],
    bm25_ranked_ids:   List[str],
    k:            int  = 60,
    boost_ids:    Optional[set] = None,
    boost_amount: float = 0.10,
) -> List[str]:
    scores: Dict[str, float] = {}

    for rank, doc_id in enumerate(vector_ranked_ids):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)

    for rank, doc_id in enumerate(bm25_ranked_ids):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)

    if boost_ids:
        for doc_id in scores:
            if doc_id in boost_ids:
                scores[doc_id] += boost_amount

    return [doc_id for doc_id, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


# ─────────────────────────────────────────────────────────────
#  Build BM25 khi khởi động
# ─────────────────────────────────────────────────────────────
build_bm25_corpus()

# ─────────────────────────────────────────────────────────────
#  Pydantic schemas
# ─────────────────────────────────────────────────────────────
class AuditRequest(BaseModel):
    query:      str
    text:       Optional[str]                  = None
    prompt:     Optional[str]                  = None
    history:    Optional[List[Dict[str, Any]]] = []
    session_id: Optional[str]                  = None


# ─────────────────────────────────────────────────────────────
#  RAG PIPELINE (đồng bộ notebook)
#  Thay đổi so với bản cũ:
#    - Bỏ CrossEncoder + detect_category
#    - Thêm HyDE: sinh hypothetical doc → trung bình vector
#    - Thêm detect_source → RRF boost theo source
#    - Thêm BM25 account boost (TK xxx)
#    - Gemini reranker (thay CrossEncoder)
#    - Vector score threshold: 0.65 → 0.50
#    - candidate_limit: limit*4 → max(limit*6, 60)
#    - numCandidates: 200 → 600
#    - Bỏ metadata filter category (notebook không dùng)
# ─────────────────────────────────────────────────────────────
def run_rag_pipeline(user_query: str, limit: int = 10) -> List[str]:
    if len(_tokenize_vi(user_query)) < 3:
        return []

    # ── 1. HyDE ──────────────────────────────────────────────
    t0 = time.time()
    hyde_doc = generate_hyde(user_query)
    print(f"  [RAG] hyde: {time.time()-t0:.1f}s")

    # ── 2. Embedding: trung bình query + HyDE ────────────────
    t1 = time.time()
    query_vector = get_text_embedding(user_query)
    hyde_vector  = get_text_embedding(hyde_doc)
    combined_vector = [(q + h) / 2 for q, h in zip(query_vector, hyde_vector)]
    print(f"  [RAG] embed: {time.time()-t1:.1f}s")

    candidate_limit  = max(limit * 6, 60)
    detected_sources = detect_source(user_query)

    # ── 3. Vector search ─────────────────────────────────────
    pipeline = [
        {
            "$vectorSearch": {
                "index":         settings.VECTOR_INDEX_NAME,
                "path":          "vector_embeddings",
                "queryVector":   combined_vector,
                "numCandidates": 600,
                "limit":         candidate_limit,
            }
        },
        {
            "$project": {
                "_id":          1,
                "text_content": 1,
                "metadata":     1,
                "score":        {"$meta": "vectorSearchScore"},
            }
        },
    ]

    t2 = time.time()
    vector_results = list(collection.aggregate(pipeline))
    print(f"  [RAG] vector search: {time.time()-t2:.1f}s  ({len(vector_results)} docs)")

    chunk_by_id:        Dict[str, dict]  = {}
    vector_score_by_id: Dict[str, float] = {}
    vector_ranked_ids:  List[str]        = []

    for doc in vector_results:
        doc_id = str(doc["_id"])
        chunk_by_id[doc_id]        = doc
        vector_score_by_id[doc_id] = doc.get("score", 0.0)
        vector_ranked_ids.append(doc_id)

    # ── 4. BM25 search ───────────────────────────────────────
    t3 = time.time()
    bm25_ranked_ids = bm25_search(user_query, top_k=candidate_limit)

    # BM25 account boost: nếu câu hỏi nhắc đến TK xxx, tìm thêm
    account_matches = re.findall(r"TK\s*(\d{3,4})", user_query, re.IGNORECASE)
    if account_matches:
        if "511" in account_matches and "131" not in account_matches:
            account_matches.append("131")
        account_query = " ".join(f"TK {m}" for m in account_matches)
        bm25_account_ids = bm25_search(account_query, top_k=30, min_score=0.5)
        if bm25_account_ids:
            print(f"  [RAG] bm25 account boost: {account_query} → {len(bm25_account_ids)} docs")
            bm25_ranked_ids = bm25_account_ids + [
                d for d in bm25_ranked_ids if d not in set(bm25_account_ids)
            ]
    print(f"  [RAG] bm25: {time.time()-t3:.1f}s  ({len(bm25_ranked_ids)} docs)")

    # ── 5. Fetch BM25 docs không có trong vector results ─────
    missing_ids = [d for d in bm25_ranked_ids if d not in chunk_by_id]
    if missing_ids:
        t4 = time.time()
        for doc in collection.find(
            {"_id": {"$in": [ObjectId(m) for m in missing_ids]}},
            {"_id": 1, "text_content": 1, "metadata": 1},
        ):
            doc_id = str(doc["_id"])
            chunk_by_id[doc_id]        = doc
            vector_score_by_id[doc_id] = 0.0
        print(f"  [RAG] fetch missing: {time.time()-t4:.1f}s  ({len(missing_ids)} docs)")

    # ── 6. RRF với boost theo detected_source ────────────────
    boost_ids: Optional[set] = None
    if detected_sources:
        boost_ids = {
            doc_id for doc_id, doc in chunk_by_id.items()
            if doc.get("metadata", {}).get("source") in detected_sources
        }
        print(f"  [RAG] detected_sources={detected_sources}, boost_ids={len(boost_ids or set())}")

    fused_ids = reciprocal_rank_fusion(
        vector_ranked_ids,
        bm25_ranked_ids,
        boost_ids=boost_ids,
        boost_amount=0.10,
    )

    # ── 7. Pre-filter: vector score ≥ 0.50 hoặc xuất hiện trong BM25 ──
    #  (notebook dùng 0.50, bản cũ dùng 0.65)
    bm25_id_set = set(bm25_ranked_ids)
    pre_filter_ids: List[str] = []
    for doc_id in fused_ids:
        if len(pre_filter_ids) >= candidate_limit:
            break
        if not chunk_by_id.get(doc_id):
            continue
        vscore    = vector_score_by_id.get(doc_id, 0.0)
        from_bm25 = doc_id in bm25_id_set
        if vscore >= 0.50 or from_bm25:
            pre_filter_ids.append(doc_id)

    # ── 8. Gemini reranker (thay CrossEncoder) ───────────────
    t5 = time.time()
    reranked_ids = rerank(
        user_query,
        pre_filter_ids,
        chunk_by_id,
        top_n=limit,
        preferred_sources=detected_sources or None,
        min_per_source=2,
    )
    print(f"  [RAG] rerank: {time.time()-t5:.1f}s")

    return [chunk_by_id[doc_id].get("text_content", "") for doc_id in reranked_ids]


# ─────────────────────────────────────────────────────────────
#  ENDPOINTS ADMIN
# ─────────────────────────────────────────────────────────────
@app.post("/api/v1/admin/rebuild-bm25")
def rebuild_bm25_index():
    try:
        build_bm25_corpus()
        return {
            "status":  "success",
            "message": f"Đã xây dựng lại chỉ mục BM25 với {len(_bm25_doc_ids)} chunk.",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi build lại chỉ mục BM25: {str(e)}")


# ─────────────────────────────────────────────────────────────
#  ENDPOINTS PHIÊN HỘI THOẠI
# ─────────────────────────────────────────────────────────────
@app.post("/api/v1/session/create")
def create_chat_session():
    sid = create_session()
    return {"session_id": sid}


@app.get("/api/v1/session/{session_id}")
def get_chat_session(session_id: str):
    doc = get_session(session_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Phiên không tồn tại hoặc đã hết hạn (24h).")
    doc.pop("_id", None)
    return doc


# ─────────────────────────────────────────────────────────────
#  ENDPOINT 1: /api/v1/review — chat văn bản
# ─────────────────────────────────────────────────────────────
@app.post("/api/v1/review")
def review_financial_risk(request: AuditRequest):
    try:
        # Bộ nhớ phân tầng
        if request.session_id and get_session(request.session_id):
            history_context = build_history_context(request.session_id)
        else:
            history_context = build_history_context_from_list(request.history or [])

        # Guard 1 — phạm vi chủ đề
        if not is_query_in_scope(request.query):
            return {
                "status":   "not_found",
                "message":  "Câu hỏi nằm ngoài phạm vi rà soát rủi ro báo cáo tài chính mà hệ thống được thiết lập để hỗ trợ.",
                "analysis": "Vui lòng đặt câu hỏi liên quan đến kế toán, kiểm toán, hoặc rà soát rủi ro báo cáo tài chính.",
            }

        # Guard 2 — độ dài câu hỏi gốc
        if len(_tokenize_vi(request.query)) < 3:
            return {
                "status":   "not_found",
                "message":  "Hệ thống từ chối nhận định do không tìm thấy căn cứ pháp lý tương thích trong kho tri thức.",
                "analysis": "Vui lòng bổ sung thêm dữ liệu phân hệ tài khoản này vào file tri thức gốc.",
            }

        # Query rewriting
        rag_query = rewrite_query_with_history(request.query, history_context)

        valid_contexts = run_rag_pipeline(rag_query)

        if not valid_contexts:
            return {
                "status":   "not_found",
                "message":  "Hệ thống từ chối nhận định do không tìm thấy căn cứ pháp lý tương thích trong kho tri thức.",
                "analysis": "Vui lòng bổ sung thêm dữ liệu phân hệ tài khoản này vào file tri thức gốc.",
            }

        final_prompt = (
            f"{history_context}\nCăn cứ vào lịch sử hội thoại trên, hãy trả lời câu hỏi hiện tại: {request.query}"
            if history_context
            else request.query
        )
        analysis_result = generate_risk_analysis(final_prompt, valid_contexts)

        if request.session_id:
            append_turn(request.session_id, request.query, analysis_result)

        return {
            "status":               "success",
            "extracted_chunks_count": len(valid_contexts),
            "rewritten_query":      rag_query,
            "analysis":             analysis_result,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi trục dịch vụ hệ thống backend: {str(e)}")


# ─────────────────────────────────────────────────────────────
#  ENDPOINT 2: /api/v1/review-file — upload tệp đính kèm
# ─────────────────────────────────────────────────────────────
@app.post("/api/v1/review-file")
async def review_financial_file(
    files:      List[UploadFile] = File(...),
    query:      Optional[str]    = Form(None),
    history:    Optional[str]    = Form(None),
    session_id: Optional[str]    = Form(None),
):
    try:
        filenames_display = ", ".join(f.filename for f in files)
        extracted_text    = ""

        for f in files:
            filename = f.filename
            ext      = filename.split(".")[-1].lower()
            if ext not in ["pdf", "docx", "xlsx", "xls"]:
                raise HTTPException(
                    status_code=400,
                    detail=f"Định dạng tệp '{filename}' không được hỗ trợ!",
                )

            file_content = await f.read()
            file_text    = ""

            if ext == "pdf":
                pdf_reader = pypdf.PdfReader(io.BytesIO(file_content))
                for page in pdf_reader.pages:
                    text = page.extract_text()
                    if text:
                        file_text += text + "\n"

            elif ext == "docx":
                doc_obj = docx.Document(io.BytesIO(file_content))
                for para in doc_obj.paragraphs:
                    if para.text:
                        file_text += para.text + "\n"

            elif ext in ["xlsx", "xls"]:
                excel_data = pd.read_excel(io.BytesIO(file_content), sheet_name=None)
                for sheet_name, df in excel_data.items():
                    file_text += f"\n[Dữ liệu bảng tính - Sheet: {sheet_name}]\n"
                    file_text += df.to_string(index=False) + "\n"

            extracted_text += f"\n\n===== TỆP: {filename} =====\n{file_text}"

        # Giải mã history
        history_list: list = []
        if history:
            try:
                history_list = json.loads(history)
            except Exception:
                pass

        # Bộ nhớ phân tầng
        if session_id and get_session(session_id):
            history_context = build_history_context(session_id)
        else:
            history_context = build_history_context_from_list(history_list)

        effective_query = (
            query if query
            else "Hãy kiểm tra toàn bộ nội dung văn bản trên và chỉ ra các kịch bản rủi ro sai lệch số liệu."
        )

        # Guard 1 — phạm vi chủ đề (xét cả file excerpt)
        if not is_query_in_scope(effective_query, file_excerpt=extracted_text):
            return {
                "status":   "not_found",
                "message":  f"Hệ thống đã đọc xong tệp {filenames_display}, nhưng nội dung tệp và/hoặc câu hỏi nằm ngoài phạm vi rà soát rủi ro báo cáo tài chính.",
                "analysis": "Vui lòng đính kèm tệp liên quan đến kế toán, kiểm toán, hoặc báo cáo tài chính và đặt câu hỏi rà soát phù hợp.",
            }

        # Guard 2 — độ dài (chỉ khi user tự nhập query)
        if query and len(_tokenize_vi(query)) < 3:
            return {
                "status":   "not_found",
                "message":  f"Hệ thống đã đọc xong tệp {filenames_display}, nhưng từ chối nhận định do câu hỏi quá ngắn để xác định ý định rà soát.",
                "analysis": "Vui lòng nhập câu hỏi cụ thể hơn về nội dung cần rà soát trong tệp này.",
            }

        rewritten_query = rewrite_query_with_history(effective_query, history_context)

        # Prompt generate (giữ câu hỏi gốc)
        combined_prompt = ""
        if history_context:
            combined_prompt += history_context + "\n"
        combined_prompt += f"Nội dung văn bản bốc tách từ (các) tệp báo cáo [{filenames_display}]:\n{extracted_text}\n"
        if query:
            combined_prompt += f"Yêu cầu rà soát cụ thể hiện tại của kiểm toán viên: {query}"
        else:
            combined_prompt += "Yêu cầu: Hãy kiểm tra toàn bộ nội dung văn bản trên và chỉ ra các kịch bản rủi ro sai lệch số liệu."

        # Retrieval query: câu hỏi đã viết lại + trích đoạn file
        retrieval_query = f"{rewritten_query}\nTrích đoạn nội dung tệp báo cáo: {extracted_text[:2000]}"

        valid_contexts = run_rag_pipeline(retrieval_query, limit=10)

        if not valid_contexts:
            return {
                "status":   "not_found",
                "message":  f"Hệ thống đã đọc xong tệp {filenames_display}, nhưng từ chối nhận định do không tìm thấy căn cứ luật Thông tư 99 tương thích.",
                "analysis": "Không tìm thấy căn cứ pháp lý phù hợp với nội dung tệp này trong kho dữ liệu vector.",
            }

        analysis_result = generate_risk_analysis(combined_prompt, valid_contexts)

        if session_id:
            user_note = query if query else f"[Tệp đính kèm: {filenames_display}]"
            append_turn(session_id, user_note, analysis_result)

        return {
            "status":               "success",
            "extracted_chunks_count": len(valid_contexts),
            "rewritten_query":      rewritten_query,
            "analysis":             analysis_result,
        }

    except HTTPException as http_ex:
        raise http_ex
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi hệ thống trong quá trình bốc tách dữ liệu tệp: {str(e)}",
        )
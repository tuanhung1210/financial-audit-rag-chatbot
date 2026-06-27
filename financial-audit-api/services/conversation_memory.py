"""
=============================================================
  DỊCH VỤ BỘ NHỚ HỘI THOẠI PHÂN TẦNG (TIERED MEMORY)
  Lưu trữ: MongoDB Atlas (collection `conversation_sessions`)
  Cơ chế:
    - Short-term : N lượt gần nhất giữ nguyên văn (mặc định 8)
    - Long-term  : các lượt cũ hơn được Gemini nén thành
                   rolling summary, cộng dồn dần theo phiên
    - Flagged accounts: danh sách mã TK/rủi ro đã xuất hiện
                        trong phiên, dùng để tăng độ chính xác
                        khi rewrite câu hỏi nối tiếp
=============================================================
"""

from __future__ import annotations

import re
import time
import uuid
from datetime import datetime
from typing import Optional

from pymongo import MongoClient, ASCENDING
from pymongo.collection import Collection

from core.config import settings

# ─── Hằng số ────────────────────────────────────────────────
SHORT_TERM_LIMIT = 8        # số lượt giữ nguyên văn
SESSION_TTL_SECONDS = 86400 # phiên tự xoá sau 24 h (TTL index)

# Pattern nhận diện mã tài khoản kế toán Việt Nam (TK 111, TK 511, ...)
# và một số cụm từ rủi ro đặc thù domain kiểm toán tài chính.
_ACCOUNT_PATTERN = re.compile(
    r'\bTK\s*\d{3,4}\b'            # TK 111, TK 3331, …
    r'|\bThông tư\s*[\d/\-]+\b'    # Thông tư 99/2025/TT-BTC
    r'|\bĐiều\s*\d+\b'             # Điều 15
    r'|\bVAS\s*\d+\b'              # VAS 01, VAS 14, …
    r'|\bIFRS\s*\d+\b',            # IFRS 9, IFRS 15, …
    re.IGNORECASE,
)


# ─── Kết nối tái sử dụng (lazy init) ────────────────────────
_mongo_client: Optional[MongoClient] = None
_col: Optional[Collection] = None


def _get_col() -> Collection:
    global _mongo_client, _col
    if _col is None:
        _mongo_client = MongoClient(
            settings.MONGO_URI,
            tls=True,
            tlsAllowInvalidCertificates=True,
        )
        db = _mongo_client[settings.DB_NAME]
        _col = db["conversation_sessions"]

        # TTL index — MongoDB tự xóa document sau SESSION_TTL_SECONDS giây
        # kể từ thời điểm lưu trường `updated_at`.
        _col.create_index(
            [("updated_at", ASCENDING)],
            expireAfterSeconds=SESSION_TTL_SECONDS,
            background=True,
        )
        # Index hỗ trợ tra cứu nhanh theo session_id
        _col.create_index([("session_id", ASCENDING)], unique=True, background=True)
    return _col


# ─── Tạo / lấy phiên ────────────────────────────────────────

def create_session() -> str:
    """Tạo session_id mới, ghi document rỗng vào MongoDB, trả về session_id."""
    session_id = str(uuid.uuid4())
    _get_col().insert_one({
        "session_id":       session_id,
        "turns":            [],        # lịch sử nguyên văn (short-term)
        "long_term_summary": "",       # rolling summary (long-term)
        "flagged_accounts": [],        # danh sách TK/rủi ro đã xuất hiện
        "created_at":       datetime.utcnow(),
        "updated_at":       datetime.utcnow(),
    })
    return session_id


def get_session(session_id: str) -> Optional[dict]:
    """Trả về document phiên, hoặc None nếu không tồn tại / đã hết TTL."""
    return _get_col().find_one({"session_id": session_id})


# ─── Trích xuất tài khoản / rủi ro từ văn bản ───────────────

def _extract_accounts(text: str) -> list[str]:
    return list({m.group().strip() for m in _ACCOUNT_PATTERN.finditer(text)})


# ─── Nén lịch sử cũ thành summary (gọi Gemini) ──────────────

def _compress_to_summary(old_turns: list[dict], existing_summary: str) -> str:
    """
    Gọi Gemini để cộng dồn: [summary cũ] + [các lượt bị đẩy ra khỏi short-term]
    → summary mới súc tích hơn, không làm mất thông tin quan trọng.

    Fallback: nếu Gemini lỗi, ghép thô thành chuỗi ngắn để không mất dữ liệu.
    """
    # Import lazy để tránh circular import (gemini_service cũng import config)
    from google import genai
    from google.genai import types
    from core.config import settings as cfg

    turns_text = "\n".join(
        f"{'Người dùng' if t['role'] == 'user' else 'Trợ lý'}: {t['content']}"
        for t in old_turns
    )
    prefix = f"Tóm tắt phiên trước đó:\n{existing_summary}\n\n" if existing_summary else ""
    prompt = (
        f"{prefix}"
        f"Các lượt hội thoại mới cần tích hợp vào tóm tắt:\n{turns_text}\n\n"
        "Hãy viết lại TOÀN BỘ thành một đoạn tóm tắt súc tích (tối đa 300 từ), "
        "giữ lại: (1) chủ đề chính đã thảo luận, (2) các mã tài khoản/điều luật "
        "đã được đề cập, (3) kết luận rủi ro quan trọng nếu có. "
        "Không thêm lời chào hay giải thích. Chỉ trả về đoạn tóm tắt thuần túy."
    )

    try:
        client = genai.Client(api_key=cfg.GEMINI_API_KEY)
        resp = client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.0),
        )
        return (resp.text or "").strip()
    except Exception as exc:
        print(f"[TieredMemory] Lỗi nén summary, dùng fallback thô: {exc}")
        # Fallback: ghép thô, cắt bớt nếu quá dài
        fallback = (existing_summary + "\n" + turns_text).strip()
        return fallback[:2000]


# ─── Ghi lượt hội thoại mới ─────────────────────────────────

def append_turn(session_id: str, user_msg: str, assistant_msg: str) -> None:
    """
    Ghi một lượt (user → assistant) vào phiên:
    1. Thêm vào cuối mảng `turns`.
    2. Nếu len(turns) > SHORT_TERM_LIMIT: nén các lượt dư vào `long_term_summary`.
    3. Cập nhật `flagged_accounts` từ nội dung cả lượt mới.
    4. Ghi lại toàn bộ vào MongoDB.
    """
    doc = get_session(session_id)
    if doc is None:
        # Phiên không tồn tại (ví dụ đã hết TTL) — tạo mới tự động
        create_session()
        doc = get_session(session_id)
        # Gán lại session_id đúng (create_session tạo id mới, cần patch lại)
        # → dùng pattern: upsert bên dưới sẽ tạo nếu chưa có.

    turns: list[dict] = doc.get("turns", []) if doc else []
    summary: str = doc.get("long_term_summary", "") if doc else ""
    flagged: list[str] = doc.get("flagged_accounts", []) if doc else []

    # Thêm lượt mới
    turns.append({"role": "user",      "content": user_msg})
    turns.append({"role": "assistant", "content": assistant_msg})

    # Nén nếu vượt ngưỡng short-term (đơn vị: lượt đơn, không phải cặp)
    if len(turns) > SHORT_TERM_LIMIT * 2:
        overflow = turns[:-(SHORT_TERM_LIMIT * 2)]   # các lượt cũ bị đẩy ra
        turns    = turns[-(SHORT_TERM_LIMIT * 2):]   # giữ lại N cặp gần nhất
        summary  = _compress_to_summary(overflow, summary)

    # Cập nhật flagged_accounts
    new_accounts = _extract_accounts(user_msg + " " + assistant_msg)
    flagged = list(set(flagged) | set(new_accounts))

    _get_col().update_one(
        {"session_id": session_id},
        {"$set": {
            "turns":             turns,
            "long_term_summary": summary,
            "flagged_accounts":  flagged,
            "updated_at":        datetime.utcnow(),
        }},
        upsert=True,
    )


# ─── Xây dựng history_context đưa vào prompt ────────────────

def build_history_context(session_id: str) -> str:
    """
    Trả về chuỗi ngữ cảnh tổng hợp đưa vào Prompt:
      [long-term summary nếu có]
      [short-term: N lượt gần nhất nguyên văn]
      [flagged accounts nếu có]

    Trả về chuỗi rỗng nếu phiên không tồn tại hoặc chưa có lịch sử.
    """
    doc = get_session(session_id)
    if doc is None:
        return ""

    parts: list[str] = []

    summary = (doc.get("long_term_summary") or "").strip()
    if summary:
        parts.append(
            "--- TÓM TẮT CÁC NỘI DUNG ĐÃ THẢO LUẬN TRƯỚC ĐÓ TRONG PHIÊN ---\n"
            + summary
        )

    turns: list[dict] = doc.get("turns", [])
    if turns:
        lines = ["--- LỊCH SỬ GẦN NHẤT (NGUYÊN VĂN) ---"]
        for t in turns:
            role_label = "Người dùng" if t["role"] == "user" else "Trợ lý AI Kiểm toán"
            lines.append(f"{role_label}: {t['content']}")
        parts.append("\n".join(lines))

    flagged: list[str] = doc.get("flagged_accounts", [])
    if flagged:
        parts.append(
            "--- TÀI KHOẢN / ĐIỀU LUẬT ĐÃ ĐƯỢC ĐỀ CẬP TRONG PHIÊN ---\n"
            + ", ".join(sorted(flagged))
        )

    if not parts:
        return ""

    return "\n\n".join(parts) + "\n" + "-" * 65 + "\n"


# ─── Alias tiện dụng: build từ frontend history (backward compat) ──

def build_history_context_from_list(history_list: list[dict]) -> str:
    """
    Fallback: nếu frontend vẫn gửi mảng history cũ (không có session_id),
    format đơn giản như cũ để không phá vỡ luồng hiện có.
    Xóa dần khi frontend đã tích hợp session_id.
    """
    if not history_list:
        return ""
    lines = ["--- LỊCH SỬ CUỘC TRÒ CHUYỆN TRƯỚC ĐÓ ---"]
    for msg in history_list[-SHORT_TERM_LIMIT:]:
        role = "Người dùng" if msg.get("sender") == "user" else "Trợ lý AI Kiểm toán"
        lines.append(f"{role}: {msg.get('text', '')}")
    lines.append("-" * 65)
    return "\n".join(lines) + "\n"

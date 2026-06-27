from __future__ import annotations

import json
import re
import time
from typing import Optional

from google import genai
from google.genai import types

from core.config import settings

# ─── Client đơn key chính (dùng cho rewrite, generate, is_in_scope, memory) ──
_client = genai.Client(api_key=settings.GEMINI_API_KEY)

# ─── Models ────────────────────────────────────────────────────────────────────
GENERATE_MODEL = "gemini-3.1-flash-lite"

# ══════════════════════════════════════════════════════════════════════════════
#  KEY POOLS — xoay vòng round-robin độc lập từng tác vụ
#  Thêm / xoá key tại đây, không cần sửa logic bên dưới.
# ══════════════════════════════════════════════════════════════════════════════
HYDE_API_KEYS: list[str] = [
    "AQ.Ab8RN6LRT_QGVYsXSAj9BaCjW54YS9ub_VR3p34ghY8VdiQy5g",
    "AQ.Ab8RN6JP4zjkhftmtPYkfLQmlp69LIFHCKMBm5XMCDwsY4cpoQ",
    "AQ.Ab8RN6JT_w4RGiDqsgobE208-d9lRrWvRwTw1llNb237jwy5gg",
    "AQ.Ab8RN6KHQK2LkDQxzQc5RYq0-Wr1G6YaXU7M45Gnv--UTrV_bw",
    "AQ.Ab8RN6Lctc4p2UQMCtEi0k3p54YFkWVfquME8DcmHEDnuGaLHw",
]

RERANK_API_KEYS: list[str] = [
    "AQ.Ab8RN6LRD1iOnzdL2klPdb3esc29CFEOLQe2ZfEDgMcw1MCYEQ",
    "AQ.Ab8RN6JocQwIhcZoz5dwoCCIIiqf4uJDF7Sb2hZKzzPiu-w9DA",
    "AQ.Ab8RN6IVbvjtAj4LWpzxx_PTdkUCSbCFJNm2awhmA89yqaJSaw",
    "AQ.Ab8RN6Io5g9ZYpYIqhs-ZDt8L6Yzz35tx_N_Uh47L8Nv7uajMQ",
    "AQ.Ab8RN6KBn4dDVQ4zmVP9K3bPkJUnpp9DcGIeIoDTaRz37RMCjg",
    "AQ.Ab8RN6JFfzdgABNA6yUoShe7xQoDnj8MpKBZWwsbbvN8YB4SLg",
    "AQ.Ab8RN6LRTnEABOlKA5ETT2PeAx7B7BHQJZArK2nOFKyTU7TPNQ",
    "AQ.Ab8RN6J9W3sQiMtVdb1QKczrdgekUbGNOm_xoZslLOX_IcNsAg",
    "AQ.Ab8RN6LlauaXivH1nIplYMb2ubH_vLdUP5BgkIjJuyDNl5MIhA",
    "AQ.Ab8RN6KnuRw3YPEYiEUnBvgaIw4PcOR2eXMe4asz7QAQngn8Gw",
]

GENERATE_API_KEYS: list[str] = [
    "AQ.Ab8RN6I_-37jpKS2DsfGal6jT0M_GegJrJwRCTeRRcMX1aeKJQ",
    "AQ.Ab8RN6Ip1qDydTGmpMdDWxYN66CsxK0ywclJmOBK4CjUR3uRnA",
    "AQ.Ab8RN6KY-4BBCq2CtJq-mTe-HCAEOwNYPkLI_5XvHScV_6f0gg",
]

# Bộ đếm xoay vòng độc lập từng pool
_pool_counters: dict[str, int] = {"hyde": 0, "rerank": 0, "generate": 0}


def _get_next_key(pool: str) -> str:
    keys = {"hyde": HYDE_API_KEYS, "rerank": RERANK_API_KEYS, "generate": GENERATE_API_KEYS}[pool]
    idx = _pool_counters[pool] % len(keys)
    _pool_counters[pool] += 1
    return keys[idx]


# ─── Throttle ─────────────────────────────────────────────────────────────────
_MIN_SECONDS_BETWEEN_CALLS = 2.0  # throttle chung (dùng cho single-key calls)
_MIN_SECONDS_PER_POOL: dict[str, float] = {
    "hyde":     5.0,
    "rerank":   8.0,
    "generate": 5.0,
}
_last_call_time: float = 0.0
_last_call_time_per_pool: dict[str, float] = {"hyde": 0.0, "rerank": 0.0, "generate": 0.0}


def _throttle() -> None:
    global _last_call_time
    elapsed = time.time() - _last_call_time
    if elapsed < _MIN_SECONDS_BETWEEN_CALLS:
        time.sleep(_MIN_SECONDS_BETWEEN_CALLS - elapsed)
    _last_call_time = time.time()


def _throttle_pool(pool: str) -> None:
    min_gap = _MIN_SECONDS_PER_POOL.get(pool, 5.0)
    elapsed = time.time() - _last_call_time_per_pool[pool]
    if elapsed < min_gap:
        time.sleep(min_gap - elapsed)
    _last_call_time_per_pool[pool] = time.time()


# ─── Retry helpers ────────────────────────────────────────────────────────────
_RATE_LIMIT_KEYWORDS = ["billing", "quota", "rate", "429", "resource exhausted"]
_RATE_LIMIT_WAIT     = 70
_MAX_RETRIES         = 8


def _is_rate_limit_error(err: Exception) -> bool:
    return any(kw in str(err).lower() for kw in _RATE_LIMIT_KEYWORDS)


def _is_invalid_key_error(err: Exception) -> bool:
    msg = str(err).lower()
    return "api_key_invalid" in msg or "api key not valid" in msg


def _is_server_unavailable_error(err: Exception) -> bool:
    msg = str(err).lower()
    return "503" in msg or "unavailable" in msg


def _call_with_retry(fn, max_retries: int = _MAX_RETRIES,
                     wait_seconds: int = _RATE_LIMIT_WAIT):
    last_exc: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        _throttle()
        try:
            return fn()
        except Exception as e:
            last_exc = e
            if _is_rate_limit_error(e):
                if attempt < max_retries:
                    print(f"  ⏳ Rate limit — đợi {wait_seconds}s (lần {attempt}/{max_retries - 1})...")
                    time.sleep(wait_seconds)
                else:
                    raise
            elif _is_server_unavailable_error(e):
                wait = 10 * attempt
                print(f"  ⏳ Server 503 — đợi {wait}s (lần {attempt}/{max_retries})...")
                time.sleep(wait)
            else:
                raise
    raise last_exc


# ─── Pool caller ──────────────────────────────────────────────────────────────

def _call_gemini_with_pool(pool: str, prompt: str,
                            system: Optional[str] = None,
                            temperature: float = 0.2) -> str:
    """
    Gọi Gemini qua key pool (hyde / rerank / generate), tự động xoay vòng key
    khi gặp key invalid. Có throttle per-pool và retry khi rate-limit / 503.
    """
    all_keys = {"hyde": HYDE_API_KEYS, "rerank": RERANK_API_KEYS, "generate": GENERATE_API_KEYS}[pool]

    for _ in range(len(all_keys)):
        api_key = _get_next_key(pool)
        client  = genai.Client(api_key=api_key)
        cfg     = types.GenerateContentConfig(temperature=temperature)
        if system:
            cfg.system_instruction = system

        def _do():
            _throttle_pool(pool)
            resp = client.models.generate_content(
                model=GENERATE_MODEL,
                contents=prompt,
                config=cfg,
            )
            return resp.text or ""

        try:
            return _call_with_retry(_do)
        except Exception as e:
            if _is_invalid_key_error(e):
                print(f"  ⚠ Key invalid ở pool [{pool}] — thử key tiếp theo...")
                continue
            raise

    raise RuntimeError(f"Tất cả key trong pool [{pool}] đều invalid.")


# ══════════════════════════════════════════════════════════════════════════════
#  SOURCE_KEYWORDS — dùng trong detect_source()
# ══════════════════════════════════════════════════════════════════════════════
SOURCE_KEYWORDS: dict[str, list[str]] = {
    # ── IFRS / IAS ──────────────────────────────────────────────────────────
    "IFRS9": [
        "ifrs 9", "ifrs9", "công cụ tài chính", "phân loại và đo lường",
        "fvtpl", "fvoci", "amortised cost", "giá trị phân bổ",
        "tổn thất tín dụng kỳ vọng", "expected credit loss", "ecl",
    ],
    "IFRS15": [
        "ifrs 15", "ifrs15", "nghĩa vụ thực hiện", "quyền kiểm soát hàng hóa",
        "performance obligation", "variable consideration",
        "giá giao dịch phân bổ", "hợp đồng với khách hàng ifrs",
    ],
    "IAS36": [
        "ias 36", "ias36", "impairment", "suy giảm giá trị",
        "giá trị có thể thu hồi", "value in use", "giá trị sử dụng",
        "đơn vị tạo tiền", "cgu",
    ],
    "IAS1": [
        "ias 1", "ias1", "trình bày báo cáo tài chính ias",
        "other comprehensive income", "oci", "comparative information",
    ],
    "IAS37": [
        "ias 37", "ias37", "dự phòng phải trả ias", "nợ tiềm tàng",
        "contingent liability", "provisions ias", "nghĩa vụ hiện tại",
        "onerous contract",
    ],
    "ISA7": [
        "ias 7", "isa 7", "ias7", "isa7", "cash flow statement ias",
        "cash equivalents", "operating investing financing activities",
        "direct method indirect method cash",
    ],

    # ── VSA ─────────────────────────────────────────────────────────────────
    "VSA315": [
        "vsa 315", "vsa315", "kiểm soát nội bộ",
        "đánh giá rủi ro kiểm toán", "rủi ro có sai sót trọng yếu",
        "môi trường kiểm soát",
    ],
    "VSA240": [
        "vsa 240", "vsa240", "gian lận", "fraud", "red flag gian lận",
        "rủi ro gian lận", "tam giác gian lận", "sai sót cố ý",
        "can thiệp số liệu", "điều chỉnh số liệu", "che giấu sai phạm",
        "override kiểm soát", "management override", "lợi nhuận bất thường",
        "dấu hiệu gian lận tài chính",
    ],
    "VSA330": [
        "vsa 330", "vsa330", "thủ tục kiểm toán", "thử nghiệm cơ bản",
        "thử nghiệm kiểm soát", "trọng yếu thực hiện",
        "mở rộng phạm vi kiểm toán",
    ],

    # ── Văn bản Việt Nam ─────────────────────────────────────────────────────
    "thongtu214": [
        "thông tư 214", "tt214", "thông tư 214/2012",
    ],
    "thongtu99": [
        "thông tư 99", "tt99", "thông tư 99/2025", "tt99/2025",
        "tk 511", "tk511", "tài khoản 511",
        "tk 131", "tk131", "tài khoản 131",
        "tk 632", "tk632", "tài khoản 632",
        "tk 242", "tk242", "tài khoản 242",
        "tk 334", "tk334", "tài khoản 334",
        "tk 331", "tk331", "tài khoản 331",
        "tk 641", "tk641", "tk 642", "tk642",
        "b01-dn", "b09-dn", "b02-dn", "b03-dn",
        "báo cáo tài chính theo thông tư",
        "chế độ kế toán doanh nghiệp",
        "ghi nhận doanh thu thông tư",
        "doanh thu bán hàng tk 511",
        "phân bổ chi phí trả trước",
        "lưu chuyển tiền tệ thông tư",
        "báo cáo lưu chuyển tiền tệ b03",
        "kết quả hoạt động kinh doanh", "kqkd",
        "lợi nhuận tăng dòng tiền âm",
        "chênh lệch lợi nhuận dòng tiền",
        "dòng tiền kinh doanh âm",
        "tiền thu từ khách hàng",
    ],
    "thongtu133": [
        "thông tư 133", "tt133", "thông tư 133/2016",
        "chế độ kế toán doanh nghiệp nhỏ",
        "doanh nghiệp vừa và nhỏ kế toán",
    ],
    "luatketoan2015": [
        "luật kế toán", "luật kế toán 2015", "luật số 88/2015",
        "đơn vị kế toán", "người làm kế toán",
    ],
    "checklistruiro": [
        "checklist rủi ro",
        "cfo/lnst",
        "tỷ lệ chuyển đổi tiền mặt",
        "ngưỡng cảnh báo",
        "kich_ban_kiem_thu",
        "chất lượng lợi nhuận",
        # BỎ "lợi nhuận dương dòng tiền âm" — overlap nhiều câu, đã move sang thongtu99
    ],
}


# ══════════════════════════════════════════════════════════════════════════════
#  detect_source — thay detect_category cũ
# ══════════════════════════════════════════════════════════════════════════════

def detect_source(question: str) -> list[str]:
    """
    Dùng Gemini (qua pool hyde) để xác định tối đa 2 source văn bản
    liên quan nhất với câu hỏi. Trả về list source name (khớp SOURCE_KEYWORDS).
    Fallback trả về [] nếu lỗi.
    """
    source_descriptions = {
        "IFRS9":          "công cụ tài chính, phân loại đo lường, ECL, FVTPL, FVOCI",
        "IFRS15":         "ghi nhận doanh thu theo IFRS, performance obligation, hợp đồng khách hàng quốc tế",
        "IAS36":          "suy giảm giá trị tài sản, impairment, CGU",
        "IAS1":           "trình bày báo cáo tài chính theo IAS, OCI",
        "IAS37":          "dự phòng phải trả IAS, nợ tiềm tàng, contingent liability",
        "ISA7":           "báo cáo lưu chuyển tiền tệ theo chuẩn mực quốc tế IAS 7",
        "VSA315":         "đánh giá rủi ro kiểm toán, kiểm soát nội bộ, môi trường kiểm soát",
        "VSA240":         "gian lận, fraud, can thiệp số liệu, management override, dấu hiệu gian lận",
        "VSA330":         "thủ tục kiểm toán, thử nghiệm cơ bản, thử nghiệm kiểm soát, trọng yếu",
        "thongtu214":     "thông tư 214, chuẩn mực kiểm toán Việt Nam ban hành kèm TT214",
        "thongtu99":      "chế độ kế toán Việt Nam, tài khoản kế toán TK 511 131 632 242, biểu mẫu B01-DN B09-DN, lưu chuyển tiền tệ theo thông tư",
        "thongtu133":     "chế độ kế toán doanh nghiệp nhỏ và vừa, thông tư 133",
        "luatketoan2015": "luật kế toán 2015, đơn vị kế toán, người làm kế toán",
        "checklistruiro": "checklist rủi ro kiểm toán, kịch bản kiểm thử, ngưỡng cảnh báo CFO/LNST, chất lượng lợi nhuận",
    }
    known_sources = list(source_descriptions.keys())
    desc_text = "\n".join(f"- {k}: {v}" for k, v in source_descriptions.items())

    prompt = (
        f"Bạn là chuyên gia kế toán kiểm toán Việt Nam.\n"
        f"Xác định câu hỏi dưới đây liên quan đến nguồn văn bản nào.\n\n"
        f"Câu hỏi: {question}\n\n"
        f"Danh sách nguồn và mô tả:\n{desc_text}\n\n"
        f"Quy tắc:\n"
        f"- Chỉ chọn nguồn có nội dung TRỰC TIẾP giải quyết câu hỏi.\n"
        f"- Câu hỏi về tài khoản kế toán Việt Nam (TK 511, TK 131...) hoặc biểu mẫu "
        f"(B01-DN, B09-DN) → luôn chọn thongtu99.\n"
        f"- Câu hỏi về gian lận, can thiệp số liệu → chọn VSA240.\n"
        f"- Câu hỏi về doanh thu theo IFRS, performance obligation → chọn IFRS15.\n"
        f"- Câu hỏi về doanh thu theo chế độ kế toán Việt Nam → chọn thongtu99, KHÔNG chọn IFRS15.\n"
        f"- Câu hỏi so sánh số liệu giữa các chỉ tiêu BCTC (doanh thu, dòng tiền, phải thu) "
        f"mà KHÔNG nhắc đến gian lận, can thiệp, fraud → KHÔNG chọn VSA240.\n"
        f"- Câu hỏi về lợi nhuận tăng nhưng dòng tiền âm, CFO âm, chất lượng lợi nhuận → chọn checklistruiro và VSA240.\n"
        f"- CHỈ chọn VSA330 nếu câu hỏi hỏi RÕ RÀNG về: thủ tục kiểm toán cần thiết kế, "
        f"thử nghiệm cơ bản/thử nghiệm kiểm soát, mức trọng yếu, hoặc 'kiểm toán viên cần làm "
        f"gì tiếp theo/cần xử lý như thế nào'. KHÔNG chọn VSA330 cho câu hỏi chỉ hỏi về "
        f"NGHIỆP VỤ KẾ TOÁN đơn thuần (tài khoản nào cần dùng, sai sót gì, hệ quả với báo cáo "
        f"tài chính) dù chủ đề đó có thể là đối tượng kiểm toán — nếu câu hỏi không tự nhắc "
        f"đến hành động/vai trò của kiểm toán viên hoặc thủ tục kiểm toán, đừng chọn VSA330.\n"
        f"- Tương tự, KHÔNG chọn VSA315 trừ khi câu hỏi nhắc đến đánh giá rủi ro kiểm toán/môi "
        f"trường kiểm soát; KHÔNG chọn VSA240 trừ khi câu hỏi nhắc gian lận/can thiệp/thao túng.\n"
        f"- Nếu không chắc một nguồn có thực sự cần thiết hay không, ĐỪNG thêm nguồn đó — "
        f"chỉ trả về nguồn nào câu hỏi cần để được giải quyết trọn vẹn, không thêm nguồn 'cho "
        f"chắc' hay vì chủ đề nghe có vẻ liên quan.\n"
        f"- Trả về tối đa 2 nguồn quan trọng nhất.\n\n"
        f"Trả về DUY NHẤT một JSON array. Ví dụ: [\"thongtu99\", \"VSA240\"]. Không giải thích thêm."
    )
    try:
        raw = _call_gemini_with_pool("hyde", prompt, temperature=0.0)
        raw = raw.strip()
        start = raw.find("[")
        end   = raw.rfind("]") + 1
        if start == -1 or end == 0:
            return []
        result = json.loads(raw[start:end])
        return [s for s in result if s in known_sources]
    except Exception as e:
        print(f"[detect_source] Lỗi: {e}")
        return []


# ══════════════════════════════════════════════════════════════════════════════
#  generate_hyde — sinh hypothetical document
# ══════════════════════════════════════════════════════════════════════════════

def generate_hyde(query: str) -> str:
    """
    Sinh một đoạn văn bản pháp lý giả định (~200-300 từ) trả lời trực tiếp
    cho câu hỏi. Vector của đoạn này được trung bình với vector câu hỏi gốc
    để cải thiện vector search (kỹ thuật HyDE).
    """
    system = (
        "Bạn là chuyên gia kế toán kiểm toán Việt Nam. "
        "Hãy viết một đoạn văn bản pháp lý ngắn (200-300 từ) "
        "như thể đây là nội dung từ một chuẩn mực kế toán hoặc kiểm toán "
        "trả lời trực tiếp cho câu hỏi dưới đây. "
        "Chỉ viết đoạn văn, không giải thích thêm."
    )
    try:
        return _call_gemini_with_pool("hyde", query, system=system, temperature=0.3)
    except Exception as e:
        print(f"[generate_hyde] Lỗi: {e} — fallback trả về câu hỏi gốc")
        return query


# ══════════════════════════════════════════════════════════════════════════════
#  rerank — Gemini listwise reranker (thay CrossEncoder cũ)
# ══════════════════════════════════════════════════════════════════════════════

def rerank(
    query: str,
    candidate_ids: list[str],
    chunk_by_id: dict,
    top_n: int = 10,
    preferred_sources: Optional[list[str]] = None,
    min_per_source: int = 2,
) -> list[str]:
    """
    Rerank danh sách candidate_ids bằng Gemini (1 lần call, listwise).
    Trả về top_n doc_id xếp theo thứ tự liên quan giảm dần.
    min_per_source: khi preferred_sources có ≥2 nguồn, đảm bảo MỖI nguồn trong
    preferred_sources giữ tối thiểu min_per_source chunk trong kết quả cuối
    (nếu có đủ candidate thuộc nguồn đó), thay vì để 1 nguồn áp đảo hết top_n.
    Fallback: giữ nguyên thứ tự gốc nếu Gemini lỗi.
    """
    if not candidate_ids:
        return []
    valid_ids = [d for d in candidate_ids if d in chunk_by_id]
    if not valid_ids:
        return []

    chunks_text = ""
    for i, doc_id in enumerate(valid_ids):
        text = chunk_by_id[doc_id].get("text_content", "")[:800]
        chunks_text += f"\n[{i}] {text}\n"

    source_hint = ""
    if preferred_sources:
        source_hint = (
            f" [!] Câu hỏi này liên quan đến nguồn: {preferred_sources}. "
            f"Ưu tiên các đoạn từ nguồn này NẾU nội dung thực sự liên quan. "
            f"KHÔNG ưu tiên chỉ vì tên nguồn — nội dung phải khớp câu hỏi.\n\n"
        )

    prompt = (
        f"Câu hỏi của kiểm toán viên: {query}\n\n"
        f"{source_hint}"
        f"Dưới đây là {len(valid_ids)} đoạn trích từ các chuẩn mực kế toán và kiểm toán, "
        f"được đánh số từ 0 đến {len(valid_ids)-1}:\n"
        f"{chunks_text}\n"
        f"Nhiệm vụ: Xếp hạng các đoạn theo tiêu chí sau:\n"
        f"1. Đoạn nào chứa QUY ĐỊNH, SỐ LIỆU hoặc HƯỚNG DẪN CỤ THỂ về tài khoản kế toán, "
        f"biểu mẫu hoặc nguyên tắc được nhắc ĐÍCH DANH trong câu hỏi — xếp lên đầu. "
        f"Đặc biệt ưu tiên: đoạn chứa BÚT TOÁN KẾ TOÁN (Nợ TK..., Có TK...) liên quan "
        f"đến tài khoản được nhắc trong câu hỏi phải xếp CAO hơn đoạn chỉ nói về khái "
        f"niệm hay định nghĩa chung.\n"
        f"2. Nếu câu hỏi NHẮC TÊN một chuẩn mực kiểm toán/kế toán cụ thể (VSA240, VSA330, "
        f"VSA315, IAS1, IAS36, IAS37, IFRS9, IFRS15...) — dù là nhắc kèm câu hỏi nghiệp vụ "
        f"khác — đoạn trích TỪ ĐÚNG chuẩn mực đó PHẢI được xếp vào nhóm liên quan, không bị "
        f"loại chỉ vì có đoạn nghiệp vụ kế toán khác trùng từ khóa nhiều hơn. Câu hỏi hỏi "
        f"'kiểm toán viên cần làm gì' hoặc trích dẫn ≥2 chuẩn mực luôn được coi là hỏi TRỰC "
        f"TIẾP về cả các chuẩn mực đó.\n"
        f"3. Đoạn chỉ chứa từ khóa bề mặt giống câu hỏi nhưng không có quy định giải quyết "
        f"được vấn đề cụ thể — xếp xuống cuối.\n"
        f"Trả về DUY NHẤT một JSON array chứa các index theo thứ tự từ liên quan "
        f"nhất đến ít nhất. Ví dụ: [3, 0, 7, 1, ...]. Không giải thích thêm."
    )

    try:
        raw = _call_gemini_with_pool("rerank", prompt, temperature=0.0)
        raw = raw.strip()
        start = raw.find("[")
        end   = raw.rfind("]") + 1
        if start == -1 or end == 0:
            raise ValueError("Không tìm thấy JSON array trong response reranker")
        ranked_indices = json.loads(raw[start:end])

        ranked_ids: list[str] = []
        seen: set[str] = set()
        for idx in ranked_indices:
            if isinstance(idx, int) and 0 <= idx < len(valid_ids):
                doc_id = valid_ids[idx]
                if doc_id not in seen:
                    ranked_ids.append(doc_id)
                    seen.add(doc_id)
        # Bổ sung các id bị bỏ sót (reranker không trả về đủ)
        for doc_id in valid_ids:
            if doc_id not in seen:
                ranked_ids.append(doc_id)

        # Đảm bảo đa dạng nguồn: nếu có ≥2 preferred_sources, mỗi nguồn
        # giữ tối thiểu min_per_source chunk trong top_n (round-robin),
        # tránh 1 nguồn áp đảo hết slot dù model rerank đã ưu tiên đúng.
        if preferred_sources and len(set(preferred_sources)) >= 2:
            def _src(doc_id):
                return chunk_by_id[doc_id].get("metadata", {}).get("source")

            result = []
            reserved_quota = {s: min_per_source for s in set(preferred_sources)}
            remaining = list(ranked_ids)

            # Bước 1: round-robin lấy đủ quota cho từng preferred source theo
            # đúng thứ tự rerank đã xếp (không random, vẫn ưu tiên chunk tốt nhất của mỗi nguồn)
            for s in preferred_sources:
                if reserved_quota.get(s, 0) <= 0:
                    continue
                taken = 0
                still_remaining = []
                for doc_id in remaining:
                    if taken < reserved_quota[s] and _src(doc_id) == s:
                        result.append(doc_id)
                        taken += 1
                    else:
                        still_remaining.append(doc_id)
                remaining = still_remaining
                reserved_quota[s] = 0  # đã xử lý nguồn này

            # Bước 2: lấp đầy phần còn lại của top_n theo đúng thứ tự rerank gốc
            for doc_id in remaining:
                if len(result) >= top_n:
                    break
                result.append(doc_id)

            result = result[:top_n]
        else:
            result = ranked_ids[:top_n]

        debug_sources = [
            chunk_by_id[doc_id].get("metadata", {}).get("source")
            for doc_id in result
        ]
        print(f"  → sources sau rerank: {debug_sources}")
        return result

    except Exception as e:
        print(f"  ⚠ Reranker lỗi ({e}) — fallback giữ thứ tự gốc")
        return valid_ids[:top_n]


# ══════════════════════════════════════════════════════════════════════════════
#  rewrite_query_with_history — giữ nguyên, dùng key chính
# ══════════════════════════════════════════════════════════════════════════════

def rewrite_query_with_history(current_query: str, history_context: str) -> str:
    """
    Viết lại câu hỏi hiện tại thành câu hỏi độc lập, đầy đủ ngữ cảnh,
    dựa trên lịch sử hội thoại. Fallback trả về câu hỏi gốc nếu lỗi.
    """
    if not history_context:
        return current_query

    system_instruction = (
        "Bạn là một bộ viết lại truy vấn (Query Rewriter) chuyên dùng cho hệ thống RAG rà soát rủi ro báo cáo tài chính.\n"
        "Nhiệm vụ DUY NHẤT của bạn: dựa vào lịch sử hội thoại được cung cấp, viết lại câu hỏi hiện tại của người dùng "
        "thành MỘT câu hỏi độc lập, đầy đủ ngữ cảnh, có thể hiểu được mà không cần đọc lại lịch sử.\n\n"
        "Quy tắc:\n"
        "1. Nếu câu hỏi hiện tại đã đầy đủ ngữ cảnh, không phụ thuộc vào lịch sử (ví dụ đã nêu rõ chủ thể, đối tượng), "
        "hãy giữ nguyên hoặc chỉ chỉnh sửa nhẹ, không suy diễn thêm.\n"
        "2. Nếu câu hỏi hiện tại mơ hồ, thiếu chủ thể, dùng đại từ thay thế ('cái đó', 'ngành đó', 'thế còn... thì sao') "
        "thì PHẢI thay thế bằng thông tin cụ thể được suy ra từ lịch sử hội thoại.\n"
        "3. KHÔNG trả lời câu hỏi, KHÔNG thêm giải thích, KHÔNG thêm lời chào hay dẫn nhập.\n"
        "4. CHỈ trả về duy nhất một câu hỏi đã được viết lại, không có ký tự hay định dạng nào khác."
    )

    prompt = f"{history_context}\nCâu hỏi hiện tại của người dùng cần viết lại: {current_query}"

    try:
        response = _client.models.generate_content(
            model=GENERATE_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.0,
            ),
        )
        rewritten = (response.text or "").strip()
        return rewritten if rewritten else current_query
    except Exception:
        return current_query


# ══════════════════════════════════════════════════════════════════════════════
#  generate_risk_analysis — giữ nguyên, dùng key chính
# ══════════════════════════════════════════════════════════════════════════════

def generate_risk_analysis(query: str, contexts: list[str]) -> str:
    """
    Sinh câu trả lời phân tích rủi ro dựa trên câu hỏi và các đoạn context
    đã được retrieve. Dùng key chính (single-key) — không cần pool vì đây là
    tác vụ generate cuối, ít gọi hơn và có thể dùng quota key chính.
    """
    system_instruction = (
        "Bạn là một chuyên gia rà soát rủi ro báo cáo tài chính cao cấp, am hiểu các văn bản pháp luật như Thông tư 99.\n"
        "Nhiệm vụ của bạn là đối chiếu câu hỏi/tài liệu do người dùng cung cấp với các phân đoạn văn bản pháp lý trong bối cảnh.\n\n"
        "Yêu cầu xử lý:\n"
        "1. Hãy phân tích, suy luận và chỉ ra các điểm sai lệch số liệu, kịch bản rủi ro hoặc hướng dẫn nghiệp vụ dựa trên các nguyên tắc có trong bối cảnh.\n"
        "2. Định dạng câu trả lời rõ ràng bằng Markdown (sử dụng các đầu dòng, bôi đậm các tài khoản kế toán hoặc điều luật liên quan).\n"
        "3. Nếu bối cảnh hoàn toàn lạc đề và không có một chút liên quan nào đến câu hỏi, hãy trả lời: "
        "'Hệ thống chưa tìm thấy phân đoạn luật khớp hoàn toàn, dưới đây là phân tích sơ bộ dựa trên dữ liệu hiện tại...'\n"
        "4. LUÔN diễn giải lại bằng lời của bạn, KHÔNG trích dẫn nguyên văn liên tục từ bối cảnh — "
        "tóm tắt và paraphrase nội dung quy định, chỉ giữ số hiệu điều luật/tài khoản, không chép lại cả câu dài.\n"
        "TUYỆT ĐỐI không tự bịa ra các thông tư hoặc điều luật không tồn tại trong bối cảnh."
    )

    combined_context = "\n\n".join(contexts)
    prompt = (
        f"Bối cảnh văn bản pháp lý luật định:\n{combined_context}\n\n"
        f"Câu hỏi/Tình huống cần rà soát hoặc Nội dung tệp: {query}"
    )

    response = _client.models.generate_content(
        model=GENERATE_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.2,
        ),
    )
    return response.text


# ══════════════════════════════════════════════════════════════════════════════
#  is_query_in_scope — giữ nguyên, dùng key chính
# ══════════════════════════════════════════════════════════════════════════════

def is_query_in_scope(query: str, file_excerpt: Optional[str] = None) -> bool:
    """
    Phân loại nhanh: câu hỏi và/hoặc tệp đính kèm có thuộc phạm vi nghiệp vụ
    kế toán/kiểm toán/rủi ro BCTC không.
    Trả về True nếu trong phạm vi HOẶC lỗi API (fallback cho qua).
    Trả về False CHỈ KHI Gemini phân loại rõ ràng là "KHONG".
    """
    if file_excerpt:
        content_block = (
            f'Câu hỏi của người dùng: "{query}"\n\n'
            f"Trích đoạn nội dung tệp đính kèm mà người dùng muốn rà soát:\n"
            f"\"\"\"\n{file_excerpt[:1500]}\n\"\"\"\n\n"
            f"Lưu ý: câu hỏi có thể rất chung chung (ví dụ \"xem có lỗi gì không\", "
            f"\"đánh giá chi tiết\") - trong trường hợp đó, hãy phân loại dựa trên "
            f"NỘI DUNG TỆP ĐÍNH KÈM là chính."
        )
    else:
        content_block = f'Câu hỏi: "{query}"'

    classification_prompt = (
        f"Bạn là bộ phân loại chủ đề. Nhiệm vụ DUY NHẤT: xác định nội dung dưới đây có thuộc phạm vi "
        f"nghiệp vụ kế toán / kiểm toán / rà soát rủi ro báo cáo tài chính / Thông tư 99/2025/TT-BTC / "
        f"chuẩn mực VAS, IFRS hay không.\n\n"
        f"{content_block}\n\n"
        f"Trả lời CHÍNH XÁC một từ duy nhất, không giải thích, không thêm ký tự nào khác:\n"
        f"- \"CO\" nếu thuộc phạm vi kế toán/kiểm toán/rủi ro BCTC.\n"
        f"- \"KHONG\" nếu là giao tiếp xã hội, chào hỏi, hỏi về thời tiết, hoặc nội dung/tệp "
        f"hoàn toàn ngoài phạm vi tài chính/kế toán."
    )

    try:
        response = _client.models.generate_content(
            model=GENERATE_MODEL,
            contents=classification_prompt,
            config=types.GenerateContentConfig(temperature=0),
        )
        result_text = (response.text or "").strip().upper()
        return "KHONG" not in result_text
    except Exception as e:
        print(f"[is_query_in_scope] Lỗi khi gọi Gemini phân loại, fallback CHO QUA: {e}")
        return True
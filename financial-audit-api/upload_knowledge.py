"""
=============================================================
  CÔNG CỤ NẠP TRI THỨC LÊN MONGODB ATLAS
  Hỗ trợ: .docx | .pdf | .txt | URL web
=============================================================
Cài đặt (chạy 1 lần):
    pip install python-docx pdfplumber requests beautifulsoup4 langchain-text-splitters pymongo numpy
=============================================================
"""

import os
import hashlib
import numpy as np
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymongo import MongoClient
from core.config import settings
from services.embedding import get_text_embedding


# ─────────────────────────────────────────────────────────────
# CÁC PHÂN HỆ (CATEGORY) CÓ SẴN
# ─────────────────────────────────────────────────────────────
CATEGORIES = {
    "1": "phap_ly_goc",
    "2": "chuan_muc_kiem_toan",
    "3": "chuan_muc_quoc_te",
    "4": "kich_ban_kiem_thu",
    "5": "huong_dan_nghiep_vu",
}


# ─────────────────────────────────────────────────────────────
# ĐỌC NỘI DUNG THEO ĐỊNH DẠNG FILE
# ─────────────────────────────────────────────────────────────
def read_file_content(file_path: str) -> str:
    ext = Path(file_path).suffix.lower()

    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    elif ext in (".doc", ".docx"):
        from docx import Document
        W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
        doc = Document(file_path)
        parts = []

        for block in doc.element.body:
            tag = block.tag.split("}")[-1] if "}" in block.tag else block.tag

            # Đoạn văn thường
            if tag == "p":
                text = "".join(
                    node.text or "" for node in block.iter(f"{{{W}}}t")
                ).strip()
                if text:
                    parts.append(text)

            # Bảng → giữ cấu trúc hàng | cột
            elif tag == "tbl":
                rows = block.findall(f".//{{{W}}}tr")
                table_lines = []
                for row in rows:
                    cells = row.findall(f".//{{{W}}}tc")
                    cell_texts = [
                        "".join(n.text or "" for n in cell.iter(f"{{{W}}}t")).strip()
                        for cell in cells
                    ]
                    if any(cell_texts):
                        table_lines.append(" | ".join(cell_texts))
                if table_lines:
                    parts.append("\n".join(table_lines))

        return "\n\n".join(parts)

    elif ext == ".pdf":
        import pdfplumber
        pages = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                parts = []

                # Text thường
                text = page.extract_text()
                if text:
                    parts.append(text.strip())

                # Bảng → giữ cấu trúc hàng | cột
                tables = page.extract_tables()
                for table in tables:
                    for row in table:
                        row_text = " | ".join(
                            (cell.strip().replace("\n", " ") if cell else "")
                            for cell in row
                        )
                        if row_text.strip(" |"):
                            parts.append(row_text)

                if parts:
                    pages.append("\n\n".join(parts))
        return "\n\n".join(pages)

    else:
        raise ValueError(f"Định dạng không hỗ trợ: '{ext}'  (chỉ nhận .txt / .docx / .pdf)")


# ─────────────────────────────────────────────────────────────
# CRAWL NỘI DUNG TỪ URL
# ─────────────────────────────────────────────────────────────
def read_from_url(url: str) -> str:
    import requests
    from bs4 import BeautifulSoup

    print(f"   Đang tải trang: {url}")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    res = requests.get(url, headers=headers, timeout=20)
    res.encoding = "utf-8"

    soup = BeautifulSoup(res.text, "html.parser")

    # Xóa các thẻ thừa
    for tag in soup(["script", "style", "nav", "header", "footer", "aside",
                     "form", "button", "iframe", "noscript"]):
        tag.decompose()

    # Thử lấy vùng nội dung chính theo thứ tự ưu tiên
    main = (
        soup.find("div", class_="content1")           # thuvienphapluat.vn
        or soup.find("div", class_="entry-content")   # ifrs.vn, kreston.vn
        or soup.find("div", class_="post-content")
        or soup.find("div", class_="article-content")
        or soup.find("article")
        or soup.find("main")
        or soup.body
    )

    raw = main.get_text(separator="\n")
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    return "\n\n".join(lines)


# ─────────────────────────────────────────────────────────────
# HỘP THOẠI CHỌN FILE (tkinter)
# ─────────────────────────────────────────────────────────────
def pick_files() -> list:
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)

        paths = filedialog.askopenfilenames(
            title="Chọn tài liệu (có thể chọn nhiều file cùng lúc)",
            filetypes=[
                ("Tài liệu được hỗ trợ", "*.txt *.docx *.doc *.pdf"),
                ("Word", "*.docx *.doc"),
                ("PDF",  "*.pdf"),
                ("Text", "*.txt"),
                ("Tất cả", "*.*"),
            ]
        )
        root.destroy()
        return list(paths)

    except Exception:
        print("Không mở được hộp thoại — nhập đường dẫn thủ công.")
        paths = []
        print("Nhập đường dẫn file (Enter trống để kết thúc):")
        while True:
            p = input("  File: ").strip().strip('"').strip("'")
            if not p:
                break
            paths.append(p)
        return paths


# ─────────────────────────────────────────────────────────────
# CHỌN CATEGORY
# ─────────────────────────────────────────────────────────────
def pick_category() -> str:
    print("\n  Chọn phân hệ (category):")
    for k, v in CATEGORIES.items():
        print(f"    [{k}] {v}")
    print("    [0] Nhập tên mới")

    while True:
        c = input("  -> ").strip()
        if c in CATEGORIES:
            return CATEGORIES[c]
        if c == "0":
            name = input("  Tên category mới (không dấu, dùng _): ").strip()
            if name:
                return name
        print("  Vui lòng chọn lại.")


# ─────────────────────────────────────────────────────────────
# UPLOAD MỘT NGUỒN DỮ LIỆU
# ─────────────────────────────────────────────────────────────
def upload_knowledge(text_content: str, category_name: str, source_name: str):
    client = MongoClient(settings.MONGO_URI, tls=True, tlsAllowInvalidCertificates=True)
    db     = client[settings.DB_NAME]
    col    = db[settings.COLLECTION_NAME]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        separators=["---", "\n\n", "\n", ". "]
    )
    chunks = splitter.split_text(text_content)
    print(f"   + {len(chunks)} doan van ban")

    existing_ids = set(
        doc["chunk_id"] for doc in col.find(
            {"metadata.category": category_name},
            {"chunk_id": 1}
        )
    )

    batch    = []
    inserted = 0
    skipped  = 0

    print("   Dang nhung vector va day len Atlas...")
    for chunk in chunks:
        clean = chunk.strip().replace("---", "")
        if len(clean) < 10:
            continue

        chunk_id = f"{category_name}_{hashlib.md5(clean.encode()).hexdigest()[:12]}"
        if chunk_id in existing_ids:
            skipped += 1
            continue

        vector = get_text_embedding(clean)
        if vector is None:
            continue

        vector = np.array(vector, dtype=np.float32).tolist()

        batch.append({
            "chunk_id":          chunk_id,
            "text_content":      clean,
            "vector_embeddings": vector,
            "metadata": {
                "category": category_name,
                "source":   source_name,
                "lang":     "vi",
            }
        })

        if len(batch) >= 50:
            col.insert_many(batch)
            inserted += len(batch)
            batch = []
            print(f"     -> {inserted} chunks da len cloud...")

    if batch:
        col.insert_many(batch)
        inserted += len(batch)

    client.close()
    print(f"   + Them moi: {inserted}  |  Bo qua trung: {skipped}")
    return inserted, skipped


# ─────────────────────────────────────────────────────────────
# LUONG CHINH
# ─────────────────────────────────────────────────────────────
def main():
    print()
    print("=" * 58)
    print("    NAP TRI THUC")
    print("=" * 58)

    print("""
Nguon du lieu:
  [1] Chon file tu may tinh  (.docx / .pdf / .txt)
  [2] Nhap URL trang web     (tu dong crawl noi dung)
  [3] Ca hai
""")
    source_mode = input("-> Lua chon (1/2/3): ").strip()

    all_sources = []

    if source_mode in ("1", "3"):
        print("\nMo hop thoai chon file...")
        file_paths = pick_files()
        if file_paths:
            print(f"\n+ Da chon {len(file_paths)} file:")
            for p in file_paths:
                print(f"   * {Path(p).name}")
                all_sources.append({"type": "file", "value": p, "label": Path(p).name})
        else:
            print("  (Khong co file nao duoc chon)")

    if source_mode in ("2", "3"):
        print("\nNhap URL (Enter trong de ket thuc):")
        while True:
            url = input("  URL: ").strip()
            if not url:
                break
            if not url.startswith("http"):
                url = "https://" + url
            all_sources.append({"type": "url", "value": url, "label": url})

    if not all_sources:
        print("\nKhong co nguon nao. Thoat.")
        return

    print(f"""
Co {len(all_sources)} nguon duoc chon.
Ban muon:
  [1] Dung CHUNG mot category cho tat ca
  [2] Chon RIENG category cho tung nguon
""")
    cat_mode = input("-> Lua chon (1/2): ").strip()

    shared_category = None
    if cat_mode == "1":
        shared_category = pick_category()

    summary = []
    print()

    for src in all_sources:
        print(f"{'=' * 58}")
        print(f"NGUON: {src['label']}")

        category = shared_category if shared_category else pick_category()

        default_source = Path(src["value"]).stem if src["type"] == "file" else src["value"]
        source_input   = input(f"  Ten nguon (Enter = '{default_source[:50]}'): ").strip()
        source_name    = source_input if source_input else default_source

        try:
            if src["type"] == "file":
                print(f"   Dang doc file...")
                text = read_file_content(src["value"])
            else:
                text = read_from_url(src["value"])

            print(f"   + {len(text):,} ky tu")
            ins, skip = upload_knowledge(text, category, source_name)
            summary.append({"label": src["label"], "category": category,
                            "source": source_name, "inserted": ins,
                            "skipped": skip, "status": "OK"})

        except Exception as e:
            print(f"   LOI: {e}")
            summary.append({"label": src["label"], "category": "-",
                            "source": "-", "inserted": 0,
                            "skipped": 0, "status": "LOI"})

    print(f"\n{'=' * 58}")
    print("TONG KET:")
    total = 0
    for s in summary:
        print(f"  [{s['status']}] {s['label']}")
        print(f"       category : {s['category']}")
        print(f"       source   : {s['source']}")
        print(f"       da nap   : {s['inserted']} chunks  |  bo qua: {s['skipped']}")
        total += s["inserted"]
    print(f"\n  Tong chunks moi len cloud: {total}")
    print("=" * 58)


if __name__ == "__main__":
    main()
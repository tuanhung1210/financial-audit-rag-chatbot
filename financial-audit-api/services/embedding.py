from sentence_transformers import SentenceTransformer
import os

# Xác định đường dẫn tuyệt đối để lưu mô hình nhúng vào thư mục venv ở ổ D
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
cache_dir = os.path.join(project_root, 'venv', 'models')

# Khởi tạo mô hình nhúng bản địa và cấu hình thư mục lưu cache tại ổ D
embedding_model = SentenceTransformer('keepitreal/vietnamese-sbert', cache_folder=cache_dir)

def get_text_embedding(text: str) -> list[float]:
    """
    Chuyển đổi chuỗi chữ văn bản thành mảng vector 768 chiều (chạy Offline)
    """
    embedding_result = embedding_model.encode(text, normalize_embeddings=True)
    return [float(value) for value in embedding_result]
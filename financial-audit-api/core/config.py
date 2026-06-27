import os

class Settings:
    # Chuỗi kết nối và thông số cấu hình trích xuất chuẩn xác từ Ô code 1 và Ô code 5 của bạn
    MONGO_URI: str = "mongodb+srv://doantuanhung1210:zrSevAdGTbcyQucN@cluster0.s1gtpct.mongodb.net/?appName=Cluster0"
    DB_NAME: str = "do_an_chatbot_bctc"
    COLLECTION_NAME: str = "financial_rules_vector"
    VECTOR_INDEX_NAME: str = "default"
    
    # Khóa kết nối API Google Gemini AI (Sử dụng cho bản 3.1 Flash-Lite)
    GEMINI_API_KEY: str = "AQ.Ab8RN6I97kFA0PPnfxtjxvLcCiJUtBzR2SGnegevgjMp8YKi9Q"

settings = Settings()
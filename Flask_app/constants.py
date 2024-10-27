import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

UPLOAD_FOLDER = 'uploads'
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB limit
SECRET_KEY = os.getenv('SECRET_KEY')
MONGO_CONN_URL = os.getenv("MONGO_CONN_URL")
FAISS_INDEX_PATH = './FAISS_Index/faiss_index.index'
MAX_QUERIES_BEFORE_CLEAR = 10
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")

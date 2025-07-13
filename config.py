import os
import datetime
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Load .env file first
load_dotenv()

# Macros/defaults
DEFAULT_LOGS_DIR = "logs"
DEFAULT_LOG_FILE_PREFIX = "chat_logs_"
DEFAULT_LOG_FILE_EXT = ".txt"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 100

class Settings(BaseSettings):
    """Manages application settings and secrets using Pydantic for validation."""
    use_local_llm: bool = os.getenv("USE_LOCAL_LLM", "false").lower() == "true"
    hf_token: str = os.getenv("HF_TOKEN", "")
    model_id: str = os.getenv("MODEL_ID", "meta-llama/llama-3-8b-instruct")
    remote_api_url: str = os.getenv("REMOTE_API_URL", "https://router.huggingface.co/novita/v3/openai/chat/completions")
    local_api_url: str = os.getenv("LOCAL_API_URL", "http://localhost:11434/api/chat")
    qdrant_url: str = os.getenv("QDRANT_URL", "http://qdrant:6333")
    qdrant_collection_name: str = os.getenv("QDRANT_COLLECTION_NAME", "wallet_vectors")
    embed_model: str = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    embed_dimension: int = int(os.getenv("EMBED_DIMENSION", "384"))
    data_path: str = os.getenv("DATA_PATH", "data/")
    logs_dir: str = os.getenv("LOGS_DIR", DEFAULT_LOGS_DIR)
    log_file_prefix: str = os.getenv("LOG_FILE_PREFIX", DEFAULT_LOG_FILE_PREFIX)
    log_file_ext: str = os.getenv("LOG_FILE_EXT", DEFAULT_LOG_FILE_EXT)
    chunk_size: int = int(os.getenv("CHUNK_SIZE", str(DEFAULT_CHUNK_SIZE)))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", str(DEFAULT_CHUNK_OVERLAP)))
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

def get_log_filename():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    settings = Settings()
    os.makedirs(settings.logs_dir, exist_ok=True)
    return os.path.join(settings.logs_dir, f"{settings.log_file_prefix}{timestamp}{settings.log_file_ext}")

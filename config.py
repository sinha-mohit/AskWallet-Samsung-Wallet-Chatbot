import os
import datetime
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Load .env file first
load_dotenv()

# Macros/defaults
DEFAULT_USE_LOCAL_LLM = "true"
DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_QDRANT_COLLECTION_NAME = "wallet_vectors"
DEFAULT_MODEL_ID = "meta-llama/llama-3-8b-instruct"
DEFAULT_REMOTE_API_URL = "https://router.huggingface.co/novita/v3/openai/chat/completions"
DEFAULT_LOCAL_API_URL = "http://localhost:11434/api/chat"
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EMBED_DIMENSION = 384
DEFAULT_DATA_PATH = "data/"
DEFAULT_LOGS_DIR = "logs"
DEFAULT_LOG_FILE_PREFIX = "chat_logs_"
DEFAULT_LOG_FILE_EXT = ".txt"
DEFAULT_CHUNK_SIZE = 2500
DEFAULT_CHUNK_OVERLAP = 250

class Settings(BaseSettings):
    """Manages application settings and secrets using Pydantic for validation."""
    
    # --- ENVIRONMENT VARIABLE DEFAULTS ---
    # USE_LOCAL_LLM: Use local LLM if true (default: true)
    # QDRANT_URL: Vector DB URL (default: http://localhost:6333)
    # QDRANT_COLLECTION_NAME: Qdrant collection name (default: wallet_vectors)
    # MODEL_ID: LLM model ID (default: meta-llama/llama-3-8b-instruct)
    # REMOTE_API_URL: Remote LLM API URL (default: https://router.huggingface.co/novita/v3/openai/chat/completions)
    # LOCAL_API_URL: Local LLM API URL (default: http://localhost:11434/api/chat)
    # EMBED_MODEL: Embedding model (default: sentence-transformers/all-MiniLM-L6-v2)
    # EMBED_DIMENSION: Embedding dimension (default: 384)
    # DATA_PATH: Path to data directory (default: data/)
    # LOGS_DIR: Directory for logs (default: logs)
    # LOG_FILE_PREFIX: Log file prefix (default: chat_logs_)
    # LOG_FILE_EXT: Log file extension (default: .txt)

    use_local_llm: bool = os.getenv("USE_LOCAL_LLM", DEFAULT_USE_LOCAL_LLM).strip().lower() == "true"
    qdrant_url: str = os.getenv("QDRANT_URL", DEFAULT_QDRANT_URL)
    qdrant_collection_name: str = os.getenv("QDRANT_COLLECTION_NAME", DEFAULT_QDRANT_COLLECTION_NAME)
    model_id: str = os.getenv("MODEL_ID", DEFAULT_MODEL_ID)
    remote_api_url: str = os.getenv("REMOTE_API_URL", DEFAULT_REMOTE_API_URL)
    local_api_url: str = os.getenv("LOCAL_API_URL", DEFAULT_LOCAL_API_URL)
    embed_model: str = os.getenv("EMBED_MODEL", DEFAULT_EMBED_MODEL)
    embed_dimension: int = int(os.getenv("EMBED_DIMENSION", str(DEFAULT_EMBED_DIMENSION)))
    data_path: str = os.getenv("DATA_PATH", DEFAULT_DATA_PATH)
    logs_dir: str = os.getenv("LOGS_DIR", DEFAULT_LOGS_DIR)
    log_file_prefix: str = os.getenv("LOG_FILE_PREFIX", DEFAULT_LOG_FILE_PREFIX)
    log_file_ext: str = os.getenv("LOG_FILE_EXT", DEFAULT_LOG_FILE_EXT)
    chunk_size: int = int(os.getenv("CHUNK_SIZE", str(DEFAULT_CHUNK_SIZE)))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", str(DEFAULT_CHUNK_OVERLAP)))
    hf_token: str = os.getenv("HF_TOKEN", "")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    def get_llm_client(self, use_local):
        if use_local:
            from llm_client.local import LocalLLMClient
            return LocalLLMClient(self.model_id, self.local_api_url)
        from llm_client.remote import RemoteHuggingFaceClient
        return RemoteHuggingFaceClient(self.model_id, self.remote_api_url, self.hf_token)

def get_log_filename():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    settings = Settings()
    os.makedirs(settings.logs_dir, exist_ok=True)
    return os.path.join(settings.logs_dir, f"{settings.log_file_prefix}{timestamp}{settings.log_file_ext}")

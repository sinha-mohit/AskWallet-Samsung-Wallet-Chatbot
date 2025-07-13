import logging
from logging_utils import log_section
from llm_client.remote import RemoteHuggingFaceClient
from llm_client.local import LocalLLMClient
from config import Settings

def summarize_history(messages: list) -> str:
    settings = Settings()
    history = [m for m in messages if m["role"] in ("user", "assistant")]
    if not history:
        return ""
    raw_summary = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in history])
    llm = LocalLLMClient(settings.model_id, settings.local_api_url) if settings.use_local_llm else RemoteHuggingFaceClient(settings.model_id, settings.remote_api_url, settings.hf_token)
    summary_prompt = [
        {"role": "system", "content": "You are a helpful assistant. Summarize the following conversation history for context retention. Be concise and preserve important details."},
        {"role": "user", "content": raw_summary}
    ]
    try:
        summary = llm.generate(summary_prompt)
        log_section("SUMMARY", f"History summarized:\n{summary}")
        return f"Summary of previous conversation:\n{summary}"
    except Exception as e:
        logging.error(f"Summarization failed: {e}")
        return f"Summary of previous conversation:\n{raw_summary}"

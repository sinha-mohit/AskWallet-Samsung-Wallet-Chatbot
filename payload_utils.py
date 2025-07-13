"""
payload_utils.py
Utility for building efficient, clear LLM payloads for chat completion.
"""
from typing import List, Dict
from logging_utils import log_section

def build_llm_payload(
    user_question: str,
    context: str,
    history: List[Dict],
    instructions: str = "You are an AI assistant. Use only the CONTEXT to answer. If not answerable, reply 'I don't know.'",
    log_file: str = None
) -> List[Dict]:
    """
    Build a clear, efficient LLM payload for chat completion.
    - history: list of dicts [{'role': 'user', 'content': ...}, {'role': 'assistant', 'content': ...}]
    - context: concise, relevant context string
    - user_question: latest user question
    Removes duplicate user messages and only includes the latest user message and recent assistant responses.
    Logs key steps if log_file is provided.
    """
    # Remove duplicate user messages, keep only the latest user message
    filtered_history = []
    seen_user = False
    for msg in reversed(history):
        if msg["role"] == "user":
            if not seen_user:
                filtered_history.insert(0, msg)
                seen_user = True
            # Skip older user messages
        elif msg["role"] == "assistant":
            filtered_history.insert(0, msg)
    if log_file:
        log_section("HISTORY FILTER", f"Filtered history: {filtered_history}")
    # Limit to last 2 exchanges (user+assistant)
    trimmed_history = filtered_history[-4:] if len(filtered_history) > 4 else filtered_history

    log_section("HISTORY TRIM", f"Trimmed history: {trimmed_history}")

    messages = [
        {"role": "system", "content": instructions},
        {"role": "context", "content": f"CONTEXT:\n{context.strip()}"}
    ]
    messages.extend(trimmed_history)
    # Ensure only the latest user message is present
    if not trimmed_history or trimmed_history[-1]["role"] != "user":
        messages.append({"role": "user", "content": user_question})

    log_section("PAYLOAD BUILD", f"Final payload: {messages}")

    return messages

"""
llm_orchestrator.py
Handles payload building, LLM invocation, and response post-processing.
"""
from payload_utils import build_llm_payload
from context_filter import filter_context
from summarization import summarize_history
from logging_utils import log_section

MEMORY_WINDOW = 3
SUMMARY_TRIGGER = 6

class LLMOrchestrator:
    def __init__(self, settings, log_file, SYSTEM_PROMT, llm_client):
        self.settings = settings
        self.log_file = log_file
        self.system_prompt = SYSTEM_PROMT
        self.llm_client = llm_client

    def build_payload(self, user_prompt, context, history):
        summary_msg = None
        if len(history) > SUMMARY_TRIGGER:
            summary = summarize_history(history[:-MEMORY_WINDOW])
            summary_msg = {"role": "system", "content": summary}
            log_section("SUMMARY ADDED", f"History summarized:\n{summary}")
        filtered_context = filter_context(context, user_prompt)
        if not filtered_context:
            log_section("FALLBACK", "Fallback response used due to empty or irrelevant context.")
        payload = build_llm_payload(
            user_question=user_prompt,
            context=filtered_context,
            history=history,
            log_file=self.log_file,
            instructions=self.system_prompt
        )
        if summary_msg:
            log_section("SUMMARY", f"Adding summary to payload: {summary_msg}")
            payload.insert(1, summary_msg)
        log_section("PAYLOAD", f"LLM payload messages:\n{payload}")
        return payload

    def get_response(self, payload_msg):
        answer = self.llm_client.generate(payload_msg)
        log_section("LLM RESPONSE", f"Generated answer:\n{answer}")
        return answer

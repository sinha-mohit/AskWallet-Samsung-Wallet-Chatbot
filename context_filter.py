from logging_utils import log_section

"""
context_filter.py
Provides context relevance checking and filtering utilities.
"""
def is_context_relevant(context: str, user_prompt: str) -> bool:
    """Simple keyword-based relevance check."""
    user_keywords = set(user_prompt.lower().split())
    context_lower = context.lower()
    return any(kw in context_lower for kw in user_keywords)

def filter_context(context: str, user_prompt: str) -> str:
    if not is_context_relevant(context, user_prompt):
        log_section("CONTEXT FILTER", "Context deemed not relevant to user prompt; using empty context.")
        return ""
    return context

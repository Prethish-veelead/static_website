from typing import Any, Dict, List


def format_context(retrieved_chunks: List[Dict[str, Any]]) -> str:
    """Format URL context with source labels."""
    context_parts = []
    for chunk in retrieved_chunks:
        text = chunk.get("text", "")
        source = chunk.get("source", "Unknown source")
        context_parts.append(f"[Source: {source}]\n{text}")

    return "\n\n".join(context_parts)


def format_history(history: List[Dict[str, str]]) -> str:
    """Format conversation history."""
    if not history:
        return ""

    history_parts = []
    for msg in history[-8:]:
        if isinstance(msg, dict):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
        else:
            role = getattr(msg, "role", "unknown")
            content = getattr(msg, "content", "")

        role = str(role).upper()
        history_parts.append(f"{role}: {content}")

    return "\n".join(history_parts)


def build_prompt(
    question: str,
    retrieved_chunks: List[Dict[str, Any]],
    history: List[Dict[str, str]],
) -> str:
    """Build complete prompt with URL context, history, and question."""
    context = format_context(retrieved_chunks)
    history_text = format_history(history)

    return f"""Context extracted from configured nutrition web sources:
{context}

Conversation history:
{history_text}

Question: {question}"""

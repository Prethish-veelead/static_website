import logging
import os

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a helpful nutrition assistant. Answer ONLY based on the web context provided.
If the answer is not in the context, respond with: 'I could not find this in the configured nutrition sources.'
Never make up answers. Cite the source URL when you use it."""


def generate_answer(prompt: str) -> str:
    """Generate an answer using the configured LLM provider."""
    provider = os.getenv("LLM_PROVIDER", "anthropic").lower()

    if provider == "anthropic":
        return _generate_with_anthropic(prompt)
    if provider == "azure_openai":
        return _generate_with_azure_openai(prompt)

    raise ValueError(f"Unknown LLM provider: {provider}")


# def _generate_with_anthropic(prompt: str) -> str:
#     """Generate answer using Anthropic Claude."""
#     try:
#         from anthropic import Anthropic

#         api_key = os.getenv("ANTHROPIC_API_KEY")
#         if not api_key:
#             raise ValueError("ANTHROPIC_API_KEY is not set.")

#         client = Anthropic(api_key=api_key)
#         message = client.messages.create(
#             model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
#             max_tokens=1024,
#             temperature=0,
#             system=SYSTEM_PROMPT,
#             messages=[{"role": "user", "content": prompt}],
#         )

#         answer = message.content[0].text
#         logger.info("Generated answer using Anthropic Claude")
#         return answer
#     except Exception as exc:
#         logger.error("Error generating answer with Anthropic: %s", exc)
#         raise


def _generate_with_azure_openai(prompt: str) -> str:
    """Generate answer using Azure OpenAI via the OpenAI Python SDK."""
    try:
        from openai import AzureOpenAI

        api_key = os.getenv("AZURE_OPENAI_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4.1")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")

        missing = [
            name
            for name, value in (
                ("AZURE_OPENAI_KEY", api_key),
                ("AZURE_OPENAI_ENDPOINT", endpoint),
                ("AZURE_OPENAI_CHAT_DEPLOYMENT", deployment),
            )
            if not value
        ]
        if missing:
            raise ValueError(f"Missing Azure OpenAI configuration: {', '.join(missing)}")

        client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint,
        )

        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_completion_tokens=1024,
        )

        answer = response.choices[0].message.content
        if not answer:
            raise ValueError("Azure OpenAI returned an empty response.")

        logger.info("Generated answer using Azure OpenAI deployment '%s'", deployment)
        return answer
    except Exception as exc:
        logger.error("Error generating answer with Azure OpenAI: %s", exc)
        raise

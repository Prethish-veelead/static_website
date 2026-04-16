import os
from pathlib import Path

from dotenv import load_dotenv
from openai import AzureOpenAI


def main() -> None:
    env_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(env_path)

    api_key = os.getenv("AZURE_OPENAI_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "https://newaichat.openai.azure.com/")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
    deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4.1")

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
        raise ValueError(
            f"Missing Azure OpenAI configuration: {', '.join(missing)}. "
            f"Create {env_path} and add the missing values."
        )

    client = AzureOpenAI(
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version=api_version,
    )

    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Reply with only: Azure OpenAI is working."},
        ],
        temperature=0,
        max_completion_tokens=64,
    )

    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()

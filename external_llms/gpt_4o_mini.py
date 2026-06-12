import json
import os
import time
import threading
import requests
import urllib3
import pandas as pd
from typing import List, Dict
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure via environment variables, e.g.:
#   export AZURE_OPENAI_ENDPOINT="https://<resource>.openai.azure.com/openai/deployments/gpt-4o-mini/chat/completions?api-version=2024-06-01"
#   export AZURE_OPENAI_API_KEY="<your-api-key>"


def get_llm_completion(
    messages,
    temperature=0,
    max_tokens=2400,
    top_p=0,
    response_format_json=True,
):
    url = os.environ["AZURE_OPENAI_ENDPOINT"]
    api_key = os.environ["AZURE_OPENAI_API_KEY"]

    headers = {
        'Content-Type': 'application/json',
        'api-key': api_key
    }

    data = {
        'messages': messages,
        'max_tokens': max_tokens,
        'temperature': temperature,
        'top_p': top_p,
        'seed': 42
    }
    if response_format_json:
        data['response_format'] = {'type': 'json_object'}

    data = str.encode(json.dumps(data))

    output = requests.post(
            url=url,
            data=data,
            headers=headers,
            verify=False
    )

    if output.status_code == 200:
        return output.json()['choices'][0]['message']['content']
    elif output.status_code == 400:
        print(output.json())
        return ""
    else:
        print(f"HTTP {output.status_code}: {output.text[:200]}")
        return ""


if __name__ == "__main__":
    print(get_llm_completion(
        messages=[
            {
                "role": "system",
                "content": "I'm an AI agent."
            },
            {
                "role": "user",
                "content": "What color is the sky? Answer in JSON."
            }
        ]
    ))
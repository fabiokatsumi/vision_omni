import requests
from .utils import is_image_path, encode_image

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def run_openrouter_interleaved(messages: list, system: str, model_name: str, api_key: str, max_tokens=256, temperature=0):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    final_messages = [{"role": "system", "content": system}]

    if isinstance(messages, list):
        for item in messages:
            contents = []
            if isinstance(item, dict):
                for cnt in item["content"]:
                    if isinstance(cnt, str):
                        if is_image_path(cnt):
                            base64_image = encode_image(cnt)
                            content = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        else:
                            content = {"type": "text", "text": cnt}
                    else:
                        content = {"type": "text", "text": str(cnt)}
                    contents.append(content)
                message = {"role": "user", "content": contents}
            else:
                contents.append({"type": "text", "text": item})
                message = {"role": "user", "content": contents}
            final_messages.append(message)
    elif isinstance(messages, str):
        final_messages = [{"role": "user", "content": messages}]

    payload = {
        "model": model_name,
        "messages": final_messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    response = requests.post(f"{OPENROUTER_BASE_URL}/chat/completions", headers=headers, json=payload, timeout=120)

    try:
        data = response.json()
        text = data['choices'][0]['message']['content']
        token_usage = int(data.get('usage', {}).get('total_tokens', 0))
        return text, token_usage
    except Exception as e:
        print(f"OpenRouter error: {e}. Response: {response.json()}")
        return str(response.json()), 0

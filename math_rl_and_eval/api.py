import openai
import os
import json
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_current_model():
    try:
        with open("models/model_registry.json", "r") as f:
            return json.load(f)["latest_model"]
    except:
        return os.getenv("BASE_MODEL", "gpt-3.5-turbo")

def get_chat_response(prompt):
    model = get_current_model()
    messages = [{"role": "user", "content": prompt}]
    response = openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2
    )
    return response.choices[0].message.content

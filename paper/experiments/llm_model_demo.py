# %%
import os

from dotenv import load_dotenv
from google import genai
from openai import OpenAI

load_dotenv()


# %%
# Use Hugging Face inference provider API to access models hosted on Hugging Face
hf_router_client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"],
)

completion = hf_router_client.chat.completions.create(
    model="openai/gpt-oss-20b:groq",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    temperature=0.1,
)

print(completion.choices[0].message)
# Example output:
# ChatCompletionMessage(content='The capital of France is **Paris**.', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None, reasoning='We need to answer: "What is the capital of France?" The answer: Paris. Should we provide additional info? The user likely wants the capital. So answer succinctly: Paris.')

# %%
hf_router_client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"],
)

completion = hf_router_client.chat.completions.create(
    model="meta-llama/Llama-3.3-70B-Instruct:groq",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    temperature=0.1,
)

print(completion.choices[0].message.content)
# Example output:
# ChatCompletionMessage(content='The capital of France is Paris.', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None)


# %%

google_genai_client = genai.Client()

generate_content_config = genai.types.GenerateContentConfig(
    temperature=0.1,
)
response = google_genai_client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Explain how AI works in a few words",
    config=generate_content_config,
)

print(response.text)
# Example output:
# AI learns from data to find patterns and make decisions or predictions.


# %%

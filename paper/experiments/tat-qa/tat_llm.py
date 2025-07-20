# %%
# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="next-tat/tat-llm-7b-fft")

pipe("Hello, how to use tat-llm?", max_new_tokens=200)

# %%

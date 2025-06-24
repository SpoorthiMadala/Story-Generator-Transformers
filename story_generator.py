import os
os.environ["TRANSFORMERS_NO_TF"] = "1"  # Disable TensorFlow backend

from transformers import pipeline, set_seed

# Load model only once
generator = pipeline('text-generation', model='gpt2-medium')  # You can try other models too
set_seed(42)

def generate_story(prompt: str, max_length: int = 200) -> str:
    result = generator(prompt, max_length=max_length, num_return_sequences=1)
    return result[0]['generated_text']

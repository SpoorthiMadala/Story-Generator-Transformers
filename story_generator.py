from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "EleutherAI/gpt-neo-1.3B"


tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

def generate_story(prompt, max_tokens=300):

    full_prompt = f"Write a creative short story based on this idea:\n{prompt}\n"

   
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)


    output = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=0.9,
        top_p=0.95,
        repetition_penalty=1.2,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )


    result = tokenizer.decode(output[0], skip_special_tokens=True)
    return result.replace(full_prompt, "").strip()

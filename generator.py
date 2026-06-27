from transformers import pipeline

def generate_reply(prompt, model_name='gpt2-medium', device=0):
    generator = pipeline("text-generation", model=model_name, device=device)
    reply = generator(prompt, max_new_tokens=180, truncation=True)[0]['generated_text']
    # Always split after the last occurrence of "Assistant:"
    return reply.split('Assistant:')[-1].strip()

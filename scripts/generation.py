import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load the tokenizer and model
tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')

# Generate text using the fine-tuned model
def generate_text(prompt, max_length=200, num_return_sequences=1, temperature=0.7, top_k=50, top_p=0.9):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
    output = model.generate(
        input_ids,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        no_repeat_ngram_size=2,
        num_beams=5,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        early_stopping=True
    )

    return tokenizer.decode(output[0], skip_special_tokens=True)

# Example generation
prompt = "What are the signs and symptoms of Lymphocytic Choriomeningitis?"
generated_text = generate_text(prompt)
print(generated_text)

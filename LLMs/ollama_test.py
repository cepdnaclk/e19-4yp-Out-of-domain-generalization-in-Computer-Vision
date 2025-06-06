import ollama

# Define your conversation messages
messages = [
    {'role': 'user',
        'content': 'Give 50 textual descriptions pairs of visual discriminative features to identify whether the central region of an histopathological image patch contains tumor tissue or not. The patch is extracted from an H&E‑stained whole‑slide image of a lymph node section. Only give the output as python code in the format - prompts: list[tuple[negative: str, positive: str]]'},
]

# Get a response from the model
response = ollama.chat(
    model='hf.co/unsloth/medgemma-27b-text-it-GGUF:Q4_K_M', messages=messages)

# Print the model's reply
print(response['message']['content'])

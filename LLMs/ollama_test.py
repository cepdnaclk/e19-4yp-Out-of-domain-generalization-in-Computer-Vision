import ollama

# Define your conversation messages
messages = [
    {'role': 'system', 'content': 'You are a helpful assistant.'},
    {'role': 'user', 'content': 'Why is the sky blue?'},
]

# Get a response from the model
response = ollama.chat(model='deepseek-r1:14b', messages=messages)

# Print the model's reply
print(response['message']['content'])

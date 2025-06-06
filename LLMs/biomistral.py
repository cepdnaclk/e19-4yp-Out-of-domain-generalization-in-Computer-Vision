from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("BioMistral/BioMistral-7B")
model = AutoModel.from_pretrained("BioMistral/BioMistral-7B")

# ask the model to generate a text
input_text = "What is the function of the p53 protein in human cells?"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model(**inputs)
# decode the output
output_text = tokenizer.decode(outputs.logits.argmax(dim=-1)[0])
print(f"Input: {input_text}")
print(f"Output: {output_text}")

# Note: The above code is a simple example of how to use the BioMistral model.

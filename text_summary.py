from transformers import BartTokenizer, BartForConditionalGeneration

# Load the pre-trained model and tokenizer
model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# Example text to summarize
text = " Inside Out 2 picks up with Riley in her early teenage years, a period fraught with new challenges and emotional upheavals. Adolescence is a critical developmental stage characterized by rapid physical, emotional, and psychological changes. This sequel aims to capture the essence of this transformative period, delving deeper into Riley's mind as she faces the multifaceted experiences of growing up."

# Tokenize the input text
inputs = tokenizer([text], max_length=1024, return_tensors="pt", truncation=True)

# Generate the summary
summary_ids = model.generate(inputs["input_ids"], num_beams=4, length_penalty=2.0, max_length=150, min_length=30, no_repeat_ngram_size=2)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Print the summary
print("Original Text:\n", text)
print("\nSummary:\n", summary)

#using BART abstractive summarization.

from transformers import BartForConditionalGeneration, BartTokenizer

# Load pre-trained BART model and tokenizer
model_name = "facebook/bart-large-cnn"
model = BartForConditionalGeneration.from_pretrained(model_name)
tokenizer = BartTokenizer.from_pretrained(model_name)

# Input text for summarization
text = """
Natural Language Processing (NLP) is a branch of artificial intelligence (AI) that focuses on the interaction between computers and human languages. 
Its primary goal is to enable machines to understand, interpret, and generate human language in a way that is both meaningful and useful.
 NLP combines linguistics and machine learning techniques to process and analyzes large amounts of natural language data. 
 Applications of NLP are vast and include sentiment analysis, language translation, chatbots, speech recognition, and text summarization.
NLP techniques help machines break down and understand human language by analyzing structure (syntax), meaning (semantics), and context. 
Key components of NLP include tokenization (breaking text into words or phrases), part-of-speech tagging (identifying word types like nouns or verbs), 
named entity recognition (extracting specific information such as names or dates), and dependency parsing (understanding the relationships between words in a sentence).
 Recent advancements in NLP are largely driven by deep learning and transformer models, such as OpenAI’s GPT and Google’s BERT,
   which have revolutionized how machines generate and comprehend language. These models leverage vast amounts of text data and computational power to
	 achieve impressive language generation capabilities, making NLP a powerful tool in various industries, including healthcare, customer service, and finance."""

# Tokenize the input text
inputs = tokenizer.encode("summarize: " + text, 
                          return_tensors="pt", 
                          max_length=1024, 
                          truncation=True)

# Generate summary
summary_ids = model.generate(inputs, 
                             max_length=130, 
                             min_length=30, 
                             length_penalty=2.0, 
                             num_beams=4, 
                             early_stopping=True)

# Decode and print the summary
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print("Summary:", summary)

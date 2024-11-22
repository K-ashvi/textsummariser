#using library : extractive summarisation
#tf idf
import nltk 
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize, sent_tokenize 

# Input text - to summarize 
text = """Natural Language Processing (NLP) is a branch of artificial intelligence (AI) that focuses on the interaction between computers and human languages. 
Its primary goal is to enable machines to understand, interpret, and generate human language in a way that is both meaningful and useful.
 NLP combines linguistics and machine learning techniques to process and analyze large amounts of natural language data. 
 Applications of NLP are vast and include sentiment analysis, language translation, chatbots, speech recognition, and text summarization.
NLP techniques help machines break down and understand human language by analyzing structure (syntax), meaning (semantics), and context. 
Key components of NLP include tokenization (breaking text into words or phrases), part-of-speech tagging (identifying word types like nouns or verbs), 
named entity recognition (extracting specific information such as names or dates), and dependency parsing (understanding the relationships between words in a sentence).
 Recent advancements in NLP are largely driven by deep learning and transformer models, such as OpenAI’s GPT and Google’s BERT,
   which have revolutionized how machines generate and comprehend language. These models leverage vast amounts of text data and computational power to
	 achieve impressive language generation capabilities, making NLP a powerful tool in various industries, including healthcare, customer service, and finance."""

#tokenise
stopWords = set(stopwords.words("english")) 
words = word_tokenize(text) 

#frequency table 

freqTable = dict() 
for word in words: 
	word = word.lower() 
	if word in stopWords: 
		continue
	if word in freqTable: 
		freqTable[word] += 1
	else: 
		freqTable[word] = 1

# Creating a dictionary 

sentences = sent_tokenize(text) 
sentenceValue = dict() 

for sentence in sentences: 
	for word, freq in freqTable.items(): 
		if word in sentence.lower(): 
			if sentence in sentenceValue: 
				sentenceValue[sentence] += freq 
			else: 
				sentenceValue[sentence] = freq 
sumValues = 0
for sentence in sentenceValue: 
	sumValues += sentenceValue[sentence] 

# Average value of a sentence from the given text 
average = int(sumValues / len(sentenceValue)) 

summary = '' 
for sentence in sentences: 
	if (sentence in sentenceValue) and (sentenceValue[sentence] > (1.2 * average)): 
		summary += " " + sentence 
print(summary) 

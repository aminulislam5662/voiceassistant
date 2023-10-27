import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import networkx as nx
import numpy as np
import re
# Download the stopwords dataset if you haven't already
nltk.download("stopwords")

# # Your input text
# input_text = """
# Muhammad was the prophet and founder of Islam. Most of his early life was spent as a merchant. At age 40, he began to have revelations from Allah that became the basis for the Koran and the foundation of Islam. By 630 he had unified most of Arabia under a single religion. As of 2015, there are over 1.8 billion Muslims in the world who profess, “There is no God but Allah, and Muhammad is his prophet.”Muhammad was a prophet and founder of Islam.After the conflict with Mecca was finally settled, Muhammad took his first true Islamic pilgrimage to that city and in March, 632, he delivered his last sermon at Mount Arafat.
# """

# Define a function to preprocess the text
def preprocess_text(text):
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    words = [word.lower() for word in words if word.isalnum()]
    return sentences, words

# Define a function to calculate sentence similarity based on cosine distance
def sentence_similarity(sent1, sent2):
    stopwords = set(nltk.corpus.stopwords.words("english"))
    sent1 = [word for word in sent1 if word not in stopwords]
    sent2 = [word for word in sent2 if word not in stopwords]
    return nltk.jaccard_distance(set(sent1), set(sent2))

# Define a function to generate the summary
# Define a function to generate the summary
# Define a function to generate the summary
def generate_summary(text, num_sentences=5):
    sentences, _ = preprocess_text(text)
    stop_words = set(stopwords.words("english"))
    sentence_similarity_matrix = np.zeros((len(sentences), len(sentences)))

    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                sentence_similarity_matrix[i][j] = sentence_similarity(sentences[i], sentences[j])

    nx_graph = nx.from_numpy_array(np.array(sentence_similarity_matrix))
    scores = nx.pagerank(nx_graph)

    # Ensure that num_sentences does not exceed the number of available sentences
    num_sentences = min(num_sentences, len(sentences))

    ranked_sentences = sorted(((scores[i], sentence) for i, sentence in enumerate(sentences)), reverse=True)
    summary = " ".join([ranked_sentences[i][1] for i in range(num_sentences)])
    summary = re.sub(r'[\n\t\r]', ' ', summary)  # Replace newline, tabs, and carriage returns with spaces
    summary = ''.join(char for char in summary if ord(char) < 128)  # Remove non-ASCII characters

    return summary

# # Generate an extractive summary
# extractive_summary = generate_summary(input_text, num_sentences=3)

# # Print the extractive summary
# print("Extractive Summary:")
# print(extractive_summary)

# Generate a summary
# summary = generate_summary(input_text)

# Print the summary
# print(summary)

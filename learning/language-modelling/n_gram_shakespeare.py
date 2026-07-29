from collections import Counter
import random
import re
from urllib.request import Request, urlopen

SOURCE_URL = "https://www.gutenberg.org/files/100/100-0.txt"

request = Request(
    SOURCE_URL,
    headers={"User-Agent": "Mozilla/5.0"}
)

with urlopen(request, timeout=60) as response:
    raw_text = response.read().decode("utf-8")

# Remove Project Gutenberg's header and footer.
start = re.search(
    r"\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK .*?\*\*\*",
    raw_text,
    flags=re.IGNORECASE | re.DOTALL,
)
end = re.search(
    r"\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK .*?\*\*\*",
    raw_text,
    flags=re.IGNORECASE | re.DOTALL,
)

if start and end:
    shakespeare_text = raw_text[start.end():end.start()]
else:
    shakespeare_text = raw_text

# Remove stage directions such as [Enter HAMLET] and clean whitespace.
shakespeare_text = re.sub(r"\[[^\]]*\]", " ", shakespeare_text)
shakespeare_text = re.sub(r"\s+", " ", shakespeare_text)

# Split the complete collection into sentence-like training examples.
corpus = [
    sentence.strip()
    for sentence in re.split(r"[.!?]+", shakespeare_text)
    if len(sentence.split()) >= 3
]

corpus = [sentence.lower() for sentence in corpus]
corpus = [sentence.split() for sentence in corpus]
corpus = [["<start>"] + sentence + ["<end>"] for sentence in corpus]

flat_tokens = [token for sentence in corpus for token in sentence]
token_counts = [[token, count] for token, count in Counter(flat_tokens).items()]
token_counts = Counter(flat_tokens)

bigrams = [
    (sentence[i], sentence[i + 1])
    for sentence in corpus
    for i in range(len(sentence) - 1)
]

bigram_counts = [[list(pair), count] for pair, count in Counter(bigrams).items()]
bigram_counts = Counter(bigrams)

token_counts = Counter(flat_tokens)
bigram_counts = Counter(bigrams)

def prob(previous_token, next_token):
    denominator = token_counts[previous_token]

    if denominator == 0:
        return 0.0

    numerator = bigram_counts[(previous_token, next_token)]
    return numerator / denominator

def sentence_prob(sentence_tokens):
    total_prob = 1.0

    for i in range(len(sentence_tokens) - 1):
        prev_token = sentence_tokens[i]
        next_token = sentence_tokens[i + 1]

        p = prob(prev_token, next_token)
        total_prob *= p

        if total_prob == 0.0:
            return 0.0

    return total_prob

V = len(token_counts)

def prob_smoothed(previous_token, next_token):
    numerator = bigram_counts[(previous_token, next_token)] + 1
    denominator = token_counts[previous_token] + V

    return numerator / denominator

def sentence_prob_smoothed(sentence_tokens):
    total_prob = 1.0

    for i in range(len(sentence_tokens) - 1):
        prev_token = sentence_tokens[i]
        next_token = sentence_tokens[i + 1]

        p = prob_smoothed(prev_token, next_token)
        total_prob *= p

    return total_prob

unseen_sentence = ["<start>", "the", "world", "protest", "<end>"]

def generate_sentence_unsmoothed(max_length=20):
    current_token = "<start>"
    generated_sequence = [current_token]

    vocab = [t for t in token_counts.keys() if t != "<start>"]

    for _ in range(max_length):
        probabilities = [prob(current_token, next_token) for next_token in vocab]

        if sum(probabilities) == 0:
            break

        next_token = random.choices(vocab, weights=probabilities, k=1)[0]
        generated_sequence.append(next_token)

        if next_token == "<end>":
            break

        current_token = next_token

    words = [t for t in generated_sequence if t not in ("<start>", "<end>")]
    return " ".join(words)

random.seed(42)

print("Generated Sentence:")
print(generate_sentence_unsmoothed())
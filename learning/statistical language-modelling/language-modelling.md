# Language Modelling

## 1. Introduction

**Language modelling** is the task of assigning probabilities to sequences of tokens or predicting the probability distribution of the next token given the previous tokens.

A language model may estimate the probability of an entire sequence:

```text
P(w1, w2, ..., wT)
```

Alternatively, it may predict the next token:

```text
P(wt | w1, w2, ..., w(t-1))
```

---

## 2. Natural Language Processing

**Natural Language Processing (NLP)** is the field concerned with enabling computers to process, analyse, and generate human language.

Two major areas of NLP are:

1. **Natural Language Understanding (NLU)**  
   Concerned with interpreting meaning, intent, relationships, and context.

2. **Natural Language Generation (NLG)**  
   Concerned with generating coherent and grammatically correct language.

---

## 3. Training a Language Model

Language models are trained using large amounts of text data.

During training, the model learns patterns such as:

- which tokens commonly appear together,
- grammatical structures,
- relationships between words,
- contextual patterns,
- and common sequences of language.

The model can then generate new text by predicting a probability distribution over possible next tokens.

### Example

Given the context:

> I am going to the ...

The model may assign the following probabilities:

| Possible next token | Probability |
|---|---:|
| `store` | 0.40 |
| `park` | 0.30 |
| `beach` | 0.20 |
| `movies` | 0.10 |

The probabilities satisfy:

```text
0.40 + 0.30 + 0.20 + 0.10 = 1
```

Using greedy decoding, the highest-probability token is selected:

> I am going to the store.

---

## 4. Model Computation and Decoding

For fixed parameters and the same input, the model's computation is generally deterministic: it produces the same output probability distribution.

However, the next token does not always have to be the token with the highest probability.

The method used to select the next token is called a **decoding strategy**.

### 4.1 Greedy Decoding

Greedy decoding always selects the token with the highest probability.

```text
wt = arg max P(w | context)
     w in V
```

This is deterministic but may produce repetitive or less varied text.

### 4.2 Sampling

Sampling selects a token randomly according to the probability distribution produced by the model.

Tokens with higher probabilities are more likely to be selected, but lower-probability tokens can also be chosen.

### 4.3 Temperature Sampling

Temperature modifies how sharp or flat the probability distribution is.

- A **lower temperature** makes the distribution sharper and more predictable.
- A **higher temperature** makes the distribution flatter and more varied.

### 4.4 Top-k Sampling

Top-k sampling restricts selection to the `k` tokens with the highest probabilities.

The next token is sampled only from this reduced group.

### 4.5 Top-p Sampling

Top-p sampling, also known as **nucleus sampling**, selects the smallest set of tokens whose cumulative probability is at least `p`.

The next token is then sampled from that set.

---

# Major Generations of Language Models

## 1. Statistical Language Models

Statistical language models estimate language probabilities using counts and statistical patterns from a corpus.

Examples include:

- unigram models,
- bigram models,
- trigram models,
- and general n-gram models.

## 2. Neural Language Models

Neural language models use neural networks to learn representations and patterns in language.

Examples include:

- feed-forward neural language models,
- recurrent neural networks,
- long short-term memory networks,
- and gated recurrent units.

### Common Architectures

- **RNN** — Recurrent Neural Network
- **LSTM** — Long Short-Term Memory
- **GRU** — Gated Recurrent Unit

LSTMs and GRUs were designed to improve how recurrent neural networks handle longer-term dependencies.

## 3. Transformer Language Models

Transformer language models use attention mechanisms instead of relying primarily on recurrence.

Major categories include:

- **decoder-only models**,
- **encoder-only models**,
- **encoder-decoder models**.

---

## 5. Attention Is All You Need

The 2017 paper [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) introduced the Transformer architecture.

The Transformer relied primarily on attention mechanisms rather than recurrent or convolutional structures.

### Self-Attention

Transformers use **self-attention** to calculate contextual relationships between tokens within the same sequence.

Self-attention assigns different levels of importance to different token representations when constructing a contextual representation.

For example, in the sentence:

> The animal did not cross the road because it was tired.

The model may use attention to determine that the token `it` is closely related to `animal`.

Transformers can generally process long-range relationships more efficiently than recurrent architectures such as standard RNNs and LSTMs.

---

# Tokens and Tokenisation

## 6. Tokens

A **token** is a unit of text processed by a language model.

A token may represent:

- an entire word,
- part of a word,
- punctuation,
- whitespace,
- a character,
- or a special symbol.

For example, depending on the tokenizer, the word:

```text
unbelievable
```

might be divided into tokens such as:

```text
un
believ
able
```

The process of dividing text into tokens is called **tokenisation**.

---

# Statistical Language Models

## 7. Maximum Likelihood Estimation

**Maximum Likelihood Estimation (MLE)** estimates probabilities using the number of times events occur in the training corpus.

For a bigram model:

```text
P(wi | w(i-1)) = C(w(i-1), wi) / C(w(i-1))
```

Where:

- `C(w(i-1), wi)` is the number of times the bigram occurs,
- `C(w(i-1))` is the number of times the preceding token occurs.

MLE selects the probability values that maximise the likelihood of the observed training data.

---

## 8. Conditional Probability

Conditional probability is the probability of an event occurring given that another event has already occurred.

```text
P(A | B) = P(A ∩ B) / P(B)
```

This is valid provided that:

```text
P(B) > 0
```

In language modelling, the events may be words, tokens, or sequences of tokens.

For example:

```text
P(book | a)
```

represents the probability that `book` occurs after `a`.

---

## 9. Probability Distributions

For a given context, a language model assigns a probability to every possible next token.

The probabilities must satisfy two conditions:

1. Each probability must be between `0` and `1`.
2. The probabilities across the vocabulary must sum to `1`.

```text
0 <= P(w | context) <= 1
```

```text
Sum of P(w | context) over every w in V = 1
```

Where `V` represents the model's vocabulary.

---

# N-Gram Language Models

## 10. Definition

An **n-gram language model** approximates the probability of the next token using only the previous `n - 1` tokens.

```text
P(wi | w1, ..., w(i-1))
≈
P(wi | w(i-n+1), ..., w(i-1))
```

The text is divided into subsequences containing `n` tokens.

### Types of N-Grams

| Model | Context used |
|---|---|
| Unigram | No previous tokens |
| Bigram | Previous 1 token |
| Trigram | Previous 2 tokens |
| 4-gram | Previous 3 tokens |
| General n-gram | Previous `n - 1` tokens |

![Unigram, bigram, and trigram example](https://storage.googleapis.com/wandb-production.appspot.com/madhana/images/projects/37263309/0271586a.png?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gorilla-files-url-signer%40wandb-production.iam.gserviceaccount.com%2F20260729%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20260729T063542Z&X-Goog-Expires=3599&X-Goog-Signature=b906d6ca06404008006b3c9caf4f645d11d9c93a8b40a5748ffce60bf6524346c5ca488eb9fec96e4d6779e6c004eac9929d2cebfc3311a34495d5226e88b367de9bbab173f7cd84ebab3ad95bb9937c47c0cbd859f9b9309834076077e80dd3852b2ec5807ca88c54ebcc8f02510d55c9d50d1143858b2fd5819d41fc41d67dbad18808f3c0d426b1b42ae9c2c518c4540273ecf6ea4df7bcbc94a0a2c64b4ac4c4fa8df1dca8588999d71a134a80ae93b1be24545954c8f10d1e42c0ad44f01d1f1e1b1173bd132adcc737fd4bf37044ae8a40054c03d22d8d76fe31af59177b532b7e3402683c914557820f951572604cacec45bcaa57529c16207e72aa24&X-Goog-SignedHeaders=host&X-User=wandb&response-content-type=image%2Fpng)

---

## 11. Probability Concepts Used in N-Gram Models

### 11.1 Chain Rule

The chain rule states that the probability of a sequence of events is the product of the probability of each event given the events before it.

```text
P(w1, w2, ..., wk)
=
P(w1)
× P(w2 | w1)
× P(w3 | w1, w2)
× ...
× P(wk | w1, ..., w(k-1))
```

In compact form:

```text
P(w1, ..., wk)
=
Product from i = 1 to k of P(wi | w1, ..., w(i-1))
```

### 11.2 Bayes' Rule

Bayes' rule is used to update the probability of an event using new information.

```text
P(A | B) = [P(B | A) × P(A)] / P(B)
```

Bayes' rule is important in probability and machine learning, although it is not required for the basic derivation of an n-gram model.

### 11.3 Markov Assumption

The **Markov assumption** states that the probability of the next token depends only on a limited number of previous tokens.

For an n-gram model:

```text
P(wi | w1, ..., w(i-1))
≈
P(wi | w(i-n+1), ..., w(i-1))
```

For a bigram model:

```text
P(wi | w1, ..., w(i-1))
≈
P(wi | w(i-1))
```

---

# Building a Bigram Language Model

## 12. Step 1: Create a Text Corpus

Consider the following corpus:

```text
Logesh read a novel
Deepak played tennis
Joshua read a book while travelling
Mukilan went to the store
```

A bigram model examines pairs of consecutive tokens.

Examples include:

```text
Logesh read
read a
a novel
Joshua read
a book
book while
```

### Estimating a Bigram Probability

The token `a` appears twice as a preceding token:

```text
a novel
a book
```

The bigram `a book` appears once.

Therefore:

```text
P(book | a) = C(a, book) / C(a)
```

```text
P(book | a) = 1 / 2 = 0.5
```

---

## 13. Step 2: Apply the Chain Rule

Let a sequence be represented as:

```text
W = (w1, w2, w3, ..., wk)
```

Using the chain rule:

```text
P(W)
=
P(w1)
× P(w2 | w1)
× ...
× P(wk | w1, ..., w(k-1))
```

This expression considers the complete history of all previous tokens.

---

## 14. Step 3: Apply the Markov Assumption

The complete history can be difficult to model because the number of possible contexts becomes extremely large.

The n-gram model simplifies the calculation using the Markov assumption:

```text
P(wi | w1, ..., w(i-1))
≈
P(wi | w(i-n+1), ..., w(i-1))
```

For a bigram model, `n = 2`:

```text
P(wi | w1, ..., w(i-1))
≈
P(wi | w(i-1))
```

---

## 15. Step 4: Construct the Bigram Model

For a bigram language model:

```text
P(W)
≈
P(w1)
× P(w2 | w1)
× ...
× P(wk | w(k-1))
```

In product notation:

```text
P(W)
≈
P(w1) × Product from i = 2 to k of P(wi | w(i-1))
```

However, this version does not clearly model where a sentence begins or ends.

---

# Sentence-Boundary Tokens

## 16. Start and End Tokens

A language model must represent the beginning and end of a sequence.

This can be done using special tokens:

- `<start>` — beginning of a sequence,
- `<end>` — end of a sequence.

The corpus becomes:

```text
<start> Logesh read a novel <end>
<start> Deepak played tennis <end>
<start> Joshua read a book while travelling <end>
<start> Mukilan went to the store <end>
```

The probability of a sentence is then:

```text
P(W)
=
P(w1 | start)
× P(w2 | w1)
× ...
× P(wk | w(k-1))
× P(end | wk)
```

---

## 17. Example Sentence Probability

Consider the sentence:

```text
Logesh read a book
```

Its bigram probability is:

```text
P(W)
=
P(Logesh | start)
× P(read | Logesh)
× P(a | read)
× P(book | a)
× P(end | book)
```

From the corpus:

```text
P(Logesh | start) = 1 / 4
```

```text
P(read | Logesh) = 1
```

Both occurrences of `read` are followed by `a`, so:

```text
P(a | read) = 1
```

The token `a` is followed once by `novel` and once by `book`, so:

```text
P(book | a) = 1 / 2
```

However, `book` is followed by `while` in the training corpus, not by `<end>`:

```text
P(end | book) = 0
```

Therefore:

```text
P(W)
=
(1 / 4)
× 1
× 1
× (1 / 2)
× 0
```

```text
P(W) = 0
```

Although the sentence is grammatically reasonable, the unsmoothed bigram model assigns it zero probability because the following bigram does not appear in the training corpus:

```text
book <end>
```

---

# Sparsity and Smoothing

## 18. Data Sparsity

An n-gram model can only directly estimate probabilities for token sequences that occur in its training corpus.

Even a large corpus cannot contain every possible valid sequence.

This causes the **data sparsity problem**.

---

## 19. Zero-Frequency Problem

If an n-gram has never appeared in the training corpus, maximum likelihood estimation assigns it a probability of zero.

```text
C(w(i-1), wi) = 0
```

Therefore:

```text
P(wi | w(i-1)) = 0
```

Because sentence probabilities are calculated by multiplication, one zero-probability bigram causes the probability of the entire sentence to become zero.

---

## 20. Add-One Smoothing

One simple solution is **add-one smoothing**, also called **Laplace smoothing**.

Add one to every possible bigram count:

```text
P_Laplace(wi | w(i-1)) = [C(w(i-1), wi) + 1]/[C(w(i-1)) + |V|]
```

Where:

- `C(w(i-1), wi)` is the bigram count,
- `C(w(i-1))` is the count of the preceding token,
- `|V|` is the vocabulary size.

For an unseen bigram:

```text
C(w(i-1), wi) = 0
```

Add-one smoothing gives:

```text
P_Laplace(wi | w(i-1)) = 1 / [C(w(i-1)) + |V|]
```

The probability is therefore small, but no longer zero.

### Important Note

A complete adjusted sentence probability cannot be calculated until the following are defined precisely:

- the vocabulary,
- whether `<start>` and `<end>` are included in the vocabulary,
- token normalisation rules,
- and the exact denominator used for each preceding token.

Add-one smoothing is useful for learning, but more advanced smoothing methods generally perform better in practical n-gram language models.

---

# Limitations of N-Gram Models

## 21. Main Limitations

N-gram language models have several important limitations:

1. **Limited context**  
   They only consider the previous `n - 1` tokens.

2. **Data sparsity**  
   Many valid n-grams may not appear in the training corpus.

3. **Large storage requirements**  
   The number of possible n-grams grows rapidly as `n` and the vocabulary size increase.

4. **Poor long-range dependency modelling**  
   They cannot effectively connect information separated by many tokens.

5. **Fixed representations**  
   They treat words or tokens as discrete symbols rather than learned contextual representations.

6. **Unknown words**  
   Words outside the training vocabulary require special handling, often using an `<unk>` token.

---

# Connection to Modern Language Models

Both n-gram models and modern neural language models attempt to estimate:

```text
P(wi | context)
```

The primary difference is how they represent and use context.

An n-gram model:

- uses discrete frequency counts,
- uses a short fixed context,
- and stores probabilities for observed token sequences.

A modern neural language model:

- learns continuous vector representations,
- can use much longer contexts,
- and generalises patterns to sequences that were not directly observed during training.

---

# References

1. Madhana, *Language Modeling: A Beginner's Guide*  
   <https://wandb.ai/madhana/Language-Models/reports/Language-Modeling-A-Beginner-s-Guide---VmlldzozMzk3NjI3>

2. 3Blue1Brown, *Large Language Models Explained Briefly*  
   <https://www.youtube.com/watch?v=LPZh9BOjkQs>

3. Vaswani et al., *Attention Is All You Need*  
   <https://arxiv.org/abs/1706.03762>
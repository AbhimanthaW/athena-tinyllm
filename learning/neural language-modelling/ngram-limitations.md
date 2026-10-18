#### **Limitation 1: Sparsity**

Suppose:
```
C(cat,runs) = 0

P(runs | cat) = 0
```

sentence in reality ≠ impossible

Add-one smoothing
- unseen events receive nonzero probability

but (cat, runs) and (cat, quantum) receive same treatment
- indiscriminate

#### **Limitation 2: no sharing between similar words**

P(runs | dog) = 0.8
P(barks | dog) = 0.2

suppose *puppy* appears less than *dog*

C(dog, runs) doesn't tell anything about C(puppy, runs)

**Model has no Geometry**
```
"dog"
"puppy"
"cat"
"runs"
```

identifiers distinguish tokens, but they don't represent relationships between them

no meaningful mathematical statement like
$$
distance(dog, puppy) < distance(dog, democracy)
$$
neural language models introduce learned numerical representations so relationships can emerge

**embeddings**
$$
[ e_{\text{dog}} = \begin{bmatrix} 0.7\\ -0.2\\ 0.4 \end{bmatrix}, \qquad e_{\text{puppy}} = \begin{bmatrix} 0.65\\ -0.16\\ 0.38 \end{bmatrix}.
$$

distance/angle becomes mathematically meaningful
$$[ \|e_{\text{dog}}-e_{\text{puppy}}\|.
$$

**Parameter Sharing**

$$
f_\theta(w_{t-1})
$$

| Symbol     | explanation            |
| ---------- | ---------------------- |
| $w_{t-1}$  | context token          |
| $\theta$   | all learned parameters |
| $f_\theta$ | neural computation     |
$$
f_\theta(w_{t-1}) = P(w_t\mid w_{t-1})
$$
$$
e_{\text{dog}}\rightarrow f_\theta(e_{\text{dog}}) ≈ e_{\text{puppy}}\rightarrow f_\theta(e_{\text{puppy}}).
$$
representations similar -- function produces similar predictions
- **parameter sharing**

**Generalisation rather than Memorisation**
$$
e_{\text{dog}} \rightarrow \text{shared computation} \rightarrow P(w_t)
$$

$$
e_{\text{puppy}} \rightarrow \text{same shared computation} \rightarrow P(w_t)
$$
information learned while processing `dog` can modify parameters that also affect predictions involving `puppy`

#### Limitation 3: parameter growth in n-gram models

n-grams = V^N

V = vocab size , n = number of items

as \(n\) increases:
- the theoretical parameter space explodes,
- most n-grams are unseen,
- counts become extremely sparse.

**neural models don't simply eliminate parameter growth**
- vocabulary size \(V\),
- embedding dimension \(d\).

embedding matrix
$$
V\times d
$$
output matrix
$$
d\times V
$$
*parameters are organised into shared representations and shared transformations*

**N-gram:** $$P(j\mid i)$$
**Neural model:**
learn representations $$e_i\in\mathbb R^d$$
transformations shared across many tokens$$f_\theta(e_i)$$
#### Limitation 4: fixed context

bigram assumes (Markov Property)
$$P(w_t\mid w_1,\ldots,w_{t-1}) \approx P(w_t\mid w_{t-1})$$
$$\text{more context} \Rightarrow \text{more possible contexts} \Rightarrow \text{greater sparsity}$$
#### Limitation 5: n-gram contexts are atomic

Statistics gathered under
$$(\text{the},\text{dog})$$
are not automatically shared with
$$(\text{a},\text{dog})$$

#### Transition

current:
```
previous token → count lookup → probabilities
```

neural language model:
```
context tokens → numerical vectors → learned transformations → scores → probabilities
```

```
token IDs → embeddings → hidden layer → logits → softmax → P(w | context)
```

The n-gram model learns probabilities for discrete events. A neural language model learns a function over learned numerical representations of those events.

#### what remains unchanged

perviously:
$$P(w_t\mid w_{t-1})$$
next:
$$P_\theta(w_t\mid\text{context})$$

| maximise                                      | taking logarithms                                |
| --------------------------------------------- | ------------------------------------------------ |
| $$\prod_t P_\theta(w_t\mid\text{context}_t)$$ | $$\sum_t\log P_\theta(w_t\mid\text{context}_t)$$ |

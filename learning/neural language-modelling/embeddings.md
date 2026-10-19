**computers need numerical representations**
$$h=Wx+b$$
```
x is an input vector,
W is a learned weight matrix,
b is is a learned bias vector,
h is  is the resulting vector.
```

*How do we turn discrete tokens into numerical objects on which we can perform meaningful mathematical operations?*

Token: `"dog"`
**↓**
Vocabulary index: `2`
**↓**
One-hot vector: [0,0,1,0,0]
**↓**
Embedding: [0.3,-0.8,0.2]

**Vocabulary Indices**

corpus
$$\mathcal V=\{\text{cat, dog, puppy, runs, sleeps}\}$$

| Token  | Token ID |
| ------ | -------- |
| cat    | 0        |
| dog    | 1        |
| puppy  | 2        |
| runs   | 3        |
| sleeps | 4        |
```
dog runs
```
becomes:
```
[1, 3]
```

⚠️ Token IDs have no useful numerical geometry

- Token IDs identify words. They do not describe words.

**One-Hot Encoding**

Let the vocabulary size be:  `V=5`

|        |     |     |     |     |     |
| :----: | :-: | :-: | :-: | :-: | :-: |
|  cat   |  1  |  0  |  0  |  0  |  0  |
|  dog   |  0  |  1  |  0  |  0  |  0  |
| puppy  |  0  |  0  |  1  |  0  |  0  |
|  runs  |  0  |  0  |  0  |  1  |  0  |
| sleeps |  0  |  0  |  0  |  0  |  1  |

Mathematically, the one-hot vector for `dog` is:
$$
x_{\text{dog}}= \begin{bmatrix} 0\\1\\0\\0\\0 \end{bmatrix} \in\mathbb R^5 
$$

$\mathbb R^5$  = space of vectors containing five real-valued components

*One-hot encoding avoids implying that token 2 is numerically twice token 1*

Consider two different one-hot vectors:
$$
x_{\text{dog}}= \begin{bmatrix}0\\1\\0\\0\\0\end{bmatrix}, \qquad x_{\text{puppy}}= \begin{bmatrix}0\\0\\1\\0\\0\end{bmatrix}
$$
Euclidean distance:
$$\begin{aligned} \|x_{\text{dog}}-x_{\text{puppy}}\|_2 &=\sqrt{(1-0)^2+(0-1)^2}\\ &=\sqrt2 \end{aligned}$$
$$\|x_{\text{dog}}-x_{\text{sleeps}}\|_2=\sqrt2$$
every pair of distinct one-hot vectors has distance $\sqrt2$

*One-hot encoding represents identity, but it still doesn't represent semantic similarity*

**Embeddings: learning a numerical space**

$$d=3$$
$d$ is the embedding dimension

$$ E\in\mathbb R^{V\times d} $$
For vocabulary size $V=5$

$$ E\in\mathbb R^{5\times3} $$

suppose:

$$
E= \begin{bmatrix} 0.2&-0.1&0.5\\ 0.7&0.3&-0.2\\ 0.6&0.4&-0.1\\ -0.3&0.8&0.2\\ 0.1&-0.5&0.9 \end{bmatrix}
$$

| Token  | Embedding        |
| ------ | ---------------- |
| cat    | $[0.2,-0.1,0.5]$ |
| dog    | $[0.7,0.3,-0.2]$ |
| puppy  | $[0.6,0.4,-0.1]$ |
| runs   | $[-0.3,0.8,0.2]$ |
| sleeps | $[0.1,-0.5,0.9]$ |
**How to obtain an embedding?**

$$ x_{\text{dog}}= \begin{bmatrix}0\\1\\0\\0\\0\end{bmatrix} $$
$$ \boxed{e_{\text{dog}}=E^\top x_{\text{dog}}} $$

Dimensions:
$$ \underbrace{E^\top}_{3\times5} \underbrace{x_{\text{dog}}}_{5\times1} = \underbrace{e_{\text{dog}}}_{3\times1} $$

$$ e_{\text{dog}}= \begin{bmatrix}0.7\\0.3\\-0.2\end{bmatrix} $$

*Embedding lookup is mathematically equivalent to multiplying a one-hot vector by an embedding matrix*

$$ x_{\text{dog}} = \text{One-hot Vector} $$
$$ E^\top = \text{Embedding Matrix (transpose)} $$
$$ e_{\text{dog}} = \text{Embedding Result} $$

**Improvement**

| $e_{\text{dog}}= \begin{bmatrix}0.7\\0.3\\-0.2\end{bmatrix}$ | $e_{\text{puppy}}= \begin{bmatrix}0.6\\0.4\\-0.1\end{bmatrix}$ |
| ------------------------------------------------------------ | -------------------------------------------------------------- |
Euclidian distance:

$\begin{aligned} \|e_{\text{dog}}-e_{\text{puppy}}\|_2 &=\sqrt{0.1^2+(-0.1)^2+(-0.1)^2}\\ &=\sqrt{0.03}\\ &\approx0.173 \end{aligned}$

meaningful numerical measurement in the model's representation space

During training, the model's prediction errors will be used to adjust the matrix entries
The hope is that useful relationships emerge because they help the model predict tokens.

**embedding is a parameter, not a fixed dictionary**

$$ E_{\text{initial}} \longrightarrow \text{training} \longrightarrow E_{\text{learned}} $$

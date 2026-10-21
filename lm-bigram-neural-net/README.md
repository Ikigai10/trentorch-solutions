# Bigram Neural Network

Intermediate | language-modeling | neural-networks | gradient-descent

### The problem, from first principles

The count table and a tiny neural network can represent the same model. Give every symbol a one-hot vector, multiply by a weight matrix `W` of shape `(V, V)` to get **logits** (unnormalized log-probabilities), and apply softmax to turn each row into a distribution. Multiplying a one-hot vector by `W` just selects a row, so row `i` of `W` plays the role of the logits for "what follows symbol `i`". The difference is how you obtain the numbers: instead of counting, you start with `W = 0` (a uniform model) and nudge it by gradient descent to lower the NLL of the training pairs. This is the first step from statistics to learning: the same loss, the same softmax and the same update rule scale up unchanged to transformers.

### From theory to code

Implement `bigram_net_loss`, `bigram_net_gradient` and `train_bigram_net` for training pairs `(xs[k], ys[k])` meaning symbol `ys[k]` followed symbol `xs[k]`.

### Constraints

- `xs` and `ys` are integer arrays of the same length and `W` has shape `(V, V)`. Row `xs[k]` of `W` holds the logits for pair `k`.
- `bigram_net_loss(W, xs, ys)` is the mean over pairs of `-log softmax(W[xs[k]])[ys[k]]`. Use a numerically stable softmax by subtracting the row maximum.
- `bigram_net_gradient(W, xs, ys)` is the exact gradient of that mean loss with respect to `W`: for each pair add `softmax(W[x]) - onehot(y)` to row `x`, then divide by the number of pairs.
- `train_bigram_net(xs, ys, V, steps, lr)` starts from `W = zeros((V, V))` and applies `steps` plain gradient-descent updates `W -= lr * gradient`. Return `W`.

### Hints

<details>
<summary>Hint 1</summary>

Let `P = softmax(W[xs])` have shape `(n, V)`. Subtract 1 at the true column of each row, then scatter-add the rows into a `(V, V)` array indexed by `xs` with `np.add.at`.

</details>

<details>
<summary>Hint 2</summary>

`np.add.at(grad, xs, P)` accumulates correctly even when the same row index repeats, which `grad[xs] += P` does not.

</details>

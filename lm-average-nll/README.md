# Average Negative Log-Likelihood

Beginner | language-modeling | evaluation

### The problem, from first principles

Having built a model, how good is it? The honest question is how much probability it gave to what actually happened. For one word, multiply the probability of each step: a pair that the model finds likely contributes a number near one and a pair it finds unlikely contributes a number near zero. Products of many small numbers underflow, so we take logs and add. Flipping the sign gives the **negative log-likelihood** (NLL), where lower is better, and dividing by the number of steps makes models comparable across datasets of different sizes. A model that knows nothing and spreads probability evenly over `V` symbols scores exactly `ln V`, so that number is the bar any real model should beat.

### From theory to code

Implement `average_nll(words, probs, itos)`, the mean negative log-probability that the model assigns to every adjacent pair in the wrapped words.

### Constraints

- `probs` is a row-stochastic `(V, V)` array where `probs[i, j]` is the probability of `itos[j]` after `itos[i]`. `itos[0]` is the boundary symbol `.`.
- Wrap each word as `.` + word + `.`. Every adjacent pair counts once, including the pairs touching the boundary.
- Return the mean of `-log(probs[i, j])` over all pairs in all words, as a float. Use the natural logarithm.
- If a pair has probability exactly 0 the result is `inf`. Do not clip.

### Hints

<details>
<summary>Hint 1</summary>

Build `stoi` from `itos`, collect the row and column indices of every pair into two lists, then index `probs` once with both lists.

</details>

<details>
<summary>Hint 2</summary>

Mean of the negative logs is `-np.log(p).mean()`.

</details>

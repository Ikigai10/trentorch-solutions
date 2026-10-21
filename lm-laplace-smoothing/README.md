# Laplace Smoothing

Beginner | language-modeling | smoothing

### The problem, from first principles

Counts become probabilities by dividing each row by its total. That is the maximum-likelihood estimate, and it has a sharp flaw: any pair you never saw gets probability exactly zero. One unseen pair in a test word then makes the whole word impossible, and its log-probability becomes minus infinity. A language model should be a little humble about what it has not observed. Adding a small pseudo-count `alpha` to every cell before normalizing shifts a sliver of probability from the common pairs to the unseen ones, so nothing is ever impossible.

### From theory to code

Implement `smoothed_probabilities(counts, alpha)`, which turns a count matrix into a row-stochastic probability matrix with add-`alpha` smoothing.

### Constraints

- `counts` is a non-negative `(V, V)` array and `alpha > 0`.
- Return `(counts + alpha) / (row total + alpha * V)` so each row sums to 1.
- Do not modify `counts`.
- A row of all zeros becomes the uniform distribution.

### Hints

<details>
<summary>Hint 1</summary>

Add `alpha` to the whole matrix, then divide each row by its own sum with `keepdims=True` so broadcasting divides row by row.

</details>

<details>
<summary>Hint 2</summary>

The denominator is just the row sum of the smoothed matrix, so no separate formula is needed.

</details>

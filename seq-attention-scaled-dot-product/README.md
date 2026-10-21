# Scaled dot-product attention, forward

Intermediate | transformers

### The problem, from first principles

`[03-recurrent-neural-networks/07-seq2seq-bottleneck]` demonstrated the core weakness of the classic sequence-to-sequence encoder-decoder: forcing an ENTIRE input sequence through one fixed-size hidden state disproportionately preserves LATE information over early information. Attention removes this bottleneck entirely by letting every OUTPUT position directly look at EVERY input position, choosing dynamically, for each output position separately, which input positions actually matter, rather than relying on one compressed summary.

The mechanism: every position produces three vectors, a QUERY ("what am I looking for"), a KEY ("what do I have to offer, for matching purposes"), and a VALUE ("what do I have to offer, for actually being retrieved"). A query's similarity to every position's key (measured via a dot product, the SAME similarity measure `[00-math-and-statistics/01-linear-algebra/02-dot-product-norms]` introduced early in this curriculum) determines how much weight that position's VALUE gets in the final weighted average. This is genuinely just a differentiable, "soft" version of a lookup table: instead of retrieving ONE exact match (`[04-seq-modeling/02-embeddings/01-token-embedding-lookup]`'s hard, exact-id lookup), attention retrieves a WEIGHTED BLEND of every value, weighted by how well each key matched the query.

### From theory to code

Implement `scaled_dot_product_attention(query, key, value, mask)`. Compute raw similarity scores via `query @ key^T`, scale them DOWN by `1/sqrt(d_k)` (`d_k` is the query/key vectors' own dimensionality), optionally add `mask` (large negative values at positions that should never be attended to), apply `softmax` along the LAST axis (so each query position's weights sum to exactly `1`), and use those weights to compute a weighted sum of `value`.

### Constraints

- Scores must be scaled by `1/sqrt(d_k)` BEFORE the mask is added and BEFORE softmax, not after.
- `mask`, when provided, is ADDED to the scaled scores (not multiplied, not applied after softmax).
- `softmax` must be applied along the LAST axis (over KEY positions), so each ROW of `attention_weights` (one query position) sums to exactly `1`.
- Must work for any number of leading batch/head dimensions, not just a single unbatched `(seq_len, d_k)` pair.

### Hints

<details>
<summary>Hint 1: Computing and scaling the scores</summary>

`scores = query @ np.swapaxes(key, -2, -1) / np.sqrt(d_k)`: `np.swapaxes(key, -2, -1)` transposes just the LAST two axes of `key` (so this works correctly regardless of how many leading batch dimensions there are), giving `scores` shape `(..., seq_len_q, seq_len_k)`.

</details>

<details>
<summary>Hint 2: Applying the mask</summary>

`if mask is not None: scores = scores + mask`, a simple conditional addition; when `mask` is `None`, skip it entirely and use the raw scaled scores.

</details>

<details>
<summary>Hint 3: Softmax over the last axis, then the weighted sum</summary>

A numerically-stable softmax over the LAST axis: `scores_shift = scores - np.max(scores, axis=-1, keepdims=True)`, `weights = np.exp(scores_shift) / np.sum(np.exp(scores_shift), axis=-1, keepdims=True)`. Then `output = weights @ value`. Return `(output, weights)`.

</details>

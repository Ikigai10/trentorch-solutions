# Embedding backward (scatter-add gradient)

Intermediate | nlp | transformers | embeddings | autograd

### The problem, from first principles

`[01-token-embedding-lookup]`'s `embedding_forward` is a SELECTION operation: it picks out specific rows from `embedding_table` and returns them. `[02-deep-learning-core/04-autograd]`'s `matmul_backward` derived how gradients flow backward through a matrix multiplication, but embedding lookup isn't a matmul, it's indexing, and indexing needs its own backward rule. The intuition is straightforward once stated: only the rows that were ACTUALLY looked up during the forward pass can possibly have contributed to the loss, so every row that was never selected gets a gradient of exactly ZERO, while a row that WAS selected gets the upstream gradient from wherever it was used.

The genuinely tricky part, and the reason this question exists separately from a generic "backward for indexing" exercise: the SAME token id very commonly appears MULTIPLE times within a single batch (the word "the" might appear a dozen times across a batch of sentences), and each occurrence independently contributes its own gradient to that SAME row of the embedding table. The correct gradient for that row is the SUM of every occurrence's individual contribution, not just the last one, or the first one, a naive implementation that simply ASSIGNS `grad_table[id] = grad_output[position]` for each position, overwriting rather than accumulating, would silently drop every occurrence except the last one seen, quietly corrupting training for every token that appears more than once per batch (which, for common words, is nearly always).

### From theory to code

Implement `embedding_backward(grad_output, token_ids, vocab_size)`. Build a zero-initialized `grad_table` of shape `(vocab_size, embed_dim)`. For every position in `token_ids`, ADD (not overwrite) that position's corresponding slice of `grad_output` into `grad_table` at the row given by that position's token id.

### Constraints

- `grad_table` must start at all zeros: rows never referenced by `token_ids` must remain exactly zero.
- When a token id appears MULTIPLE times in `token_ids`, that row's final gradient must be the SUM of every occurrence's own gradient contribution.
- `grad_table`'s shape must be exactly `(vocab_size, embed_dim)`, regardless of `token_ids`'s own shape (a single sequence or a whole batch).
- Do not use a Python `for` loop with plain indexed assignment (`grad_table[id] = ...`); this silently drops accumulation for repeated ids.

### Hints

<details>
<summary>Hint 1: Flattening first</summary>

`token_ids` might be multi-dimensional (a whole batch). Flatten both `token_ids` and `grad_output` down to a simple `(num_positions,)` array of ids and a matching `(num_positions, embed_dim)` array of gradients before processing: `token_ids.reshape(-1)` and `grad_output.reshape(-1, embed_dim)`.

</details>

<details>
<summary>Hint 2: The accumulation operation</summary>

`np.add.at(grad_table, flat_ids, flat_grad)` is NumPy's built-in "scatter-add": unlike `grad_table[flat_ids] += flat_grad` (which, for REPEATED indices, only applies the LAST write due to how NumPy's fancy-indexing assignment is implemented, silently dropping earlier occurrences), `np.add.at` correctly accumulates every single occurrence, even when the same index appears many times in `flat_ids`.

</details>

<details>
<summary>Hint 3</summary>

`embed_dim = grad_output.shape[-1]` reads the embedding dimension directly off `grad_output`'s own last axis, no need to pass it in separately.

</details>

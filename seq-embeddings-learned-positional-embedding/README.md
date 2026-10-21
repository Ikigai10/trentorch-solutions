# Learned positional embedding

Beginner | nlp | transformers | embeddings

### The problem, from first principles

`[03-sinusoidal-positional-encoding]` solved the "tell the model where each token sits" problem with a fixed, hand-derived FORMULA, sine and cosine waves at carefully chosen frequencies, never updated during training. There's a simpler, more direct alternative: just let position 0, position 1, position 2, ... each have their OWN dedicated, freely learnable vector, exactly the same idea as `[01-token-embedding-lookup]`'s token embedding table, except the thing being looked up is a POSITION instead of a token identity. Gradient descent then figures out, entirely from the training data, whatever positional pattern turns out to actually be useful, rather than the model being locked into the specific sinusoidal pattern chosen in advance.

The tradeoff this simplicity introduces is real and worth naming directly: a learned positional embedding table has a FIXED maximum length (`position_table`'s number of rows), baked in at model-construction time, and it has genuinely never seen, let alone learned anything useful about, any position beyond that maximum. A model trained this way, given a sequence LONGER than any it saw during training, has no learned embedding to look up for those extra positions at all. `[03-sinusoidal-positional-encoding]`'s fixed formula, by contrast, can simply be COMPUTED for any position, including ones far beyond anything seen during training, which is part of why some architectures still prefer the sinusoidal (or RoPE, the most modern approach, covered later in this track) formulation specifically for its ability to generalize to longer sequences.

### From theory to code

Implement `learned_positional_embedding(seq_len, position_table)`. `position_table` has shape `(max_seq_len, embed_dim)`, a genuinely trainable parameter (unlike `[03-sinusoidal-positional-encoding]`'s fixed table). Return its first `seq_len` rows, one per position, in order.

### Constraints

- Row `0` of the result must be `position_table`'s row `0` (position 0's embedding), row `1` must be `position_table`'s row `1`, and so on, in order, with no reordering.
- The result's shape must be exactly `(seq_len, embed_dim)`.
- Must not modify `position_table` itself.

### Hints

<details>
<summary>Hint 1</summary>

Since positions are always used IN ORDER, starting from 0, no arbitrary id-based lookup (like `[01-token-embedding-lookup]`'s fancy indexing) is needed here at all, just a plain slice: `position_table[:seq_len]`.

</details>

<details>
<summary>Hint 2</summary>

A NumPy slice (as opposed to fancy indexing with an array of indices) returns a VIEW into the original array by default, not a copy, this is fine here since the function is only ever meant to READ from `position_table`, never write to the returned result in a way that should affect the original table.

</details>

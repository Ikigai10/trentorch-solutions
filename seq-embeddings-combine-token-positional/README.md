# Combine token and positional embeddings

Beginner | nlp | transformers | embeddings

### The problem, from first principles

Every previous question in this track built one HALF of what a Transformer's input actually needs: `[01-token-embedding-lookup]` produces a vector encoding WHAT each token is, and `[03-sinusoidal-positional-encoding]`/`[04-learned-positional-embedding]` produce a vector encoding WHERE each token sits in the sequence. Neither alone is enough: a model that only sees "what" has no sense of word order at all (the "the dog bit the man" problem `[03-sinusoidal-positional-encoding]`'s Statement raised directly); a model that only saw "where" would have no idea WHICH word is actually at each position. This question is the simplest possible way to combine both signals into a single representation: just ADD them together, element-wise.

Addition (rather than, say, concatenation) might look like it should destroy information, mixing "what" and "where" into the same numbers rather than keeping them in separate dimensions, but it works remarkably well in practice, and is what the original Transformer paper actually does. Part of why: `embed_dim` is typically large (512, 768, or more), so there's enormous room for the token-identity signal and the positional signal to occupy substantially different, only lightly-overlapping DIRECTIONS within that high-dimensional space, without erasing each other, a phenomenon closely related to why high-dimensional random vectors tend to be nearly orthogonal to each other purely by chance.

### From theory to code

Implement `combine_embeddings(token_embeddings, positional_embeddings)`. `token_embeddings` has shape `(batch_size, seq_len, embed_dim)`; `positional_embeddings` has shape `(seq_len, embed_dim)` (the SAME positional encoding is reused for every sequence in the batch, since position 0 means the same thing regardless of which sequence it's part of). Add them together, letting NumPy's broadcasting handle applying the same `(seq_len, embed_dim)` positional table across every item in the batch.

### Constraints

- The output shape must match `token_embeddings`'s shape exactly, `(batch_size, seq_len, embed_dim)`.
- `positional_embeddings` (no batch dimension) must be broadcast identically across every sequence in the batch, not resized or tiled manually.
- Must be a pure element-wise addition, no scaling, normalization, or other transformation applied.

### Hints

<details>
<summary>Hint 1</summary>

`token_embeddings + positional_embeddings` is the entire implementation: NumPy's broadcasting rules automatically align `positional_embeddings`'s `(seq_len, embed_dim)` shape against `token_embeddings`'s trailing two dimensions, applying it identically across the leading `batch_size` dimension.

</details>

<details>
<summary>Hint 2</summary>

No reshaping, tiling, or explicit loop over the batch dimension is needed, or even correct, broadcasting handles the batch dimension automatically as long as the two arrays' TRAILING dimensions (`seq_len, embed_dim`) already match.

</details>

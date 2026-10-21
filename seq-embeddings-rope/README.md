# RoPE (Rotary Position Embeddings)

Advanced | nlp | transformers | embeddings

### The problem, from first principles

`[05-combine-token-positional-embeddings]`'s approach, add a positional vector to the token embedding once, right at the input, has a subtle limitation: that positional information has to survive, unmodified in spirit, through every subsequent layer of the network for attention to still make use of it many layers deep. RoPE (Su et al., 2021) takes a fundamentally different approach: instead of adding position information to the EMBEDDING, it applies a ROTATION directly to the query and key vectors INSIDE every attention layer, right before the attention score (`Scaled dot-product attention`, later in this Part) is computed from them. This isn't just a different place to inject the same information, it produces a genuinely different, and in practice very useful, mathematical property: the dot product between a rotated query at position `m` and a rotated key at position `n` depends ONLY on their RELATIVE distance, `m - n`, not on their absolute positions `m` and `n` individually.

This relative-position property matters enormously in practice: a model trained mostly on short sequences, where it only ever sees absolute positions up to, say, 2048, can often still handle a MUCH longer sequence at inference time, because what it actually learned to respond to is relative offsets ("the token 3 positions back"), a pattern that's exactly as valid at position 50000 as it is at position 50, whereas `[04-learned-positional-embedding]`'s fixed lookup table simply has no learned row at all for a position index it's never seen. This is a major reason RoPE, or close variants of it, is the standard positional scheme in most modern large language models (Llama, GPT-NeoX, and many others).

### From theory to code

Implement `compute_rope_angles(seq_len, dim)`, computing the rotation angle for every `(position, dimension-pair)` combination (same underlying frequency formula as `[03-sinusoidal-positional-encoding]`'s `div_term`). Implement `apply_rope(x, angles)`, rotating every consecutive PAIR of dimensions in `x` by that pair's own angle, treating `(x[..., 2i], x[..., 2i+1])` as the coordinates of a 2D point being rotated counterclockwise by `angles[..., i]` radians.

### Constraints

- `compute_rope_angles` returns shape `(seq_len, dim // 2)`: one angle per position, per dimension-PAIR (not per individual dimension).
- `apply_rope` must rotate `(x[..., 2i], x[..., 2i+1])` pairs using the standard 2D rotation matrix, applied independently to each pair with its own angle.
- Rotation must preserve each pair's VECTOR NORM exactly (a genuine rotation never changes a vector's length, only its direction).
- The dot product between two rotated vectors at different positions must depend only on their POSITIONS' DIFFERENCE, not on the absolute position values themselves.

### Hints

<details>
<summary>Hint 1: compute_rope_angles</summary>

Identical structure to `[03-sinusoidal-positional-encoding]`'s `div_term`/`position` computation: `freq = 10000.0 ** (-np.arange(0, dim, 2) / dim)` gives one frequency per dimension-pair, and `position * freq` (broadcasting a `(seq_len, 1)` column against a `(dim//2,)` row) gives every position's angle at every pair's frequency.

</details>

<details>
<summary>Hint 2: Splitting into pairs</summary>

`x1 = x[..., 0::2]` and `x2 = x[..., 1::2]` split `x`'s last dimension into its even-indexed and odd-indexed halves, EXACTLY the two coordinates of each consecutive pair, ready to rotate as 2D points.

</details>

<details>
<summary>Hint 3: The rotation itself</summary>

`np.empty_like(x)` for the output, then fill `rotated[..., 0::2] = x1 * cos(angles) - x2 * sin(angles)` and `rotated[..., 1::2] = x1 * sin(angles) + x2 * cos(angles)`, the standard 2D counterclockwise rotation matrix formula applied to each `(x1, x2)` pair independently.

</details>

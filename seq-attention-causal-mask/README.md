# Causal mask

Beginner | transformers

### The problem, from first principles

`[01-scaled-dot-product-attention]`'s attention, unrestricted, lets every position attend to EVERY other position, including positions that come AFTER it. For understanding a fully-available piece of text (translation, classification, `[03-recurrent-neural-networks/06-bidirectional-rnn]`'s exact motivation for bidirectionality), this is entirely appropriate, the whole input is already there to look at. But for GENERATING text, one token at a time, autoregressively, letting a model attend to tokens it hasn't generated yet would be a form of "cheating" that makes no sense at inference time: when actually generating token 5, tokens 6, 7, 8 genuinely don't exist yet, there's nothing there to attend to. Worse, if this restriction isn't enforced during TRAINING (where the full target sequence IS available upfront, for efficiency, via "teacher forcing"), the model would learn to rely on seeing future tokens, a capability it will never actually have at real inference time, making training and inference fundamentally mismatched.

The causal mask enforces this directly: position `i` is allowed to attend to positions `0` through `i` (itself and everything before it), and is explicitly FORBIDDEN from attending to positions `i+1` and beyond.

### From theory to code

Implement `build_causal_mask(seq_len)`, an additive mask of shape `(seq_len, seq_len)`: `0` at every `(i, j)` where `j <= i` (allowed), `-inf` at every `(i, j)` where `j > i` (forbidden). Passed as `[01-scaled-dot-product-attention]`'s `mask` argument, `-inf` added to a score BEFORE softmax drives that position's exponentiated score, and therefore its final attention weight, to exactly `0`.

### Constraints

- `mask[i, j] = 0` for every `j <= i` (the LOWER TRIANGLE, including the diagonal).
- `mask[i, j] = -inf` for every `j > i` (the strict UPPER TRIANGLE, excluding the diagonal).
- Position `i` must always be allowed to attend to ITSELF (`j == i`, on the diagonal, must be `0`, not `-inf`).
- Output shape must be exactly `(seq_len, seq_len)`.

### Hints

<details>
<summary>Hint 1</summary>

`np.triu(np.ones((seq_len, seq_len)), k=1)` gives a matrix of `1`s in the STRICT upper triangle (`k=1` excludes the main diagonal) and `0`s everywhere else, exactly marking the positions that should be forbidden.

</details>

<details>
<summary>Hint 2</summary>

Start with `mask = np.zeros((seq_len, seq_len))`, then use the upper-triangle indicator from Hint 1 to set exactly those positions to `-np.inf`: `mask[upper_triangle == 1] = -np.inf`.

</details>

<details>
<summary>Hint 3</summary>

Double-check the diagonal specifically: `k=1` in `np.triu` is what excludes the main diagonal from being marked forbidden, using `k=0` instead would incorrectly forbid a position from attending to ITSELF.

</details>

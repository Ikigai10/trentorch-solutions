# Padding, Truncation & Attention Masks

Beginner | tokenization | batching | attention-mask

### The problem, from first principles

GPUs process rectangular batches, but sentences have different lengths. To stack them, short sequences are filled with a **pad token** up to a common length and long ones are cut. The model must then be told which positions are real, which is the job of the **attention mask**: ones for real tokens and zeros for padding. Where the padding goes matters. Training usually pads on the right. Generation pads on the **left**, so that every sequence's final real token sits in the last column and the next-token prediction can be read from one fixed position.

### From theory to code

Implement `pad_batch`.

### Constraints

- `pad_batch(seqs, pad_id, max_len=None, side='right')` takes a list of integer lists. The target length is `max_len` if given, else the longest sequence.
- A sequence longer than the target length is **truncated by keeping its first `target` tokens**.
- Shorter sequences are padded with `pad_id` on `side` (`'right'` or `'left'`; anything else raises `ValueError`).
- Return `(ids, mask)`, two integer arrays of shape `(len(seqs), target)`. `mask` is `1` at real tokens and `0` at padding, even if a real token happens to equal `pad_id`.

### Hints

<details>
<summary>Hint 1</summary>

Build the mask from lengths, not by comparing ids to `pad_id`.

</details>

<details>
<summary>Hint 2</summary>

For left padding, the real tokens occupy the last `len` columns.

</details>

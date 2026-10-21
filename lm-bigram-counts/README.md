# Bigram Counts

Beginner | language-modeling | counting

### The problem, from first principles

A language model is a machine that, given what has been written so far, says how likely each possible next symbol is. The simplest honest version looks back exactly one symbol. To learn it from a list of words you count how often each symbol follows each other symbol, then read the counts as evidence. The only wrinkle is the edges of a word: a word starts and ends, and the model should learn how words start and how they stop. Adding one special boundary symbol `.` at both ends of every word turns "how words begin" and "how words end" into ordinary follow-counts.

### From theory to code

Implement `bigram_counts(words)`. It returns the count matrix and the list of symbols that labels its rows and columns. The skeleton is already in the editor.

### Constraints

- `words` is a list of non-empty strings. The symbol `.` never appears inside a word.
- The symbol list `itos` is `['.']` followed by the sorted distinct characters of all words. Row and column `i` of the matrix belong to `itos[i]`.
- Wrap each word as `.` + word + `.` and count every adjacent pair. `counts[i, j]` is the number of times `itos[j]` directly followed `itos[i]`.
- Return `(counts, itos)` with `counts` an integer array of shape `(V, V)`.

### Hints

<details>
<summary>Hint 1</summary>

Build `itos` first, then a dictionary from symbol to index so counting is one lookup.

</details>

<details>
<summary>Hint 2</summary>

`zip(seq, seq[1:])` walks adjacent pairs of the wrapped word.

</details>

<details>
<summary>Hint 3</summary>

The matrix total equals the sum over words of `len(word) + 1`, which is a handy self-check.

</details>

# BPE: single merge step

Intermediate | nlp | tokenization

### The problem, from first principles

`[01-whitespace-char-tokenizer]` and `[02-vocabulary-building]` demonstrated the two extremes: whitespace tokens (compact sequences, unbounded vocabulary, every unseen word becomes `<unk>`) versus character tokens (tiny fixed vocabulary, but every word becomes a long sequence of individual letters). Byte-Pair Encoding (BPE) is the practical middle ground essentially every modern LLM (GPT, Llama, and many others) actually uses in production: start from character-level tokens, then repeatedly find the MOST FREQUENTLY adjacent pair of tokens anywhere in the training corpus and merge it into one new token, building progressively larger, more meaningful subword units out of the raw characters. Run enough merge steps, and common whole words end up as single tokens (since their constituent character pairs get merged together long before rarer pairs do), while genuinely rare or novel words gracefully decompose into a handful of smaller, still-meaningful subword pieces, "unbelievable" might become `["un", "believ", "able"]` rather than one opaque `<unk>`, retaining SOME of the word's structure instead of losing it entirely.

This question implements exactly ONE step of that repeated process: find the single most common adjacent pair across the whole corpus, and merge every occurrence of it. `Stretch: BPE, full training loop`, immediately after this question, wraps this single step in a loop, repeating it however many times are needed to reach a target vocabulary size.

### From theory to code

Implement `get_pair_frequencies(corpus)` (count every adjacent token pair's frequency across a corpus of token-lists), `merge_pair(corpus, pair)` (replace every occurrence of `pair` with one merged token, everywhere it appears), and `bpe_single_merge_step(corpus)` (find the single most frequent pair and merge it, returning both the updated corpus and which pair was merged).

### Constraints

- `get_pair_frequencies` counts pairs WITHIN each sequence only, never across the boundary between two different sequences in the corpus.
- `merge_pair` must merge occurrences NON-OVERLAPPING and left-to-right: after merging positions `i` and `i+1` into one token, the NEXT possible merge starts checking from position `i+2`, not `i+1`.
- `bpe_single_merge_step` must pick the pair with the HIGHEST count; ties should be broken by picking the pair that sorts alphabetically GREATER (as a tuple), for a fully deterministic result.
- The merged token is the literal string CONCATENATION of the pair's two tokens (`pair[0] + pair[1]`), not joined by any separator.

### Hints

<details>
<summary>Hint 1: get_pair_frequencies</summary>

For each `sequence` in `corpus`, loop `for i in range(len(sequence) - 1)` and increment `pair_counts[(sequence[i], sequence[i+1])]` by one. A `collections.Counter` handles the "increment, starting from zero" bookkeeping automatically.

</details>

<details>
<summary>Hint 2: merge_pair's left-to-right scan</summary>

Walk each sequence with an explicit index `i` (not a `for` loop over the sequence directly, since the step size varies): if `sequence[i], sequence[i+1]` matches `pair`, append the merged token and advance `i` by `2`; otherwise append `sequence[i]` unchanged and advance `i` by `1`.

</details>

<details>
<summary>Hint 3: bpe_single_merge_step</summary>

`max(pair_counts.items(), key=lambda kv: (kv[1], kv[0]))[0]` finds the pair with the highest count, using the pair itself as a tiebreak (Python tuples compare lexicographically, matching the alphabetically-greater tiebreak rule). Then call `merge_pair(corpus, that_pair)` and return both.

</details>

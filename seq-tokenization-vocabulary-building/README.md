# Vocabulary building + unknown-token handling

Beginner | nlp | tokenization

### The problem, from first principles

`[01-whitespace-char-tokenizer]` turns raw text into a list of string tokens, but `[02-embeddings/01-token-embedding-lookup]`'s embedding lookup, and everything downstream of it, needs INTEGER ids, not strings, a lookup table is indexed by position, not by spelling. Building that string-to-integer mapping (the "vocabulary") from a training corpus is this question's first job. The second, equally important job: a vocabulary built from a FIXED training corpus will, by definition, never contain every string that might appear in future text the model encounters, someone will eventually type a typo, a brand-new word, or a name the vocabulary has never seen. Rather than crashing or silently corrupting the input, a real tokenizer reserves one special token, conventionally called `<unk>` ("unknown"), as the fallback id for anything outside the known vocabulary.

A secondary, practical concern this question also handles: including every single token that appears even ONCE in a large training corpus can bloat the vocabulary with rare misspellings, one-off names, and noise, each of which would get its own learned embedding vector that barely ever gets a useful training signal. Filtering by a minimum frequency threshold (`min_freq`) keeps the vocabulary focused on tokens common enough to be worth a dedicated embedding.

### From theory to code

Implement `build_vocabulary(token_lists, min_freq, unk_token)`, counting every token's frequency across the whole corpus (a list of token lists), reserving id `0` for `unk_token`, and assigning ids to every OTHER token with frequency `>= min_freq`, in order of descending frequency (ties broken alphabetically, for a deterministic, reproducible result). Implement `encode_with_unk(tokens, vocab, unk_token)`, mapping a list of tokens to their ids, substituting `unk_token`'s id for anything not present in `vocab`.

### Constraints

- `unk_token` must always be assigned id `0`, regardless of whether it happens to also appear literally in the corpus.
- Tokens with frequency below `min_freq` must be EXCLUDED from the vocabulary entirely (they'll map to `<unk>` at encoding time, via `encode_with_unk`).
- Remaining tokens must be assigned ids in order of DESCENDING frequency, with ties broken ALPHABETICALLY, so the same corpus always produces the exact same vocabulary.
- `encode_with_unk` must never raise a `KeyError`: any out-of-vocabulary token maps to `unk_token`'s id instead.

### Hints

<details>
<summary>Hint 1: Counting</summary>

`collections.Counter()` with `.update(tokens)` called once per sequence in `token_lists` accumulates a total frequency count across the whole corpus in a few lines.

</details>

<details>
<summary>Hint 2: Deterministic ordering</summary>

`sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))` sorts by frequency descending (`-kv[1]`, negating to reverse the usual ascending sort) with alphabetical order (`kv[0]`) as the tiebreak, a single sort key handles both requirements at once.

</details>

<details>
<summary>Hint 3: Assigning ids</summary>

Start `vocab = {unk_token: 0}`, then loop over the sorted `(token, count)` pairs, and for each one where `count >= min_freq`, assign `vocab[token] = len(vocab)` (the CURRENT size of `vocab` is exactly the next available id, since ids are assigned in order starting from `0`).

</details>

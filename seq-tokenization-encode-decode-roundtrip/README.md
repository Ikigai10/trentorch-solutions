# Encode/decode round-trip

Beginner | nlp | tokenization

### The problem, from first principles

This track's four previous questions each built one PIECE of a real tokenizer in isolation: `[01-whitespace-char-tokenizer]` splits text into tokens, `[02-vocabulary-building]` maps tokens to ids (and handles unknowns), `[03-bpe-single-merge]` and `[04-bpe-full-training-loop]` build a smarter, learned vocabulary. This question is the payoff: wire the pieces together into the complete round trip a real tokenizer actually needs to support, `encode` (text -> a list of integer ids, ready to feed `[02-embeddings/01-token-embedding-lookup]`) and `decode` (ids -> text, needed to turn a language model's OUTPUT predictions, which are token ids, back into human-readable text).

A subtlety worth naming directly: `decode(encode(text))` is generally NOT guaranteed to reproduce `text` EXACTLY, character for character. `[02-vocabulary-building]`'s `<unk>` handling is lossy by design (an unknown word becomes `<unk>`, and there's no way to recover the original spelling from that alone), and `whitespace_tokenize` itself already collapses multiple spaces and strips leading/trailing whitespace, so even a perfectly round-tripped SEQUENCE OF TOKENS might not exactly match the original text's precise whitespace. What round-tripping IS guaranteed to preserve, for text made entirely of tokens the vocabulary actually knows, is the sequence of WORDS, in order, joined back together with single spaces.

### From theory to code

Implement `encode(text, vocab, unk_token)`, chaining `whitespace_tokenize` (already provided, reused from `[01-whitespace-char-tokenizer]`) into `encode_with_unk` (already provided, reused from `[02-vocabulary-building]`). Implement `build_inverse_vocab(vocab)`, constructing the REVERSE mapping (id -> token) from `vocab`'s forward mapping. Implement `decode(ids, vocab)`, converting each id back to its token via the inverse vocabulary and joining them with single spaces.

### Constraints

- `encode` must produce exactly what calling `whitespace_tokenize` then `encode_with_unk` in sequence would produce, reusing both rather than reimplementing their logic.
- `build_inverse_vocab` must correctly invert EVERY entry in `vocab`, including id `0` (`<unk>` or whatever the unknown token is named).
- `decode` must join tokens with a SINGLE space between each pair, matching what `whitespace_tokenize` would produce if it re-tokenized the decoded string.
- For text made entirely of tokens present in `vocab` (no `<unk>` needed), `decode(encode(text), vocab)` must exactly reproduce `text`'s WORD SEQUENCE, though not necessarily its exact original whitespace.

### Hints

<details>
<summary>Hint 1: encode</summary>

`tokens = whitespace_tokenize(text)`, then `return encode_with_unk(tokens, vocab, unk_token)`, a direct two-step chain, no additional logic needed.

</details>

<details>
<summary>Hint 2: build_inverse_vocab</summary>

A dict comprehension: `{token_id: token for token, token_id in vocab.items()}`, swapping each `(token, id)` pair's roles.

</details>

<details>
<summary>Hint 3: decode</summary>

Build the inverse vocab once (via `build_inverse_vocab`), then `tokens = [inverse_vocab[token_id] for token_id in ids]`, and `return " ".join(tokens)`.

</details>

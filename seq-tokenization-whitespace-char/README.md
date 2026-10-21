# Whitespace/character tokenizer

Beginner | nlp | tokenization

### The problem, from first principles

Every question in this Part, and everything a language model does, starts from the same unavoidable first step: raw text (a sequence of Unicode characters) has to be chopped up into discrete units, TOKENS, before any of the math this curriculum has built so far (`[03-dl-training/02-layers/01-linear-forward]`'s matrix multiplications, `[02-deep-learning-core]`'s autograd engine) can be applied to it at all. A neural network has no native way to consume "the letter c followed by the letter a followed by the letter t"; it needs a fixed, finite VOCABULARY of tokens, each one identified by an integer id, so text can become a sequence of numbers a network can actually process.

The two simplest possible strategies for choosing what counts as "one token" sit at opposite extremes. Whitespace tokenization treats each SPACE-SEPARATED word as one token: fast, and each token usually carries a lot of meaning, but the vocabulary needed to cover a language is enormous (every inflected form of every word, "run," "runs," "running," "ran," all become entirely separate, unrelated tokens), and any word never seen during training becomes an unrecognizable "unknown" token, no matter how similar it is to a known one. Character tokenization treats each INDIVIDUAL character as one token: the vocabulary is tiny (just the alphabet, digits, and punctuation), and literally no input text can ever be truly "unknown," but sequences become much LONGER (a 5-letter word becomes 5 tokens instead of 1), which makes it far harder for a model to learn long-range structure. `BPE: single merge step`, later in this track, is the practical middle ground essentially every modern LLM actually uses, built by starting from character tokenization and progressively merging frequently-adjacent pairs.

### From theory to code

Implement `whitespace_tokenize(text)`, splitting `text` into a list of tokens on whitespace, and `char_tokenize(text)`, splitting `text` into a list of its individual characters.

### Constraints

- `whitespace_tokenize` must collapse any run of consecutive whitespace (multiple spaces, tabs, newlines) into a single split point, and must drop leading/trailing whitespace entirely (no empty-string tokens from it).
- `char_tokenize` must return one token per character, INCLUDING whitespace and punctuation characters (unlike `whitespace_tokenize`, nothing gets dropped or collapsed).
- Both functions must work correctly on an empty string input (`whitespace_tokenize("")` returns `[]`; `char_tokenize("")` also returns `[]`).

### Hints

<details>
<summary>Hint 1</summary>

Python's built-in `str.split()`, called with NO arguments, already does exactly the whitespace-collapsing, leading/trailing-stripping behavior this question needs: `"  hello   world  ".split()` gives `['hello', 'world']`.

</details>

<details>
<summary>Hint 2</summary>

`list(text)` converts any string directly into a list of its individual characters, `list("cat")` gives `['c', 'a', 't']`, this already handles whitespace and punctuation as their own tokens with no special-casing needed.

</details>

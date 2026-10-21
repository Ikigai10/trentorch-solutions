# Trigram Models & Longer Context

Intermediate | language-modeling | n-gram

### The problem, from first principles

Looking back one symbol is too short. After "the cat" the next word is very different from after "the dog", yet a bigram model sees only "the" or "cat" alone. Conditioning on the previous **two** tokens gives a trigram model, and the trade-off appears at once: longer context makes predictions sharper, but the number of possible contexts grows with the square of the vocabulary, so most contexts are seen rarely or never. This is the data-sparsity problem that smoothing softens and that neural models later solve by sharing statistical strength between similar contexts. At the word level we also need sentence boundaries, so each sentence is padded with two start markers and one end marker.

### From theory to code

Implement `trigram_next_distribution(sentences, w1, w2, alpha)`. It returns the vocabulary and the smoothed probability of each token following the context `(w1, w2)`.

### Constraints

- `sentences` is a list of token lists. Pad each as `['<s>', '<s>'] + tokens + ['</s>']`.
- The vocabulary `vocab` is the sorted list of distinct tokens appearing in the sentences plus `'</s>'`. The marker `'<s>'` is never predicted, so it is not in `vocab`.
- Count every triple `(a, b, c)` of consecutive padded tokens. For the context `(w1, w2)` return `(count(w1, w2, c) + alpha) / (total for this context + alpha * len(vocab))` for each `c` in `vocab`.
- `alpha > 0`. A context never seen gives the uniform distribution.
- Return `(vocab, probs)` with `probs` a NumPy array aligned with `vocab`.

### Hints

<details>
<summary>Hint 1</summary>

A dictionary keyed by the pair `(a, b)` whose value is a `Counter` of next tokens keeps the lookup cheap.

</details>

<details>
<summary>Hint 2</summary>

Only the tokens after the chosen context matter, so you can stop counting everything else.

</details>

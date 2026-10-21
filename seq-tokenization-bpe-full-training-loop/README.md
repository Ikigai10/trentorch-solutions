# Stretch: BPE, full training loop

Advanced | nlp | tokenization

### The problem, from first principles

`[03-bpe-single-merge]` implemented exactly ONE merge step: find the most frequent adjacent pair, merge it everywhere. A real BPE tokenizer's vocabulary is built by running that single step REPEATEDLY, typically tens of thousands of times for a production-scale tokenizer, each merge building on top of the results of every merge before it (so a token created by merge #500 can itself become one half of a pair merged in step #501). The full LIST of merges learned, in order, effectively IS the trained tokenizer: to encode brand new text later, you don't re-run the whole frequency-counting process from scratch, you simply replay the exact same sequence of merges, in the exact same order they were originally learned, against the new text's character-level tokens.

This ordering requirement is the subtle but essential detail this question is really about: merge #500 might only make sense to apply AFTER merges #1 through #499 have already been applied (since it might be merging two tokens that themselves only exist because of earlier merges). Applying the learned merges out of order, or in a different order than they were learned, would produce a completely different, likely broken tokenization.

### From theory to code

Implement `train_bpe(corpus, num_merges)`, calling `[03-bpe-single-merge]`'s `bpe_single_merge_step` (already provided, reused via `load_solution`) exactly `num_merges` times in a loop, feeding each step's output corpus into the NEXT step's input, and collecting every merged pair, IN ORDER, into a list. Implement `apply_merges(tokens, merges)`, replaying a previously-learned list of merges, in the SAME order, against a fresh sequence of character-level tokens (for text the tokenizer wasn't trained on).

### Constraints

- `train_bpe` must feed each merge step's OUTPUT corpus as the NEXT step's input (this is what lets later merges build on top of earlier ones), not repeatedly re-run merges against the original, unmerged corpus.
- `train_bpe`'s returned `merges` list must be in the EXACT order the merges were learned (the order `bpe_single_merge_step` returned them across the loop's iterations).
- `apply_merges` must apply the merges in that SAME order, calling `[03-bpe-single-merge]`'s `merge_pair` once per learned merge, in sequence.
- If `num_merges` exceeds the number of distinct pairs still available to merge, `train_bpe` doesn't need to handle that edge case gracefully (assume `num_merges` is always achievable for the given corpus).

### Hints

<details>
<summary>Hint 1: train_bpe</summary>

`merges = []`, then `for _ in range(num_merges): corpus, merged_pair = bpe_single_merge_step(corpus); merges.append(merged_pair)`. Reassigning `corpus` inside the loop, to the PREVIOUS iteration's returned corpus, is what makes each merge build on the last.

</details>

<details>
<summary>Hint 2: apply_merges</summary>

`for pair in merges: tokens = merge_pair([tokens], pair)[0]`. `merge_pair` expects a CORPUS (a list of sequences), so wrap the single `tokens` sequence in a one-element list before calling it, and unwrap the result (`[0]`) afterward.

</details>

<details>
<summary>Hint 3</summary>

The loop in `apply_merges` must iterate `merges` in the exact order given (a plain `for pair in merges:`, no sorting or reordering), since later merges in the list were specifically learned to apply AFTER the earlier ones.

</details>

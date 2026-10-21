# Special Tokens & Chat Templates

Beginner | tokenization | chat-models | special-tokens

### The problem, from first principles

A chat model is trained on one long stream of text, so the boundaries between a system message, a user turn and an assistant turn have to be _written into the text_ with reserved marker strings. These **special tokens** (such as `<|user|>` or `<|end|>`) must be recognized as single atomic tokens, never split by the ordinary tokenizer into pieces, and the exact template the model was fine-tuned with must be reproduced at inference time, or quality quietly degrades. Two small jobs make this concrete: render a list of messages into the template string, and split a rendered string so that special tokens stay whole while ordinary text is left for the normal tokenizer.

### From theory to code

Implement `render_chat` and `split_on_specials`.

### Constraints

- `messages` is a list of dicts with keys `role` and `content`. `render_chat(messages, add_generation_prompt)` renders each message as `<|{role}|>\n{content}<|end|>\n`, concatenated in order. If `add_generation_prompt` is true, append `<|assistant|>\n` so the model continues as the assistant.
- `split_on_specials(text, specials)` returns a list of pieces in order. Each occurrence of a string from `specials` is its own piece, and the text between occurrences is one piece (empty pieces are omitted). At any position prefer the **longest** special that matches.
- Concatenating the pieces must reproduce `text` exactly.

### Hints

<details>
<summary>Hint 1</summary>

Scan left to right: at each index test the specials sorted by length descending with `text.startswith(s, i)`.

</details>

<details>
<summary>Hint 2</summary>

Accumulate ordinary characters in a buffer and flush it whenever a special matches.

</details>

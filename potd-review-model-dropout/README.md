# REVIEW MODEL DROPOUT

Beginner | neural-networks

**Difficulty:** Easy
**Tags:** Neural Networks

---

### Story

Amazon's review-helpfulness classifier uses dropout during training, and because dropout involves
randomness, the mask is given directly rather than generated, so every submission is deterministically
checkable.

---

### The Math

```
out = (x * m) / p_keep
```

where `m` is the given binary mask and `p_keep` is the keep probability (inverted-dropout scaling,
matching a framework's `Dropout` at train time).

### Input Format

```
n p_keep
x_1 ... x_n
m_1 ... m_n
```

`m_i` is `0` or `1`, given directly, not generated.

### Output Format

`out`, `n` values, 6 decimals.

### Constraints

- `1 <= n <= 10^5`, `0 < p_keep <= 1`
- Time limit: 1.0 second.

---

### Example

**Input**

```
4 0.75
1.0 2.0 3.0 4.0
1 0 1 1
```

**Output**

```
1.333333 0.000000 4.000000 5.333333
```

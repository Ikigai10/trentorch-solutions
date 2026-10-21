# THE GRADIENT SEATBELT

Beginner | optimization

**Difficulty:** Easy
**Tags:** Optimization

---

### Story

PayPal's fraud model occasionally sees gradient spikes from rare, extreme-value transactions.
Global-norm gradient clipping is the seatbelt that keeps one outlier batch from blowing up training.

---

### The Math

```
if ||g||_2 > tau:
    g <- g * (tau / ||g||_2)
else:
    g <- g
```

### Input Format

```
d tau
g_1 ... g_d
```

### Output Format

Clipped gradient, `d` values, 6 decimals.

### Constraints

- `1 <= d <= 10^4`, `tau > 0`
- Time limit: 1.0 second.

---

### Example

**Input**

```
2 2.0
3.0 4.0
```

**Output**

```
1.200000 1.600000
```

**Explanation:** `||g||_2 = 5.0 > tau = 2.0`, so scale by `2.0/5.0 = 0.4`:
`(3.0, 4.0) * 0.4 = (1.2, 1.6)`.

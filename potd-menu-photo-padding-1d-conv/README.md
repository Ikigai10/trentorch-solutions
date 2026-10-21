# MENU PHOTO PADDING

Beginner | computer-vision

**Difficulty:** Easy
**Tags:** Computer Vision

---

### Story

Zomato's menu-photo pipeline needs 1D-signal intuition for 2D convolution padding before the full
CNN track: implement "same" versus "valid" padding for a 1D convolution as the warm-up.

---

### The Math

**Valid:** no padding, output length `= n - k + 1`.

**Same:** pad symmetrically with zeros so output length `= n` (for odd kernel size `k`, pad
`floor(k / 2)` on each side).

### Input Format

```
n k
x_1 ... x_n
kernel_1 ... kernel_k
mode  (either "valid" or "same")
```

### Output Format

Convolution output, space-separated, 6 decimals.

### Constraints

- `1 <= k <= n <= 10^4`, `k` odd
- Time limit: 1.0 second.

---

### Example

**Input**

```
5 3
1 2 3 4 5
1 0 -1
valid
```

**Output**

```
-2.000000 -2.000000 -2.000000
```

Same input with `mode = same`:

```
-2.000000 -2.000000 -2.000000 -2.000000 4.000000
```

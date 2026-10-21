# Compute a Stable L2 Norm

Beginner | problemset | maths-stats-for-ml | linear-algebra

Compute the Euclidean (L2) norm of a non-empty real vector. Entries may be extremely large or extremely small, so the computation must not overflow or underflow.

Implement `solve(x)`.

**Returns.** Return the norm as a Python float. The all-zero vector has norm `0.0`.

### Examples

**Example 1**

Input:

```python
solve([3.0, 4.0])
```

Output:

```text
5.0
```

**Example 2**

Input:

```python
solve([3e200, 4e200])
```

Output:

```text
5e+200
```

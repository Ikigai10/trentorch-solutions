# Assemble a Streamed Response Into Display Snapshots

Beginner | production-systems | streaming | real-time-agents

### The problem, from first principles

A streamed LLM response doesn't arrive as one string -- it arrives as a sequence of small chunks, and a UI rendering it live needs to know, after each chunk, what the full text looks like so far. Storing only the raw chunks means recomputing that concatenation on every single render.

### The task

Write `assemble_stream(chunks)` that returns the list of cumulative text snapshots: `snapshots[i]` is everything received up through and including `chunks[i]`.

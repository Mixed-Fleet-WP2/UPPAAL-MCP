# Models

## train_gate

[`train_gate.xml`](train_gate.xml) models a level crossing with one train and one gate. It has two timed automata: `Train` with clock `x` and `Gate` with clock `y`. They synchronise on two binary channels:

- `appr`: the train is approaching.
- `leave`: the train has left the crossing.

### Train

```
Far ──appr!, x=0──▶ Near ──x>15, x=0──▶ Cross ──leave!, crossings++──▶ Far
                  (x ≤ 20)            (x ≤ 5)
```

| Location | Invariant | Meaning |
|---|---|---|
| `Far` (initial) | none | Away from the crossing; it may stay here for any length of time. |
| `Near` | `x <= 20` | Approaching. It enters `Cross` once `x > 15`, so after more than 15 and at most 20 time units. |
| `Cross` | `x <= 5` | Inside the crossing. `x` is reset on entry, so it must leave within 5 time units. |

On leaving, the train increments `crossings` (`int[0,3]`, modulo 4). No query uses this counter.

### Gate

```
Up ──appr?, y=0──▶ Lowering ──y≥12──▶ Down ──leave?, y=0──▶ Raising ──y≥5──▶ Up
                  (y ≤ 15)                                 (y ≤ 10)
```

| Location | Invariant | Meaning |
|---|---|---|
| `Up` (initial) | none | Open; waits for `appr`. |
| `Lowering` | `y <= 15` | Closing; it reaches `Down` after 12 to 15 time units. |
| `Down` | none | Closed; waits for `leave`. |
| `Raising` | `y <= 10` | Opening; it reaches `Up` after 5 to 10 time units. |

### How they interact

Both channels are binary, so the sender and receiver move together:

- The train can only send `appr` while the gate is `Up`. If the gate is still `Raising`, the train waits in `Far`.
- The train can only send `leave` while the gate is `Down`.

Here is one cycle, with time counted from `appr`. Both clocks are reset at that moment, so `x = y`.

1. **t = 0:** the train goes to `Near` and the gate goes to `Lowering`.
2. **12 ≤ t ≤ 15:** the gate reaches `Down`.
3. **15 < t ≤ 20:** the train enters `Cross`. The gate is already `Down`.
4. **Within 5 time units:** the train leaves, and the gate starts `Raising`.
5. **5 to 10 time units later:** the gate is `Up` again.

### History

The original model let the train enter `Cross` after 10 time units, but the gate could need up to 15 time units to close. This broke R2. The train's guard is now the strict `x > 15`, which rules out entering at t = 15 while the gate is still `Lowering`. The comment in the model's global declarations still describes the old bug.

### Requirements

Confirmed requirements are in [`queries/train_gate.q`](../queries/train_gate.q):

- **R1:** the system never deadlocks (`A[] not deadlock`).
- **R2:** whenever the train is in the crossing, the gate is down (`A[] (train.Cross imply gate.Down)`).

Candidate requirements waiting for confirmation are in [`queries/candidates/train_gate.q`](../queries/candidates/train_gate.q).

To check them all, run `verify(model="train_gate")` through the uppaal MCP server.

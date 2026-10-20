# UPPAAL query language cheat sheet (symbolic model checking)

## Path formulas

| Query | Name | Meaning | Trace verifyta gives |
|---|---|---|---|
| `A[] p` | invariantly / safety | `p` holds in **every reachable state** | counterexample if violated (last state violates `p`) |
| `E<> p` | possibly / reachability | **some reachable state** satisfies `p` | witness if satisfied (last state satisfies `p`) |
| `A<> p` | eventually / liveness | on **every path**, `p` eventually holds | counterexample if violated (loop or deadlock/time-lock) |
| `E[] p` | potentially always | there is **a path** along which `p` holds forever | witness if satisfied |
| `p --> q` | leads to | whenever `p` holds, `q` eventually holds; same as `A[] (p imply A<> q)` | counterexample if violated |

Duality: `A[] p` == `not E<> not p`, and `A<> p` == `not E[] not p`.
Path quantifiers cannot be nested; `-->` is the only exception.

## State predicates (`p`, `q`)

- Locations: `train.Cross` (process `train` is in location `Cross`). With templates
  instantiated as an array: `P(1).Wait`.
- Variables and clocks: `crossings == 2`, `train.x <= 5`, `gate.y - train.x > 3`.
  Local variables are accessed as `process.var`.
- Boolean operators: `and`, `or`, `not`, `imply` (also `&&`, `||`, `!`).
- Quantifiers over bounded ranges: `forall (i : id_t) P(i).Idle`, `exists (i : int[0,3]) a[i] > 0`.
- `deadlock`: true in a state with no enabled action transition, now or after any delay.
  Typical use: `A[] not deadlock`.

## Requirement patterns

| Requirement | Query |
|---|---|
| "Never X" | `A[] not X` |
| "Whenever A, then B" (state-wise) | `A[] (A imply B)` |
| Mutual exclusion | `A[] not (P1.CS and P2.CS)` |
| "X is possible" (also a sanity check) | `E<> X` |
| "Every request is eventually served" | `P.Req --> P.Served` |
| "Always eventually X" | `A<> X` (from the initial state), or `true --> X` (from every reachable state) |
| "No deadlock" | `A[] not deadlock` |
| Bounded response "served within 10" | add an observer clock `z` reset on request, then `A[] (P.Req imply z <= 10)` |

UPPAAL has no time-bounded `A<>`/`-->` operator; express bounded response with a
clock and an `A[]` query, or with an observer automaton.

## Pitfalls

- **Vacuity**: `A[] (p imply q)` holds trivially if `p` is unreachable. Check `E<> p` first.
- **Liveness and waiting**: `A<> p` and `p --> q` are violated when a process can stay
  forever in a location without an invariant, or the system can deadlock. A
  counterexample that just "waits forever" may point to a missing invariant
  (modeling error), not a design bug.
- **Time-locks**: invariants that cannot be left show up as deadlocks for `A[] not deadlock`.
- **Strict vs non-strict bounds**: `x >= 15` and `x > 15` differ at the boundary
  instant where several transitions may be enabled at once.
- **Integer ranges**: `int` is 16-bit `[-32768, 32767]`; overflow is a run-time
  error reported as verdict "error", not a property violation.

## Query file format (`.q`)

One query per line; `/* ... */` comments before a query document it, `//` comments
are ignored. Requirements are recorded as `/* REQ: <text> */` above their formula.

```
/* REQ: whenever the train is in the crossing, the gate is down */
A[] (train.Cross imply gate.Down)
```

## Verdicts returned by this server

`satisfied` (proved), `violated` (disproved, with trace), `maybe` (approximation was
inconclusive), `timeout` and `error` (not decided). Only `satisfied` means the
property holds; `maybe`, `timeout` and `error` are **unknown**.

Statistical (SMC) queries (`Pr[...]`, `simulate`, `E[<=...]`) are not supported.

# Sokoban Solver — push-optimal, room by room

[![make.fun · sokoban-solver-01-run-20261121](https://make.fun/badge/sokoban-solver-01-run-20261121.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121?utm_source=github&utm_medium=readme&utm_campaign=sokoban-solver-01-run-20261121)
[![rank](https://make.fun/badge/sokoban-solver-01-run-20261121/kai0x3/sokoban-solver/rank.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=sokoban-solver-01-run-20261121)
[![reward](https://make.fun/badge/sokoban-solver-01-run-20261121/kai0x3/sokoban-solver/reward.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121?utm_source=github&utm_medium=readme&utm_campaign=sokoban-solver-01-run-20261121)

Entry for "Build the best Sokoban solver, 21 November 2026" ($100, 10 places paid).
Free entry, no fee. Ends Mon 23 Nov 2026 03:00 UTC (10:00 WIB).

A single `main.ts` (Node 22, no dependencies) that answers every instance of
a set in one run: it reads `/input/<id>/level.txt` (XSB) and writes the
player's moves to `/output/<id>/solution.txt`.

## Approach

The scoring pays the full 1,000 points for a level only at its push-optimal
count P*, and under half a push later — so this solver is built to find
push-optimal solutions, not merely solutions.

- **Room decomposition.** A generated level is a hallway with independent
  rooms off it, and a box can never leave its room, so P* is the sum of the
  rooms' own optima. The solver finds the rooms by blacking out the dead
  squares of the player's row (a box on the hallway reaches no goal) and
  splitting the floor into parts; each part holding boxes is a room. Rooms
  are then solved one at a time with every other box on the board standing
  as a wall, the player walking on from wherever the previous room's answer
  left them.
- **A\* over pushes.** Each room is searched in push space: a state is the
  sorted box squares plus the canonical lowest square of the player's
  reachable region, kept in a transposition table with its best known cost.
  The heuristic is a minimum-cost matching of boxes to goals over true
  single-box push distances (computed by a reverse pull-BFS per goal),
  which is admissible — so the first solution an A\* run returns is
  push-optimal for that room, and the rooms add up to P* for the level.
- **Deadlock pruning that never touches the optimum.** Pushes landing on
  dead squares (from which no goal is reachable at all) are cut by the
  distance table; a box completing a frozen 2×2 block of walls and boxes
  with a box off its goal is cut; and a conservative recursive freeze test
  cuts boxes that can provably never be pushed again. Doubtful cases always
  count as movable, so no solvable state is ever pruned.
- **Deterministic by construction.** All budgets are expansion counts,
  never the clock; ties in the open list break by fixed rules (lower f,
  then deeper g, then older state). Two executions on the same set write
  byte-identical answers.
- **Fallbacks.** A room whose optimal search exhausts its budget is retried
  with a greedy best-first dive; a level that does not split into clean
  rooms is attempted whole (push A\*, then a step-level A\*). An instance
  that somehow still fails gets an empty — but valid — answer file, so the
  entry always answers all five instances.

## Local results

On the kit's practice set the solver scores the maximum **1,000,000**
(1,000 points on every instance, each solved in exactly P* pushes:
13, 16, 29, 47 and 60), finishing the whole set in about 0.35 s of the
60 s wall budget. The house baseline (plain-astar 1.0.0) solves only the
first practice instance and scores 200,000. On 12 freshly drawn sets from
the kit's own generator (arbitrary seeds), the solver also scored
1,000,000 on every set, solving each in 0.21–0.47 s.

## Layout

- `main.ts` — the whole solver: parser, room splitter, push-space A\*,
  deadlock tests, move reconstruction, and the step-level fallback.
- `LICENSE` — MIT.

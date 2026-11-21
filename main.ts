// sokoban-solver: a push-optimal Sokoban solver for make.fun's generated levels.
//
// How it works, in short:
//   * A generated level is a hallway with independent rooms off it; no box can
//     ever leave its room, so the level's push optimum is the sum of its rooms'
//     optima. This program finds the rooms (see splitRooms), then solves each
//     room on its own with the other rooms' boxes standing as walls.
//   * Each room is solved by an A* search over pushes (never over steps): a
//     state is the sorted box squares plus the canonical lowest square of the
//     player's reachable region. The estimate is a minimum-cost matching of
//     boxes to goals over true single-box push distances, which is admissible,
//     so the first solution found is push-optimal.
//   * Dead positions are cut early: boxes on squares from which no goal is
//     reachable at all, boxes in a frozen 2x2 block, and boxes proved frozen
//     by a conservative recursive test. None of these can occur in a state
//     that still solves, so the optimum is untouched.
//   * Every budget below counts expansions, never the clock, so two runs of
//     this program on the same set write byte-identical answers.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** Direction table: index 0..3 is up, down, left, right. */
const DX = [0, 0, -1, 1];
const DY = [-1, 1, 0, 0];
const STEP_LETTER = ['u', 'd', 'l', 'r'];
const PUSH_LETTER = ['U', 'D', 'L', 'R'];

/** Expansion budgets: work counters only, so runs are reproducible. */
const ROOM_ASTAR_LIMIT = 2_000_000;
const ROOM_GREEDY_LIMIT = 2_000_000;
const WHOLE_ASTAR_LIMIT = 2_000_000;
const WHOLE_MOVE_LIMIT = 1_500_000;

interface Board {
  w: number;
  h: number;
  size: number;
  /** nbr[i * 4 + d] is the square next to i in direction d, or -1 off board. */
  nbr: Int32Array;
  wall: Uint8Array;
  goal: Uint8Array;
  goals: number[];
  boxes: number[];
  player: number;
}

/** Parse XSB text the way the checker does: short rows and the outside are walls. */
function parseLevel(text: string): Board {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const size = w * h;
  const wall = new Uint8Array(size).fill(1);
  const goal = new Uint8Array(size);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') {
        goal[i] = 1;
        goals.push(i);
      }
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  const nbr = new Int32Array(size * 4).fill(-1);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      for (let d = 0; d < 4; d++) {
        const nx = x + DX[d];
        const ny = y + DY[d];
        if (nx >= 0 && ny >= 0 && nx < w && ny < h) nbr[i * 4 + d] = ny * w + nx;
      }
    }
  }
  boxes.sort((a, b) => a - b);
  goals.sort((a, b) => a - b);
  return { w, h, size, nbr, wall, goal, goals, boxes, player };
}

/**
 * Squares from which a lone box can still be pushed onto some goal, found by
 * pulling boxes backwards off the goals: a square is live when a box there
 * can be pushed one step towards a live square with room to stand behind it.
 */
function liveSquares(board: Board, wall: Uint8Array, goals: number[]): Uint8Array {
  const { nbr, size } = board;
  const live = new Uint8Array(size);
  const queue: number[] = [];
  for (const g of goals) {
    live[g] = 1;
    queue.push(g);
  }
  for (let head = 0; head < queue.length; head++) {
    const cur = queue[head];
    for (let d = 0; d < 4; d++) {
      const mid = nbr[cur * 4 + d]; // a box here could be pushed onto cur
      if (mid < 0 || live[mid] || wall[mid]) continue;
      const stand = nbr[mid * 4 + d]; // the player pushes from the far side
      if (stand < 0 || wall[stand]) continue;
      live[mid] = 1;
      queue.push(mid);
    }
  }
  return live;
}

/**
 * Push distances from every square to one goal: dist[s] is the fewest pushes
 * that move a lone box from s onto the goal, or -1 when it cannot be done.
 * A breadth-first walk over pushes, run backwards from the goal.
 */
function pushDistances(board: Board, wall: Uint8Array, goalSq: number): Int32Array {
  const { nbr, size } = board;
  const dist = new Int32Array(size).fill(-1);
  dist[goalSq] = 0;
  const queue = [goalSq];
  for (let head = 0; head < queue.length; head++) {
    const cur = queue[head];
    for (let d = 0; d < 4; d++) {
      const mid = nbr[cur * 4 + d];
      if (mid < 0 || dist[mid] >= 0 || wall[mid]) continue;
      const stand = nbr[mid * 4 + d];
      if (stand < 0 || wall[stand]) continue;
      dist[mid] = dist[cur] + 1;
      queue.push(mid);
    }
  }
  return dist;
}

/** Flood fill of the player's reachable squares; returns the region's lowest square. */
function floodRegion(
  board: Board,
  wall: Uint8Array,
  occupied: Uint8Array,
  from: number,
  seen: Int32Array,
  stamp: number,
  stack: Int32Array,
): number {
  const { nbr } = board;
  let low = from;
  let n = 0;
  stack[n++] = from;
  seen[from] = stamp;
  while (n > 0) {
    const p = stack[--n];
    if (p < low) low = p;
    for (let d = 0; d < 4; d++) {
      const q = nbr[p * 4 + d];
      if (q >= 0 && !wall[q] && !occupied[q] && seen[q] !== stamp) {
        seen[q] = stamp;
        stack[n++] = q;
      }
    }
  }
  return low;
}

/** Whether the box that just landed on `t` completes a frozen 2x2 of walls and boxes with a box off goal. */
function frozenSquareBlock(
  board: Board,
  wall: Uint8Array,
  occupied: Uint8Array,
  goal: Uint8Array,
  t: number,
): boolean {
  const { w } = board;
  const solid = (i: number): boolean => i >= 0 && (wall[i] === 1 || occupied[i] === 1);
  const corners = [t, t - 1, t - w, t - w - 1];
  for (const a of corners) {
    const b = a + 1;
    const c = a + w;
    const e = c + 1;
    if (!solid(a) || !solid(b) || !solid(c) || !solid(e)) continue;
    if (
      (occupied[a] && !goal[a]) ||
      (occupied[b] && !goal[b]) ||
      (occupied[c] && !goal[c]) ||
      (occupied[e] && !goal[e])
    ) {
      return true;
    }
  }
  return false;
}

/**
 * A conservative freeze test: true only when the box on square `s` can never
 * be pushed again, whatever the other boxes do. Along one axis a push needs
 * both neighbouring squares clear at push time; a wall never clears, and a
 * box already proved frozen never clears. Anything doubtful counts as
 * movable, so the test never condemns a box that could still move.
 */
function boxFrozen(
  board: Board,
  wall: Uint8Array,
  occupied: Uint8Array,
  s: number,
  visiting: Set<number>,
): boolean {
  if (visiting.has(s)) return false; // a cycle of boxes: assume they can sort themselves out
  visiting.add(s);
  // Axis pairs: (0,1) is up/down, (2,3) is left/right.
  for (const [d1, d2] of [
    [0, 1],
    [2, 3],
  ] as const) {
    const a = board.nbr[s * 4 + d1];
    const b = board.nbr[s * 4 + d2];
    const blocked = (q: number): boolean => {
      if (q < 0 || wall[q]) return true;
      if (!occupied[q]) return false;
      return boxFrozen(board, wall, occupied, q, visiting);
    };
    if (!blocked(a) && !blocked(b)) {
      visiting.delete(s);
      return false; // this axis can still slide one way or the other
    }
  }
  visiting.delete(s);
  return true;
}

/** True when some box off its goal is frozen solid: the position can never be solved. */
function hasFrozenBox(
  board: Board,
  wall: Uint8Array,
  occupied: Uint8Array,
  boxes: number[],
): boolean {
  for (const b of boxes) {
    if (board.goal[b]) continue;
    if (boxFrozen(board, wall, occupied, b, new Set())) return true;
  }
  return false;
}

/** The shortest walk from `from` to `to` avoiding walls and boxes, as step letters. */
function walkPath(
  board: Board,
  wall: Uint8Array,
  occupied: Uint8Array,
  from: number,
  to: number,
): string {
  if (from === to) return '';
  const { nbr, size } = board;
  const prev = new Int32Array(size).fill(-1);
  const via = new Int32Array(size).fill(-1);
  const queue = [from];
  prev[from] = from;
  for (let head = 0; head < queue.length; head++) {
    const p = queue[head];
    if (p === to) break;
    for (let d = 0; d < 4; d++) {
      const q = nbr[p * 4 + d];
      if (q < 0 || wall[q] || occupied[q] || prev[q] !== -1) continue;
      prev[q] = p;
      via[q] = d;
      queue.push(q);
    }
  }
  if (prev[to] === -1) return '';
  let out = '';
  for (let p = to; p !== from; p = prev[p]) out = STEP_LETTER[via[p]] + out;
  return out;
}

interface PushSolution {
  pushes: number;
  moves: string;
}

/**
 * A best-first search over pushes on one board. With `optimal` set it is A*
 * (cost so far plus the matching estimate) and its first solution is
 * push-optimal; without it, it is a greedy dive on the estimate alone, which
 * trades optimality for reach when the optimal search is out of budget.
 * Either way the budget is an expansion count, so the outcome is fixed.
 */
function searchPushes(
  board: Board,
  wall: Uint8Array,
  goals: number[],
  startBoxes: number[],
  startPlayer: number,
  limit: number,
  optimal: boolean,
): PushSolution | null {
  const { nbr, size, goal } = board;
  const live = liveSquares(board, wall, goals);
  const dists = goals.map((g) => pushDistances(board, wall, g));
  const nBoxes = startBoxes.length;

  // The estimate: cheapest assignment of boxes to distinct goals over the
  // push-distance table (brute force over permutations; a room holds at most
  // four boxes, and the whole board is only ever a fallback). A box that can
  // reach no goal makes the whole position hopeless.
  const estimate = (boxes: number[]): number => {
    if (nBoxes === 0) return 0;
    if (nBoxes <= 5) {
      const used = new Array<boolean>(goals.length).fill(false);
      let best = Infinity;
      const walk = (bi: number, acc: number): void => {
        if (acc >= best) return;
        if (bi === nBoxes) {
          best = acc;
          return;
        }
        for (let gi = 0; gi < goals.length; gi++) {
          if (used[gi]) continue;
          const dd = dists[gi][boxes[bi]];
          if (dd < 0) continue;
          used[gi] = true;
          walk(bi + 1, acc + dd);
          used[gi] = false;
        }
      };
      walk(0, 0);
      return best;
    }
    let sum = 0;
    for (const b of boxes) {
      let m = Infinity;
      for (let gi = 0; gi < goals.length; gi++) {
        const dd = dists[gi][b];
        if (dd >= 0 && dd < m) m = dd;
      }
      if (m === Infinity) return Infinity;
      sum += m;
    }
    return sum;
  };

  const occupied = new Uint8Array(size);
  const seen = new Int32Array(size);
  const mark = new Int32Array(size);
  const stack = new Int32Array(size);
  let stamp = 0;

  // State store: parallel arrays, states referenced by index everywhere.
  const stBoxes: number[][] = [startBoxes.slice()];
  const stPlayer: number[] = [startPlayer]; // a square the player stands on
  const stG: number[] = [0];
  const stParent: number[] = [-1];
  const stFrom: number[] = [-1]; // the pushed box's square before the push
  const stTo: number[] = [-1]; // ... and after it
  const bestG = new Map<string, number>();

  const keyOf = (boxes: number[], regionLow: number): string => {
    let k = regionLow + '|';
    for (const b of boxes) k += b + ',';
    return k;
  };

  for (const b of startBoxes) occupied[b] = 1;
  const startLow = floodRegion(board, wall, occupied, startPlayer, seen, ++stamp, stack);
  for (const b of startBoxes) occupied[b] = 0;
  bestG.set(keyOf(startBoxes, startLow), 0);
  const h0 = estimate(startBoxes);
  if (h0 === Infinity) return null;

  // Binary heap of state indices on (f, then deeper g, then older index).
  const heap: number[] = [];
  const fOf = new Map<number, number>([[0, h0]]);
  const before = (a: number, b: number): boolean => {
    const fa = fOf.get(a)!;
    const fb = fOf.get(b)!;
    if (fa !== fb) return fa < fb;
    if (stG[a] !== stG[b]) return stG[a] > stG[b];
    return a < b;
  };
  const heapPush = (s: number): void => {
    heap.push(s);
    let i = heap.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!before(heap[i], heap[p])) break;
      [heap[i], heap[p]] = [heap[p], heap[i]];
      i = p;
    }
  };
  const heapPop = (): number => {
    const top = heap[0];
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && before(heap[c + 1], heap[c])) c++;
        if (!before(heap[c], heap[i])) break;
        [heap[i], heap[c]] = [heap[c], heap[i]];
        i = c;
      }
    }
    return top;
  };
  heapPush(0);

  const allOnGoals = (boxes: number[]): boolean => boxes.every((b) => goal[b] === 1);
  if (allOnGoals(startBoxes)) return { pushes: 0, moves: '' };

  let expanded = 0;
  let found = -1;
  while (heap.length > 0 && found < 0) {
    const s = heapPop();
    const boxes = stBoxes[s];
    // Region of this state, recomputed from the stored standing square.
    for (const b of boxes) occupied[b] = 1;
    floodRegion(board, wall, occupied, stPlayer[s], seen, ++stamp, stack);
    const regionStamp = stamp;
    for (let bi = 0; bi < boxes.length && found < 0; bi++) {
      const b = boxes[bi];
      for (let d = 0; d < 4; d++) {
        const stand = nbr[b * 4 + (d ^ 1)]; // the player pushes from the opposite side
        const dest = nbr[b * 4 + d];
        if (stand < 0 || dest < 0) continue;
        if (seen[stand] !== regionStamp) continue; // the player cannot get behind the box
        if (wall[dest] || occupied[dest] || !live[dest]) continue;
        // Try the push on the occupancy map, judge it, then undo.
        occupied[b] = 0;
        occupied[dest] = 1;
        const next = boxes.slice();
        next[bi] = dest;
        next.sort((x, y) => x - y);
        const dead =
          frozenSquareBlock(board, wall, occupied, goal, dest) ||
          hasFrozenBox(board, wall, occupied, next);
        let key = '';
        let h = 0;
        if (!dead) {
          const low = floodRegion(board, wall, occupied, b, mark, ++stamp, stack);
          key = keyOf(next, low);
          const known = bestG.get(key);
          if (known !== undefined && known <= stG[s] + 1) {
            occupied[dest] = 0;
            occupied[b] = 1;
            continue;
          }
          h = estimate(next);
          if (h === Infinity) {
            occupied[dest] = 0;
            occupied[b] = 1;
            continue;
          }
        }
        occupied[dest] = 0;
        occupied[b] = 1;
        if (dead) continue;
        if (expanded >= limit) {
          for (const x of boxes) occupied[x] = 0;
          return null;
        }
        expanded++;
        const t = stBoxes.length;
        bestG.set(key, stG[s] + 1);
        stBoxes.push(next);
        stPlayer.push(b); // after the push the player stands where the box was
        stG.push(stG[s] + 1);
        stParent.push(s);
        stFrom.push(b);
        stTo.push(dest);
        fOf.set(t, optimal ? stG[t] + h : h);
        if (allOnGoals(next)) {
          found = t;
          break;
        }
        heapPush(t);
      }
    }
    for (const x of boxes) occupied[x] = 0;
  }
  if (found < 0) return null;

  // Replay the push chain, walking the player to each push's standing square.
  const chain: number[] = [];
  for (let s = found; s > 0; s = stParent[s]) chain.push(s);
  chain.reverse();
  let moves = '';
  let player = startPlayer;
  const cur = startBoxes.slice();
  for (const s of chain) {
    const from = stFrom[s];
    const to = stTo[s];
    const d = DX.map((_, i) => i).find((i) => nbr[from * 4 + i] === to)!;
    for (const b of cur) occupied[b] = 1;
    moves += walkPath(board, wall, occupied, player, nbr[from * 4 + (d ^ 1)]);
    for (const b of cur) occupied[b] = 0;
    moves += PUSH_LETTER[d];
    cur[cur.indexOf(from)] = to;
    cur.sort((x, y) => x - y);
    player = from;
  }
  return { pushes: chain.length, moves };
}

interface Room {
  squares: number[];
  boxes: number[];
  goals: number[];
}

/**
 * The level's rooms: black out the dead squares of the player's own row (the
 * hallway: a box there could never reach a goal), and each remaining part of
 * the floor that holds boxes is one room. Null when the floor does not split
 * into clean rooms (boxes and goals unbalanced in a part, a goal in the cut),
 * in which case the caller solves the level whole.
 */
function splitRooms(board: Board): Room[] | null {
  const { w, wall, goals, boxes, player, size } = board;
  const live = liveSquares(board, wall, goals);
  const rowStart = Math.floor(player / w) * w;
  const cut = new Uint8Array(size);
  for (let i = rowStart; i < rowStart + w; i++) {
    if (!wall[i] && !live[i]) cut[i] = 1;
  }
  if (boxes.some((b) => cut[b])) return null;
  const part = new Int32Array(size).fill(-1);
  const rooms: Room[] = [];
  for (let s = 0; s < size; s++) {
    if (wall[s] || cut[s] || part[s] >= 0) continue;
    const squares = [s];
    part[s] = rooms.length;
    for (let i = 0; i < squares.length; i++) {
      for (let d = 0; d < 4; d++) {
        const q = board.nbr[squares[i] * 4 + d];
        if (q >= 0 && !wall[q] && !cut[q] && part[q] < 0) {
          part[q] = rooms.length;
          squares.push(q);
        }
      }
    }
    rooms.push({ squares, boxes: [], goals: [] });
  }
  for (const b of boxes) rooms[part[b]].boxes.push(b);
  for (const g of goals) {
    if (part[g] < 0) return null;
    rooms[part[g]].goals.push(g);
  }
  if (rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

/** Where the player stands after playing `moves` from `from` (walks and pushes alike move the player one square). */
function playerAfter(board: Board, from: number, moves: string): number {
  let p = from;
  for (const ch of moves.toLowerCase()) {
    const d = STEP_LETTER.indexOf(ch);
    p = board.nbr[p * 4 + d];
  }
  return p;
}

/**
 * Solve a level room by room: each room's boxes are searched with every other
 * box on the board standing as a wall, and the player walks on from wherever
 * the previous room's answer left them. Because no box can leave its room,
 * the rooms' push optima add up to the level's.
 */
function solveByRooms(board: Board): PushSolution | null {
  const rooms = splitRooms(board);
  if (rooms === null) return null;
  let player = board.player;
  let pushes = 0;
  let moves = '';
  const currentBoxes = board.boxes.slice();
  for (const room of rooms) {
    const wall = board.wall.slice();
    for (const b of currentBoxes) if (!room.boxes.includes(b)) wall[b] = 1;
    const roomBoxes = room.boxes.slice().sort((a, b) => a - b);
    const solved =
      searchPushes(board, wall, room.goals, roomBoxes, player, ROOM_ASTAR_LIMIT, true) ??
      searchPushes(board, wall, room.goals, roomBoxes, player, ROOM_GREEDY_LIMIT, false);
    if (solved === null) return null;
    // Apply just this room's moves to the shared box list, from the state
    // the level was in when the room's turn began.
    const replay = replayBoxes(board, currentBoxes, player, solved.moves);
    currentBoxes.length = 0;
    currentBoxes.push(...replay);
    player = playerAfter(board, player, solved.moves);
    pushes += solved.pushes;
    moves += solved.moves;
  }
  return { pushes, moves };
}

/** The box squares after playing a whole move string from the level's start. */
function replayBoxes(board: Board, startBoxes: number[], startPlayer: number, moves: string): number[] {
  const occupied = new Uint8Array(board.size);
  for (const b of startBoxes) occupied[b] = 1;
  let player = startPlayer;
  const boxes = startBoxes.slice();
  for (const ch of moves.toLowerCase()) {
    const d = STEP_LETTER.indexOf(ch);
    const next = board.nbr[player * 4 + d];
    if (next >= 0 && occupied[next]) {
      const beyond = board.nbr[next * 4 + d];
      occupied[next] = 0;
      occupied[beyond] = 1;
      boxes[boxes.indexOf(next)] = beyond;
    }
    player = next;
  }
  boxes.sort((a, b) => a - b);
  return boxes;
}

/**
 * Last resorts for a level that does not split into rooms: an optimal push
 * search over the whole board, then a plain step-by-step A* in the spirit of
 * a textbook solver (fewest steps, any solution beats none).
 */
function solveWhole(board: Board): PushSolution | null {
  const byPushes = searchPushes(board, board.wall, board.goals, board.boxes, board.player, WHOLE_ASTAR_LIMIT, true);
  if (byPushes !== null) return byPushes;
  const moves = solveBySteps(board);
  if (moves === '') return null;
  let pushes = 0;
  const occupied = new Uint8Array(board.size);
  for (const b of board.boxes) occupied[b] = 1;
  let player = board.player;
  for (const ch of moves.toLowerCase()) {
    const d = STEP_LETTER.indexOf(ch);
    const next = board.nbr[player * 4 + d];
    if (occupied[next]) pushes++;
    player = next;
  }
  return { pushes, moves };
}

/** A step-level A*: fewest steps to a solved board, Manhattan estimate, no deadlock sense. */
function solveBySteps(board: Board): string {
  const { nbr, wall, goal, goals } = board;
  const dist = (b: number): number => {
    let m = Infinity;
    const bx = b % board.w;
    const by = Math.floor(b / board.w);
    for (const g of goals) {
      const d = Math.abs((g % board.w) - bx) + Math.abs(Math.floor(g / board.w) - by);
      if (d < m) m = d;
    }
    return m;
  };
  const est = (bs: number[]): number => bs.reduce((a, b) => a + dist(b), 0);
  const key = (p: number, bs: number[]): string => p + '|' + bs.join(',');
  const sp: number[] = [board.player];
  const sb: number[][] = [board.boxes.slice()];
  const sg: number[] = [0];
  const parent: number[] = [-1];
  const step: string[] = [''];
  const best = new Map<string, number>([[key(board.player, board.boxes), 0]]);
  const heap: [number, number][] = [[est(board.boxes), 0]];
  const before = (a: [number, number], b: [number, number]): boolean =>
    a[0] < b[0] || (a[0] === b[0] && a[1] < b[1]);
  const push = (e: [number, number]): void => {
    heap.push(e);
    let i = heap.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!before(heap[i], heap[p])) break;
      [heap[i], heap[p]] = [heap[p], heap[i]];
      i = p;
    }
  };
  const pop = (): [number, number] => {
    const top = heap[0];
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && before(heap[c + 1], heap[c])) c++;
        if (!before(heap[c], heap[i])) break;
        [heap[i], heap[c]] = [heap[c], heap[i]];
        i = c;
      }
    }
    return top;
  };
  let expanded = 0;
  while (heap.length > 0 && expanded < WHOLE_MOVE_LIMIT) {
    const [, s] = pop();
    const bs = sb[s];
    if (best.get(key(sp[s], bs)) !== s) continue;
    if (bs.every((b) => goal[b] === 1)) {
      let moves = '';
      for (let t = s; t > 0; t = parent[t]) moves = step[t] + moves;
      return moves;
    }
    expanded++;
    const p = sp[s];
    for (let d = 0; d < 4; d++) {
      const q = nbr[p * 4 + d];
      if (q < 0 || wall[q]) continue;
      let nb = bs;
      let letter = STEP_LETTER[d];
      const at = bs.indexOf(q);
      if (at >= 0) {
        const beyond = nbr[q * 4 + d];
        if (beyond < 0 || wall[beyond] || bs.includes(beyond)) continue;
        nb = bs.slice();
        nb[at] = beyond;
        nb.sort((a, b) => a - b);
        letter = PUSH_LETTER[d];
      }
      const k = key(q, nb);
      const seenAt = best.get(k);
      if (seenAt !== undefined && sg[seenAt] <= sg[s] + 1) continue;
      const t = sp.length;
      sp.push(q);
      sb.push(nb);
      sg.push(sg[s] + 1);
      parent.push(s);
      step.push(letter);
      best.set(k, t);
      push([sg[t] + est(nb), t]);
    }
  }
  return '';
}

/** The moves for one level: room by room when it splits, whole-board fallbacks when it does not. */
function solveLevel(text: string): string {
  const board = parseLevel(text);
  if (board.boxes.length === 0) return '';
  const solved = solveByRooms(board) ?? solveWhole(board);
  return solved === null ? '' : solved.moves;
}

for (const id of readdirSync(INPUT_DIR).sort()) {
  const moves = solveLevel(readFileSync(join(INPUT_DIR, id, 'level.txt'), 'utf8'));
  mkdirSync(join(OUTPUT_DIR, id), { recursive: true });
  writeFileSync(join(OUTPUT_DIR, id, 'solution.txt'), `${moves}\n`);
}

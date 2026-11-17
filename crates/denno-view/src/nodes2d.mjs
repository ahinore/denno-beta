// denno-view /2d のノードツリーと、ノードを選ぶ DSL (DOM 非依存)。
//
// ツリー: モデルの構成要素を 1 つずつノードにする。葉のうち介入できるもの
// (expert / mixer / moe / shared / ple / stream) は `item` を持ち、これが
// /v1/intervene の選択項目 (view2d.mjs の selection) になる。グループのノードを
// 選ぶと、その下の item を全部選んだことになる。
//
//   model
//   ├ Embedding (vocab / N-gram, 情報のみ)
//   ├ Hybrid block h (層 4h..4h+3)
//   │ └ L07 GDN / L03 QSA
//   │   ├ Expanded residual ─ stream 0..n_hc-1
//   │   ├ Gated DeltaNet | Qwen Sparse Attn          (block attn)
//   │   ├ MoE ─ router (block moe) / shared expert (block shared)
//   │   │      └ routed experts ─ expert 0..n_expert-1
//   │   └ PLE in (N-gram)                             (block ple、ple_layers だけ)
//   └ Output (final GR read / prediction head / MTP, 情報のみ)
//
// DSL (ノード選択式) の文法:
//
//   program := query { ';' query }                 結果は和集合
//   query   := [ types ':' ] [ expr ] { '|' stage }
//   types   := '*' | type { ',' type }             省略時は介入できる葉の全種類
//   type    := expert | mixer | moe | shared | ple | stream | block
//            | layer | hybrid | residual | ffn | experts
//   stage   := ('top' | 'bottom') INT [ 'by' expr ] [ 'per' ('layer' | 'hybrid') ]
//            | 'limit' INT
//   expr    := or ;  or := and { ('or' | '||') and } ;  and := not { ('and' | '&&') not }
//   not     := ('not' | '!') not | cmp
//   cmp     := sum { ('==' | '=' | '!=' | '<' | '<=' | '>' | '>=') sum }   連鎖可: 0 <= layer < 8
//            | sum 'in' ( sum '..' sum | '[' expr { ',' expr } ']' )
//   sum     := prod { ('+' | '-') prod } ;  prod := unary { ('*' | '/' | '%') unary }
//   unary   := '-' unary | pow ;  pow := atom [ '^' unary ]
//   atom    := NUMBER | STRING | IDENT | IDENT '(' args ')' | '(' expr ')'
//
// 値 (ノードの属性。無いものは 0 / ""):
//   layer, hybrid (= layer の属するハイブリッドブロック), pos (= ブロック内の位置 0..3),
//   index (expert / stream 番号), kind ("gdn" | "qsa"), type,
//   count (リセットからの活性化回数), rate (= count / tokens), rank (層内の count 順位, 1 始まり),
//   act / w (最後のトークンの値 0..1), acc (decay 付き累積), topk, off, sel, vram, skip (0/1)
// 定数: tokens, layers, n_expert, n_used, n_hc, true, false
// 関数: abs min max sqrt log log2 exp floor ceil round clamp(x,lo,hi)
//       集約 lmax/lmin/lmean/lsum(expr) — 同じ層・同じ種類のノードで評価
//            gmax/gmin/gmean/gsum(expr) — モデル全体の同じ種類のノードで評価
//
// 例:
//   expert: count == 0 and layer >= 40           一度も選ばれていない後段の expert
//   expert: count >= 0.5 * lmax(count)           層内最多の半分以上選ばれた expert
//   expert: rate > 0.2 | top 5 by count per layer
//   mixer: kind == "qsa" and layer % 8 == 3
//   stream: index == 0 and layer in 0..11; moe: layer in [5, 9, 17]

import { normaliseShape, roleForBlock, streamValue } from './view2d.mjs';

export const LEAF_TYPES = Object.freeze(['expert', 'mixer', 'moe', 'shared', 'ple', 'stream']);
export const GROUP_TYPES = Object.freeze(['layer', 'hybrid', 'residual', 'ffn', 'experts']);
export const HYBRID_SIZE = 4;

// ---- ツリー -----------------------------------------------------------------

function layerName(l, kind) {
  return 'L' + String(l).padStart(2, '0') + ' ' + (kind === 'attn' ? 'QSA' : 'GDN');
}

/// 形 → ツリー。返り値:
///   root, nodes (先行順), byId, items (item を持つノード, 先行順)
/// 各ノード: {id, type, label, parent, children|null, depth, layer, kind, index, item,
///            lo, hi}  — items[lo..hi) がこのノードの下 (自身を含む) の item。
export function buildTree(shape) {
  const s = normaliseShape(shape);
  const nodes = [];
  const byId = new Map();
  const mk = (o, parent) => {
    const n = Object.assign({
      parent: parent || null, children: null, depth: parent ? parent.depth + 1 : 0,
      layer: parent ? parent.layer : -1, kind: parent ? parent.kind : '',
      index: -1, item: null, lo: 0, hi: 0,
    }, o);
    if (parent) (parent.children || (parent.children = [])).push(n);
    nodes.push(n);
    byId.set(n.id, n);
    return n;
  };

  const root = mk({ id: 'model', type: 'model', label: 'model · ' + s.layers + ' layers' }, null);
  const emb = mk({ id: 'emb', type: 'group', label: 'Embedding' }, root);
  mk({ id: 'vocab', type: 'info', label: 'Vocabulary embedding' }, emb);
  mk({
    id: 'ngram', type: 'info',
    label: 'N-gram embedding → ' + (s.ple_layers.length ? 'layer ' + s.ple_layers.join(',') : '-'),
  }, emb);

  const nh = Math.ceil(s.layers / HYBRID_SIZE);
  for (let h = 0; h < nh; h++) {
    const a = h * HYBRID_SIZE;
    const b = Math.min(s.layers, a + HYBRID_SIZE) - 1;
    const hy = mk({ id: 'H' + h, type: 'hybrid', hybrid: h, label: 'Hybrid block ' + h + ' · L' + a + '–' + b }, root);
    for (let l = a; l <= b; l++) {
      const kind = s.layer_kinds[l];
      const L = mk({ id: 'L' + l, type: 'layer', layer: l, kind, label: layerName(l, kind) }, hy);
      const res = mk({ id: 'L' + l + '/res', type: 'residual', label: 'Expanded residual' }, L);
      for (let i = 0; i < s.n_hc; i++) {
        mk({
          id: 'L' + l + '/s' + i, type: 'stream', index: i, label: 'stream ' + i,
          item: { type: 'stream', layer: l, stream: i },
        }, res);
      }
      mk({
        id: 'L' + l + '/mix', type: 'mixer',
        label: kind === 'attn' ? 'Qwen Sparse Attn' : 'Gated DeltaNet',
        item: { type: 'block', layer: l, block: 'attn' },
      }, L);
      const ffn = mk({ id: 'L' + l + '/ffn', type: 'ffn', label: 'MoE' }, L);
      mk({
        id: 'L' + l + '/moe', type: 'moe', label: 'router (all routed experts)',
        item: { type: 'block', layer: l, block: 'moe' },
      }, ffn);
      mk({
        id: 'L' + l + '/sh', type: 'shared', label: 'shared expert',
        item: { type: 'block', layer: l, block: 'shared' },
      }, ffn);
      const ex = mk({ id: 'L' + l + '/ex', type: 'experts', label: 'routed experts (' + s.n_expert + ')' }, ffn);
      for (let e = 0; e < s.n_expert; e++) {
        mk({
          id: 'L' + l + '/e' + e, type: 'expert', index: e, label: 'expert ' + e,
          item: { type: 'expert', layer: l, expert: e },
        }, ex);
      }
      if (s.ple_layers.indexOf(l) >= 0) {
        mk({
          id: 'L' + l + '/ple', type: 'ple', label: 'PLE in (N-gram)',
          item: { type: 'block', layer: l, block: 'ple' },
        }, L);
      }
    }
  }
  const out = mk({ id: 'out', type: 'group', label: 'Output' }, root);
  mk({ id: 'grfinal', type: 'info', label: 'final GR read' }, out);
  mk({ id: 'head', type: 'info', label: 'Prediction head' }, out);
  mk({ id: 'mtp', type: 'info', label: 'MTP modules' }, out);

  // items[lo..hi) を先行順で振る。
  const items = [];
  const walk = (n) => {
    n.lo = items.length;
    if (n.item) items.push(n);
    if (n.children) for (const c of n.children) walk(c);
    n.hi = items.length;
  };
  walk(root);
  return { shape: s, root, nodes, byId, items };
}

/// 開いているノードだけを並べた行 (仮想スクロール用)。root 自身は含めない。
export function visibleRows(tree, expanded) {
  const out = [];
  const walk = (n) => {
    out.push(n);
    if (n.children && expanded.has(n.id)) for (const c of n.children) walk(c);
  };
  for (const c of tree.root.children) walk(c);
  return out;
}

/// ノードの祖先 id (近い順ではなく root 側から)。
export function ancestorIds(node) {
  const out = [];
  for (let p = node.parent; p; p = p.parent) out.unshift(p.id);
  return out;
}

/// ノード → 選択項目の配列 (グループは下の全 item)。
export function nodeItems(tree, node) {
  const out = [];
  for (let i = node.lo; i < node.hi; i++) out.push(tree.items[i].item);
  return out;
}

/// ノードの選択状態: 'none' | 'some' | 'all' | 'na' (item が無い)。
/// `has(item)` は呼び出し側の判定 (view2d.mjs の hasItem を部分適用したもの)。
export function nodeCheckState(tree, node, has) {
  const n = node.hi - node.lo;
  if (n === 0) return 'na';
  let k = 0;
  for (let i = node.lo; i < node.hi; i++) if (has(tree.items[i].item)) k++;
  return k === 0 ? 'none' : k === n ? 'all' : 'some';
}

// ---- 属性 -----------------------------------------------------------------

/// DSL とツリー表示が使う値の出どころ。
///   data = { st (createMapState), off: {experts_off, blocks_off, streams_off}, has(item) }
export function createEnv(tree, data) {
  const s = tree.shape;
  const st = data && data.st;
  const off = (data && data.off) || {};
  const offE = new Set((off.experts_off || []).map((p) => p[0] + ':' + p[1]));
  const offB = new Set((off.blocks_off || []).map((p) => p[0] + ':' + p[1]));
  const offS = new Set((off.streams_off || []).map((p) => p[0] + ':' + p[1]));
  const has = (data && data.has) || (() => false);
  const tokens = st ? st.token | 0 : 0;
  const role = (l) => (st && st.role && st.role[l]) || {};
  const rcount = (l) => (st && st.roleCount && st.roleCount[l]) || {};
  const rankCache = new Map();

  const env = {
    tree, shape: s, tokens,
    count(n) {
      const l = n.layer;
      switch (n.type) {
        case 'expert': return st && st.count && st.count[l] ? st.count[l][n.index] : 0;
        case 'mixer': return rcount(l)[n.kind === 'attn' ? 'attn' : 'gdn'] || 0;
        case 'moe': return rcount(l).router || 0;
        case 'shared': return rcount(l).shex || 0;
        case 'ple': return rcount(l).ple || 0;
        case 'stream': case 'residual': case 'layer': return rcount(l).any || 0;
        case 'ffn': return rcount(l).router || 0;
        case 'experts': {
          let t = 0;
          const a = st && st.count && st.count[l];
          if (a) for (let i = 0; i < a.length; i++) t += a[i];
          return t;
        }
        case 'hybrid': {
          let m = 0;
          for (const c of n.children || []) m = Math.max(m, env.count(c));
          return m;
        }
        default: return 0;
      }
    },
    act(n) {
      const l = n.layer;
      switch (n.type) {
        case 'expert': return st && st.last && st.last[l] ? st.last[l][n.index] : 0;
        case 'mixer': return roleForBlock(role(l), 'attn', n.kind);
        case 'moe': case 'ffn': return role(l).router || 0;
        case 'shared': return role(l).shex || 0;
        case 'ple': return role(l).ple || 0;
        case 'stream': case 'residual': case 'layer': return streamValue(role(l));
        case 'experts': {
          const a = st && st.last && st.last[l];
          let m = 0;
          if (a) for (let i = 0; i < a.length; i++) if (a[i] > m) m = a[i];
          return m;
        }
        case 'hybrid': {
          let m = 0;
          for (const c of n.children || []) m = Math.max(m, env.act(c));
          return m;
        }
        default: return 0;
      }
    },
    acc(n) {
      if (n.type === 'expert') return st && st.accum && st.accum[n.layer] ? st.accum[n.layer][n.index] : 0;
      return env.act(n);
    },
    rank(n) {
      if (n.type !== 'expert') return 0;
      let r = rankCache.get(n.layer);
      if (!r) {
        const a = (st && st.count && st.count[n.layer]) || new Uint32Array(s.n_expert);
        const idx = Array.from({ length: a.length }, (_, i) => i).sort((x, y) => a[y] - a[x] || x - y);
        r = new Uint32Array(a.length);
        // 同点は同順位 (1, 2, 2, 4 …)
        for (let k = 0; k < idx.length; k++) {
          r[idx[k]] = k > 0 && a[idx[k]] === a[idx[k - 1]] ? r[idx[k - 1]] : k + 1;
        }
        rankCache.set(n.layer, r);
      }
      return r[n.index];
    },
    flag(n, name) {
      const l = n.layer;
      if (name === 'sel') {
        if (n.item) return has(n.item) ? 1 : 0;
        if (n.hi === n.lo) return 0;
        for (let i = n.lo; i < n.hi; i++) if (!has(tree.items[i].item)) return 0;
        return 1;
      }
      if (name === 'off') {
        if (n.type === 'expert') return offE.has(l + ':' + n.index) ? 1 : 0;
        if (n.type === 'stream') return offS.has(l + ':' + n.index) ? 1 : 0;
        if (n.item && n.item.type === 'block') return offB.has(l + ':' + n.item.block) ? 1 : 0;
        return 0;
      }
      if (n.type !== 'expert' || !st) return 0;
      if (name === 'topk') return st.topk && st.topk[l] ? st.topk[l][n.index] : 0;
      if (name === 'vram') return st.residency && st.residency[l] ? st.residency[l][n.index] : 0;
      if (name === 'skip') return st.skip && st.skip[l] ? st.skip[l][n.index] : 0;
      return 0;
    },
  };
  return env;
}

const ATTRS = Object.freeze({
  layer: (n) => n.layer,
  hybrid: (n) => (n.layer >= 0 ? Math.floor(n.layer / HYBRID_SIZE) : (n.hybrid != null ? n.hybrid : -1)),
  pos: (n) => (n.layer >= 0 ? n.layer % HYBRID_SIZE : -1),
  index: (n) => n.index,
  i: (n) => n.index,
  kind: (n) => (n.kind === 'attn' ? 'qsa' : n.kind === 'gdn' ? 'gdn' : ''),
  type: (n) => n.type,
  count: (n, env) => env.count(n),
  rate: (n, env) => (env.tokens > 0 ? env.count(n) / env.tokens : 0),
  rank: (n, env) => env.rank(n),
  act: (n, env) => env.act(n),
  w: (n, env) => env.act(n),
  acc: (n, env) => env.acc(n),
  topk: (n, env) => env.flag(n, 'topk'),
  off: (n, env) => env.flag(n, 'off'),
  sel: (n, env) => env.flag(n, 'sel'),
  vram: (n, env) => env.flag(n, 'vram'),
  skip: (n, env) => env.flag(n, 'skip'),
  tokens: (n, env) => env.tokens,
  layers: (n, env) => env.shape.layers,
  n_expert: (n, env) => env.shape.n_expert,
  n_used: (n, env) => env.shape.n_expert_used,
  n_hc: (n, env) => env.shape.n_hc,
  true: () => 1,
  false: () => 0,
});

export const DSL_ATTRS = Object.freeze(Object.keys(ATTRS));

const FUNCS = Object.freeze({
  abs: [1, Math.abs], sqrt: [1, Math.sqrt], log: [1, Math.log], log2: [1, Math.log2],
  exp: [1, Math.exp], floor: [1, Math.floor], ceil: [1, Math.ceil], round: [1, Math.round],
  min: [-2, Math.min], max: [-2, Math.max],
  clamp: [3, (x, a, b) => Math.max(a, Math.min(b, x))],
});
const AGGS = Object.freeze(['lmax', 'lmin', 'lmean', 'lsum', 'gmax', 'gmin', 'gmean', 'gsum']);

export const DSL_FUNCS = Object.freeze([...Object.keys(FUNCS), ...AGGS]);

const TYPE_ALIASES = Object.freeze({
  '*': LEAF_TYPES,
  block: ['mixer', 'moe', 'shared', 'ple'],
});
const ALL_TYPES = new Set([...LEAF_TYPES, ...GROUP_TYPES]);

// ---- 字句 -----------------------------------------------------------------

export class DslError extends Error {
  constructor(message, pos) {
    super(message);
    this.name = 'DslError';
    this.pos = pos | 0;
  }
}

const OPS = ['||', '&&', '==', '!=', '<=', '>=', '..', '<', '>', '=', '+', '-', '*', '/', '%', '^',
  '!', '(', ')', '[', ']', ',', ':', ';', '|'];

export function tokenize(src) {
  const s = String(src == null ? '' : src);
  const out = [];
  let i = 0;
  while (i < s.length) {
    const c = s[i];
    if (c === ' ' || c === '\t' || c === '\n' || c === '\r') { i++; continue; }
    if (c === '#') { while (i < s.length && s[i] !== '\n') i++; continue; }
    const start = i;
    if (/[0-9]/.test(c) || (c === '.' && /[0-9]/.test(s[i + 1] || '') && s[i - 1] !== '.')) {
      // 1..3 の '..' を小数点と取り違えない
      let j = i;
      while (j < s.length && /[0-9]/.test(s[j])) j++;
      if (s[j] === '.' && s[j + 1] !== '.') {
        j++;
        while (j < s.length && /[0-9]/.test(s[j])) j++;
      }
      if ((s[j] === 'e' || s[j] === 'E') && /[-+0-9]/.test(s[j + 1] || '')) {
        let k = j + 1;
        if (s[k] === '+' || s[k] === '-') k++;
        if (/[0-9]/.test(s[k] || '')) {
          while (k < s.length && /[0-9]/.test(s[k])) k++;
          j = k;
        }
      }
      out.push({ t: 'num', v: Number(s.slice(i, j)), pos: start });
      i = j;
      continue;
    }
    if (/[A-Za-z_]/.test(c)) {
      let j = i + 1;
      while (j < s.length && /[A-Za-z0-9_]/.test(s[j])) j++;
      out.push({ t: 'id', v: s.slice(i, j), pos: start });
      i = j;
      continue;
    }
    if (c === '"' || c === "'") {
      let j = i + 1;
      let v = '';
      while (j < s.length && s[j] !== c) {
        if (s[j] === '\\' && j + 1 < s.length) { v += s[j + 1]; j += 2; continue; }
        v += s[j++];
      }
      if (j >= s.length) throw new DslError('文字列が閉じていない', start);
      out.push({ t: 'str', v, pos: start });
      i = j + 1;
      continue;
    }
    const op = OPS.find((o) => s.startsWith(o, i));
    if (!op) throw new DslError('使えない文字: ' + c, start);
    out.push({ t: 'op', v: op, pos: start });
    i += op.length;
  }
  out.push({ t: 'end', v: '', pos: s.length });
  return out;
}

// ---- 構文 -----------------------------------------------------------------

const KEYWORDS = new Set(['and', 'or', 'not', 'in', 'top', 'bottom', 'by', 'per', 'limit']);

/// 文字列 → AST。{ queries: [{types, expr|null, stages, pos}] }
export function parseDsl(src) {
  const toks = tokenize(src);
  let p = 0;
  const peek = (k = 0) => toks[Math.min(p + k, toks.length - 1)];
  const isOp = (v, k = 0) => peek(k).t === 'op' && peek(k).v === v;
  const isKw = (v, k = 0) => peek(k).t === 'id' && peek(k).v === v;
  const next = () => toks[p++];
  const expectOp = (v) => {
    if (!isOp(v)) throw new DslError("'" + v + "' が要る", peek().pos);
    return next();
  };
  const expectInt = (what) => {
    const t = peek();
    if (t.t !== 'num' || !Number.isInteger(t.v) || t.v < 0) throw new DslError(what + ' には 0 以上の整数が要る', t.pos);
    next();
    return t.v;
  };

  function parseTypes() {
    // 先読み: (type {',' type} | '*') ':' のときだけ型の並びとみなす
    let k = 0;
    if (isOp('*', 0) && isOp(':', 1)) {
      next(); next();
      return { names: ['*'], types: LEAF_TYPES.slice() };
    }
    for (;;) {
      if (peek(k).t !== 'id') return null;
      if (isOp(',', k + 1)) { k += 2; continue; }
      if (isOp(':', k + 1)) break;
      return null;
    }
    const names = [];
    const types = new Set();
    for (;;) {
      const t = next();
      const name = t.v;
      if (TYPE_ALIASES[name]) for (const x of TYPE_ALIASES[name]) types.add(x);
      else if (ALL_TYPES.has(name)) types.add(name);
      else {
        throw new DslError('知らない種類: ' + name + ' (使えるもの: ' +
          [...LEAF_TYPES, 'block', ...GROUP_TYPES, '*'].join(', ') + ')', t.pos);
      }
      names.push(name);
      if (isOp(',')) { next(); continue; }
      expectOp(':');
      break;
    }
    return { names, types: [...types] };
  }

  function parseExpr() { return parseOr(); }
  function parseOr() {
    let a = parseAnd();
    while (isOp('||') || isKw('or')) {
      const t = next();
      a = { t: 'or', a, b: parseAnd(), pos: t.pos };
    }
    return a;
  }
  function parseAnd() {
    let a = parseNot();
    while (isOp('&&') || isKw('and')) {
      const t = next();
      a = { t: 'and', a, b: parseNot(), pos: t.pos };
    }
    return a;
  }
  function parseNot() {
    if (isOp('!') || isKw('not')) {
      const t = next();
      return { t: 'not', a: parseNot(), pos: t.pos };
    }
    return parseCmp();
  }
  const CMP = ['==', '=', '!=', '<', '<=', '>', '>='];
  function parseCmp() {
    const first = parseSum();
    if (isKw('in')) {
      const t = next();
      if (isOp('[')) {
        next();
        const list = [];
        if (!isOp(']')) {
          for (;;) {
            list.push(parseExpr());
            if (isOp(',')) { next(); continue; }
            break;
          }
        }
        expectOp(']');
        return { t: 'inlist', a: first, list, pos: t.pos };
      }
      const lo = parseSum();
      expectOp('..');
      const hi = parseSum();
      return { t: 'inrange', a: first, lo, hi, pos: t.pos };
    }
    if (peek().t !== 'op' || CMP.indexOf(peek().v) < 0) return first;
    const args = [first];
    const ops = [];
    while (peek().t === 'op' && CMP.indexOf(peek().v) >= 0) {
      const t = next();
      ops.push(t.v === '=' ? '==' : t.v);
      args.push(parseSum());
    }
    return { t: 'cmp', ops, args, pos: first.pos };
  }
  function parseSum() {
    let a = parseProd();
    while (isOp('+') || isOp('-')) {
      const t = next();
      a = { t: 'bin', op: t.v, a, b: parseProd(), pos: t.pos };
    }
    return a;
  }
  function parseProd() {
    let a = parseUnary();
    while (isOp('*') || isOp('/') || isOp('%')) {
      const t = next();
      a = { t: 'bin', op: t.v, a, b: parseUnary(), pos: t.pos };
    }
    return a;
  }
  function parseUnary() {
    if (isOp('-')) {
      const t = next();
      return { t: 'neg', a: parseUnary(), pos: t.pos };
    }
    if (isOp('+')) { next(); return parseUnary(); }
    return parsePow();
  }
  function parsePow() {
    const a = parseAtom();
    if (isOp('^')) {
      const t = next();
      return { t: 'bin', op: '^', a, b: parseUnary(), pos: t.pos };
    }
    return a;
  }
  function parseAtom() {
    const t = peek();
    if (t.t === 'num') { next(); return { t: 'num', v: t.v, pos: t.pos }; }
    if (t.t === 'str') { next(); return { t: 'str', v: t.v, pos: t.pos }; }
    if (isOp('(')) {
      next();
      const e = parseExpr();
      expectOp(')');
      return e;
    }
    if (t.t === 'id') {
      if (KEYWORDS.has(t.v)) throw new DslError("'" + t.v + "' はここに書けない", t.pos);
      next();
      if (isOp('(')) {
        next();
        const args = [];
        if (!isOp(')')) {
          for (;;) {
            args.push(parseExpr());
            if (isOp(',')) { next(); continue; }
            break;
          }
        }
        expectOp(')');
        if (AGGS.indexOf(t.v) >= 0) {
          if (args.length !== 1) throw new DslError(t.v + '() は引数 1 つ', t.pos);
          return { t: 'agg', name: t.v, a: args[0], pos: t.pos };
        }
        const f = FUNCS[t.v];
        if (!f) throw new DslError('知らない関数: ' + t.v + ' (使えるもの: ' + DSL_FUNCS.join(', ') + ')', t.pos);
        if (f[0] >= 0 && args.length !== f[0]) throw new DslError(t.v + '() は引数 ' + f[0] + ' つ', t.pos);
        if (f[0] < 0 && args.length < -f[0]) throw new DslError(t.v + '() は引数 ' + (-f[0]) + ' つ以上', t.pos);
        return { t: 'call', name: t.v, args, pos: t.pos };
      }
      if (!Object.prototype.hasOwnProperty.call(ATTRS, t.v)) {
        throw new DslError('知らない名前: ' + t.v + ' (使えるもの: ' + DSL_ATTRS.join(', ') + ')', t.pos);
      }
      return { t: 'id', name: t.v, pos: t.pos };
    }
    if (t.t === 'end') throw new DslError('式が途中で終わっている', t.pos);
    throw new DslError("'" + t.v + "' はここに書けない", t.pos);
  }

  function parseStage() {
    const t = peek();
    if (isKw('top') || isKw('bottom')) {
      next();
      const n = expectInt(t.v);
      let by = null;
      let per = null;
      if (isKw('by')) { next(); by = parseExpr(); }
      if (isKw('per')) {
        next();
        const g = peek();
        if (!(isKw('layer') || isKw('hybrid'))) throw new DslError("per の後は layer か hybrid", g.pos);
        next();
        per = g.v;
      }
      return { op: t.v, n, by: by || { t: 'id', name: 'count', pos: t.pos }, per, pos: t.pos };
    }
    if (isKw('limit')) {
      next();
      return { op: 'limit', n: expectInt('limit'), pos: t.pos };
    }
    throw new DslError("'|' の後は top / bottom / limit", t.pos);
  }

  function parseQuery() {
    const pos = peek().pos;
    const ty = parseTypes();
    let expr = null;
    if (!(peek().t === 'end' || isOp(';') || isOp('|'))) expr = parseExpr();
    const stages = [];
    while (isOp('|')) {
      next();
      stages.push(parseStage());
    }
    if (!(peek().t === 'end' || isOp(';'))) {
      throw new DslError("'" + peek().v + "' はここに書けない (and / or で繋ぐか ';' で区切る)", peek().pos);
    }
    return {
      types: ty ? ty.types : LEAF_TYPES.slice(),
      typeNames: ty ? ty.names : [],
      expr, stages, pos,
    };
  }

  const queries = [];
  for (;;) {
    while (isOp(';')) next();
    if (peek().t === 'end') break;
    queries.push(parseQuery());
  }
  if (!queries.length) throw new DslError('式が空', 0);
  return { queries };
}

// ---- 評価 -----------------------------------------------------------------

function truthy(v) {
  if (typeof v === 'string') return v.length > 0;
  return Number(v) !== 0 && !Number.isNaN(Number(v));
}

function num(v, pos) {
  if (typeof v === 'string') throw new DslError('文字列で計算はできない: "' + v + '"', pos);
  return Number(v);
}

function eq(a, b) {
  if (typeof a === 'string' || typeof b === 'string') return String(a) === String(b);
  return Number(a) === Number(b);
}

function compare(op, a, b, pos) {
  if (op === '==') return eq(a, b);
  if (op === '!=') return !eq(a, b);
  const x = num(a, pos), y = num(b, pos);
  if (op === '<') return x < y;
  if (op === '<=') return x <= y;
  if (op === '>') return x > y;
  return x >= y;
}

function makeEvaluator(env, peersOf) {
  const aggCache = new Map();   // ast -> Map(groupKey -> value)
  const ev = (a, n) => {
    switch (a.t) {
      case 'num': return a.v;
      case 'str': return a.v;
      case 'id': return ATTRS[a.name](n, env);
      case 'neg': return -num(ev(a.a, n), a.pos);
      case 'not': return truthy(ev(a.a, n)) ? 0 : 1;
      case 'and': return truthy(ev(a.a, n)) && truthy(ev(a.b, n)) ? 1 : 0;
      case 'or': return truthy(ev(a.a, n)) || truthy(ev(a.b, n)) ? 1 : 0;
      case 'bin': {
        const x = num(ev(a.a, n), a.pos), y = num(ev(a.b, n), a.pos);
        switch (a.op) {
          case '+': return x + y;
          case '-': return x - y;
          case '*': return x * y;
          case '/': return y === 0 ? 0 : x / y;          // 0 割りは 0 (count / 0 が多いので)
          case '%': return y === 0 ? 0 : ((x % y) + y) % y;
          case '^': return Math.pow(x, y);
          default: throw new DslError('bad op ' + a.op, a.pos);
        }
      }
      case 'cmp': {
        let left = ev(a.args[0], n);
        for (let i = 0; i < a.ops.length; i++) {
          const right = ev(a.args[i + 1], n);
          if (!compare(a.ops[i], left, right, a.pos)) return 0;
          left = right;
        }
        return 1;
      }
      case 'inrange': {
        const x = num(ev(a.a, n), a.pos);
        return x >= num(ev(a.lo, n), a.pos) && x <= num(ev(a.hi, n), a.pos) ? 1 : 0;
      }
      case 'inlist': {
        const x = ev(a.a, n);
        for (const e of a.list) if (eq(x, ev(e, n))) return 1;
        return 0;
      }
      case 'call': {
        const f = FUNCS[a.name][1];
        return f(...a.args.map((e) => num(ev(e, n), e.pos)));
      }
      case 'agg': {
        const global = a.name[0] === 'g';
        const key = n.type + '|' + (global ? '*' : n.layer);
        let m = aggCache.get(a);
        if (!m) { m = new Map(); aggCache.set(a, m); }
        if (m.has(key)) return m.get(key);
        const peers = peersOf(n, global);
        const op = a.name.slice(1);
        let v = op === 'max' ? -Infinity : op === 'min' ? Infinity : 0;
        for (const q of peers) {
          const x = num(ev(a.a, q), a.pos);
          if (op === 'max') v = Math.max(v, x);
          else if (op === 'min') v = Math.min(v, x);
          else v += x;
        }
        if (op === 'mean') v = peers.length ? v / peers.length : 0;
        if (!Number.isFinite(v)) v = 0;
        m.set(key, v);
        return v;
      }
      default: throw new DslError('bad node ' + a.t, a.pos);
    }
  };
  return ev;
}

/// DSL を評価して、合うノードを返す。
///   返り値 { nodes, items, byType, ast }
///     nodes — 合ったノード (先行順。重複なし)
///     items — それらの選択項目 (グループは下の全 item。重複なし)
export function runDsl(src, tree, data) {
  const ast = typeof src === 'string' ? parseDsl(src) : src;
  const env = createEnv(tree, data);
  // 種類ごとの全ノード (と層ごと) を 1 回だけ作る
  const byTypeAll = new Map();
  const byTypeLayer = new Map();
  for (const n of tree.nodes) {
    if (!byTypeAll.has(n.type)) byTypeAll.set(n.type, []);
    byTypeAll.get(n.type).push(n);
    const k = n.type + '|' + n.layer;
    if (!byTypeLayer.has(k)) byTypeLayer.set(k, []);
    byTypeLayer.get(k).push(n);
  }
  const peersOf = (n, global) => (global ? byTypeAll.get(n.type) : byTypeLayer.get(n.type + '|' + n.layer)) || [];
  const ev = makeEvaluator(env, peersOf);
  const order = new Map(tree.nodes.map((n, i) => [n, i]));

  const picked = new Set();
  for (const q of ast.queries) {
    let cands = [];
    for (const t of q.types) for (const n of byTypeAll.get(t) || []) cands.push(n);
    cands.sort((a, b) => order.get(a) - order.get(b));
    if (q.expr) cands = cands.filter((n) => truthy(ev(q.expr, n)));
    for (const sg of q.stages) {
      if (sg.op === 'limit') { cands = cands.slice(0, sg.n); continue; }
      const sign = sg.op === 'top' ? -1 : 1;
      const groups = new Map();
      for (const n of cands) {
        const g = sg.per === 'layer' ? n.layer : sg.per === 'hybrid' ? ATTRS.hybrid(n) : 0;
        if (!groups.has(g)) groups.set(g, []);
        groups.get(g).push(n);
      }
      const keep = new Set();
      for (const list of groups.values()) {
        const scored = list.map((n) => ({ n, v: num(ev(sg.by, n), sg.pos) }));
        scored.sort((a, b) => sign * (a.v - b.v) || order.get(a.n) - order.get(b.n));
        for (const x of scored.slice(0, sg.n)) keep.add(x.n);
      }
      cands = cands.filter((n) => keep.has(n));
    }
    for (const n of cands) picked.add(n);
  }
  const nodes = [...picked].sort((a, b) => order.get(a) - order.get(b));
  const seen = new Set();
  const items = [];
  const byType = {};
  for (const n of nodes) {
    byType[n.type] = (byType[n.type] || 0) + 1;
    for (let i = n.lo; i < n.hi; i++) {
      const it = tree.items[i];
      if (seen.has(it.id)) continue;
      seen.add(it.id);
      items.push(it.item);
    }
  }
  return { nodes, items, byType, ast };
}

/// 結果の 1 行要約 (画面のステータス用)。
export function dslSummary(res) {
  const parts = Object.keys(res.byType).map((k) => k + ' ' + res.byType[k]);
  return res.nodes.length + ' nodes' + (parts.length ? ' (' + parts.join(', ') + ')' : '') +
    ' → ' + res.items.length + ' items';
}

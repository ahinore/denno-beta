# full_qk の D2H を pinned にする設計メモ (P41・実装なし)

- Tree: `/work/denno-dev2` (P41 の変更を入れた後)。対象は q4 prefill の残差常駐経路。
- このメモは**設計だけ**。コードは 1 行も変えていない (次のタスクで実装する)。
- 数値のラベル: **MEASURED** = この repo / このホストで測った値、**RECONSTRUCTED** =
  測った値から算術で出した値、**ASSUMED** = 置いた仮定 (測っていない)。

---

## 1. 結論 (先に)

1. **いまの `/work/denno-dev2` の HEAD は、task の言う「pageable な 3 発 D2H」ではない。**
   7e9ea37 (P17b の再適用) で **1 発の連続 pinned D2H** になっている
   (`chain_qkv_prefill_q4_dev`, `cuda.rs:13031` / D2H は `cuda.rs:13090-13100`)。
   task の前提 (3 発・pageable・805.3 MB/層) は **denno-beta の f55c707 (P17b revert) 後**の
   状態で、こちらは実機で 9.78 → 22.14 s (+772 ms/層) を測って revert された形。
   したがって実装は「pinned 化」の前に **3 発へ戻す**ことを含む (統合は禁止)。
2. 提案する形は **3 発のまま・宛先だけ pinned・プールは層ループの外で 1 回だけ確保して
   使い回す・ホストはその pinned スライスを直接読む**。コピー回数・発行順・GEMM は不変。
3. 削減量は**このホストの実測単価ではほぼゼロ**の可能性が高い:
   s68 の実測は pageable 直 D2H **13.9 GB/s** / pinned **14.3 GB/s** (MEASURED,
   `docs/history/40_prefill_perf.md` の s68) で、差は 12.88 GB に対して **26 ms/request** しかない。
   一方 PF_DBG の `full_qk` バケット (2,101 ms) から GEMM を引くと D2H は 6.5 GB/s 相当で、
   こちらを信じるなら **−1.08 s/request**。**この 2 つの読みは両立しない (§7)。**
4. だからこの設計は「速くする」より **「どちらの読みが正しいかを壊さずに測れる形にする」**
   ものとして書く。3 発・同順を保つので、**最悪でも現状と同じ**で、良くて −1.08 s。

---

## 2. 現行コード (このツリーの実態)

| 箇所 | 行 | 内容 |
|---|---|---|
| `chain_qkv_prefill_q4_dev` | `cuda.rs:13031` | 3 GEMM (`:13083-13085`) → **1 発の `cuMemcpyDtoHAsync`** (`:13096`, 宛先 `pin_pf_qkv`) → `ctx.sync()` 1 回 |
| `Scratch::ensure_pin` | `cuda.rs:1156` | grow-only (`bytes` が足りなければ `PinMem::alloc`) |
| `PinMem::alloc` | `cuda.rs:1062-1073` | `cuMemHostAlloc(bytes, CU_MEMHOSTALLOC_PORTABLE)`。**WRITECOMBINED は付けない** |
| `PfQkvHostView` / `chain_pf_qkv_host_view` | `cuda.rs:13112` / `:13137` | pinned 連続領域を q/k/v の 3 スライスとして見せる (`pf_qkv_geom` で寸法照合) |
| 呼び出し元 | `qwen35.rs:2590-2598` | `chain_qkv_prefill_q4_dev` → `chain_pf_qkv_host_view` → `full_attn_host_side(QkvPre{..})` |

寸法 (27B q4・n=14,043 = 65,000 chars, MEASURED な形状): q = 690.2 MB (`n×12288×4`)、
k = v = 57.5 MB (`n×1024×4`)、**計 805.3 MB/層**、16 full 層で **12.88 GB/request**。
q が全体の 86 %。

---

## 3. 提案する形 (3 発・pinned・1 回確保)

```
qkv チェーン (層あたり):
  q GEMM → k GEMM → v GEMM            (不変: 3 launch・連結しない)
  D2H q → D2H k → D2H v               (3 発・不変: 連結しない)
     宛先 = pin_qkv の 3 スライス (q_off=0, k_off=q_bytes, v_off=q_bytes+k_bytes)
  ctx.sync() 1 回                     (不変)
ホスト: q/k/v は pin_qkv の 3 スライスを直接読む (Vec への写しをしない)
```

- プールは `PinMem` 1 本 (805.3 MB) を**スレッドローカルに grow-only** で保持し、
  3 スライスに割る。3 本の `PinMem` に分けてもよいが、ページロックは 1 回で済む方が安全。
- コピーは 3 発のまま。**発行順は現行 (q GEMM, k GEMM, v GEMM, そのあと 3 発の D2H)** を変えない
  — 途中で割り込ませる (q の D2H を q GEMM の直後に置く) と発行順が変わる上、得られる重なりは
  k/v GEMM の 2×147 GFLOP ≈ 10 ms/層 だけ (§8)。

---

## 4. 寿命と再利用

- **寿命の契約は現行と同じ**: 「同じスレッドで次の `chain_qkv_prefill_q4_dev` が走るまで有効」。
  層ループは 1 層ずつ「qkv チェーン → ホスト prepare/attend」を直列に回すので、1 層の中で
  使い切る限り競合しない (`pin_h2` / `PfQkvHostView` と同じ規約)。
- **確保はリクエスト 1 回・スレッド 1 本**: `Scratch::ensure_pin` が grow-only なので、
  最初の full 層で 805.3 MB を 1 回だけ確保し、以降の 15 層と次のリクエストは再利用する。
  n が増えたときだけ再確保 (grow)。
- ホスト側に返す型は `&[f32]` 3 本 (または `PfQkvHostView` 相当の 3 スライス struct)。
  **Vec へはコピーしない** — pinned → heap の `memcpy` はこのホストで **4.7 GB/s** (MEASURED,
  `docs/history/40_prefill_perf.md` の s68) しか出ず、12.88 GB なら +2.7 s の新規費用になる。
- **初回タッチの費用は 1 回だけ**: 新規 pinned は「確保 + 初回タッチ」で 1.48 GB/s まで落ちる
  (MEASURED, 279 MB の例 `docs/history/40_prefill_perf.md` の s68)。805 MB なら ≈0.54 s を**最初の 1 層**が
  払う。cold A/B では 1 リクエスト目を捨てる (prefill_bench の既定) か、`begin` で
  ダミー H2D を 1 発打って温める。

---

## 5. ホスト RAM とページロック

- 常駐量: **805.3 MB / スレッド** (n=14,043)。n に比例 (`n×14336×4` B)。VRAM は 0 (増えない)。
  3 本に分ける場合も合計は同じ (690.2 + 57.5 + 57.5)。
- ページロック済みなので **スワップ不可**。サーバの常駐セットに 805 MB が乗る。
  n=65,000 token 級 (n≈65,000) では 3.7 GB になる — **その場合はプールを作らずに
  現行 (pageable) へ落とす**上限を入れる (§6)。
- `CU_MEMHOSTALLOC_PORTABLE` のみ。**`WRITECOMBINED` (0x04) は付けない**:
  D2H の宛先は直後にホストが読むので、WC にすると読みが遅くなる (書きは速いが読みは遅い)。

---

## 6. 失敗時のフォールバック (fail-closed)

1. `cuMemHostAlloc` 失敗 (ページロック上限・断片化・OOM) → **確保を 1 バイトもせず**に
   `None` を返し、呼び出し側は現行の pageable 3 発経路をそのまま通す。
   `GPU_BROKEN` は立てない (pinned の失敗は致命的でない — `pin_h2` と同じ扱い)。
2. プール上限 (既定 1.0 GB など) を超える n → pinned を使わず pageable 経路。
3. イベント/コピーの実行時エラー → 従来どおり `GPU_BROKEN` → CPU 経路 (数値は不変)。
4. どの経路でも **数値はビット一致**: 変わるのは「同じバイト列をどこへ置くか」だけ。

---

## 7. 想定削減量 (計算式つき・2 つの読み)

12.88 GB/request (805.3 MB × 16 層)。削減量 = `12.88 × (1/R_pageable − 1/R_pinned)` 秒。

| 読み | R_pageable | R_pinned | D2H 現状 | D2H 後 | 削減 | 出所 |
|---|---|---|---|---|---|---|
| (A) マイクロベンチ | **13.9 GB/s** (MEASURED) | **14.3 GB/s** (MEASURED) | 927 ms | 901 ms | **−26 ms** | s68 `docs/history/40_prefill_perf.md` (345 MB の 1 発) |
| (B) PF_DBG バケット | **6.5 GB/s** (RECONSTRUCTED) | 14.3 GB/s (MEASURED) | 1,982 ms | 901 ms | **−1.08 s** | `full_qk` 2,101 ms − GEMM 131 ms (2.06 TFLOP @15.7 TFLOP/s) = 1,970 ms |
| (C) task の「~6 GB/s 上限」 | 6.0 GB/s (ASSUMED) | 14.3 GB/s (MEASURED) | 2,147 ms | 901 ms | **−1.25 s** | task の一般論。このホストの実測 (A) とは矛盾 |
| (D) task の「約 1.0 s @65k」 | 12.9 GB/s (ASSUMED) | 14.3 GB/s (MEASURED) | 1,000 ms | 901 ms | **−0.10 s** | task 本文の理論値 |

**この 4 つは互いに矛盾している** (同じ 12.88 GB に対して 927 ms〜2,147 ms)。task の
「~6 GB/s 上限」と「約 1.0 s」は同じ文の中で 2 倍食い違う。in-tree で測れているのは (A) だけで、
(B) は位相会計からの逆算 (P26 §2.2 が「位相 % と 11 TFLOP/s は両立しない」と記録した論点そのもの)。

→ **設計は (A)〜(D) のどれでも損をしない形にする**: コピー 3 発・発行順・GEMM を変えず、
宛先だけ変える。最悪 (A) で ±0、最良 (B)/(C) で −1.1〜−1.3 s。**削減量は実装後に
`full_qk` の 2 点計測 (下記) で確定させる**。

### 確定させるための計測 (次のタスク・GPU 実機)

1. `cuda_transfer_bench` (`#[ignore]`) を **805.3 MB / 690.2 MB / 57.5 MB の 3 サイズ**で回し、
   pageable 直 D2H と pinned D2H の実効 GB/s を測る (s68 は 345 MB の 1 点だけ)。
2. `DENNO_PF_DBG=1` の `sl_lay_full_qk` を cold で 3 回 (中央値)。
3. A/B: 現行 (1 発 pinned・P17b) / 3 発 pageable / 3 発 pinned の 3 点。
   **1 リクエスト目は捨てる** (初回タッチ)。
4. `DENNO_FC_KPROF=1` (P41) で `full_core` 側の prep/attn/d2h/tail も同時に取り、
   D2H の変化が下流に漏れていないことを確認する。

---

## 8. なぜこれは revert 済みの P17b ではないか

| | P17b (7e9ea37 / 実測 9.78 → 22.14 s・revert) | 本設計 |
|---|---|---|
| D2H の回数 | **3 → 1 に統合** | **3 発のまま** (統合しない) |
| 宛先 | 1 本の連続 805.3 MB (`pin_pf_qkv`) | 3 スライス (同一プールを 3 分割) |
| ホストの見え方 | `PfQkvHostView` の要素オフセット計算 (`ptr.add(q+k)`) | q/k/v がそれぞれ独立した `&[f32]` |
| 確保 | 1 回 (grow-only) — **P26 の「層ごとに確保」仮説はコードと合わない** | 1 回 (grow-only)。同じ |
| GEMM | 不変 (3 発) | 不変 (3 発)・**連結しない** |
| 実機実測 | +772 ms/層 = 805.3 MB を 1.04 GB/s で流した単価 | 未計測 |

- **統合そのものが禁止**なのは、実測が「連結は構造的に中立なはずなのに +772 ms/層」だったから
  (P26 §3)。805.3 MB ÷ 772 ms = **1.04 GB/s** は帯域ではなく「805 MB の pinned を
  層ごとに払った単価」に近い (P26 の推定・**未確認**)。ただし本ツリーの `ensure_pin` は
  grow-only で、プールはスレッドに 1 本しかない — **この仮説はコードと一致しない**。
  したがって 22.14 s の原因は**未解明**で、「1 発に統合したこと」自体が原因の可能性も残る。
  → だから統合はしない (どちらが原因でも踏まないため)。
- **GEMM の統合 (q|k|v を 1 launch に) もしない**: 出力レイアウトが変わり (行ストライド 14336
  対 列ブロック 3 分割)、D2H が連続 805 MB になる = P17b そのもの。利得は launch 2 発
  = 4〜6 µs/層 = 0.1 ms/request で、リスクに見合わない (P26 §4 の結論を維持)。

---

## 9. リスク

1. **ページロック 805 MB** の常駐 (スワップ不可)。n が大きいリクエストでは上限で切る (§6)。
2. **初回タッチ**で最初の 1 層が ≈0.54 s 遅くなる (cold A/B の 1 回目を捨てないと退行に見える)。
3. **ホスト RAM の断片化**で `cuMemHostAlloc` が失敗し得る → フォールバック必須 (§6)。
4. **P17b の原因が未解明**なので、「3 発 pinned」でも同じ 1 GB/s 現象が出る可能性は 0 ではない。
   そのときは 3 発 pageable (現行の denno-beta 形) に戻すだけで済む (数値は不変)。
5. 寿命規約 (次の qkv チェーンまで) を破る呼び出しを足すと**読む前に上書きされる**。
   `PfQkvHostView` と同じく、寸法照合キー (`pf_qkv_geom` 相当) を必ず持たせる。

---

## 10. 未検証点 (このメモで測っていないこと)

- **すべての GPU 実機値** (このコンテナは GPU 非搭載)。(B)(C)(D) は算術・仮定。
- pageable 直 D2H の実効帯域が 805.3 MB の 3 発で 13.9 GB/s を維持するか (s68 は 345 MB 1 発)。
- 22.14 s の原因 (統合 / 確保 / 初回タッチ / その他)。
- pinned 3 発にしたときの `full_qk` の実測と、`full_core` への漏れ (P41 の `DENNO_FC_KPROF`)。
- `.cu` / `.ptx` は無関係 (この設計は Rust 側のコピー先だけ)。PTX 再生成は不要。

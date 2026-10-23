# full_core の tail 区間の分解 (P50・コード変更なし)

- 対象: 27B q4・65k (n=14,043)・16 full 層・1 リクエスト。
- この文書は**分析のみ**。コードは 1 行も変えていない (変える理由が無かった。§6)。
- 数値のラベル: **[M]** 実測 (メインセッションの実機値、またはこのコンテナの CPU 実測) /
  **[R]** 実測値とコードの形状からの算術 / **[A]** 仮定。

## 1. 結論

1. `tail` (634 ms / 16 層 = **39.6 ms/層**) の中身は **ホスト側の sigmoid 出力ゲート 1 本**で、
   転送ではない。ゲートは pinned ステージング上の att に対する 3 パス
   (att read 345.1 MB + qg のゲート半列 read 345.1 MB + att write 345.1 MB = **1,035.4 MB/層**)
   で、39.6 ms は **26.1 GB/s 相当 [R]**。このコンテナで同じ 3 パスを回すと
   **40.1〜40.9 ms/層・25.3〜25.8 GB/s [M]** (16 層で 642〜654 ms = 実測 tail の 101〜103 %)。
2. tail に入り得る最大の転送は att の H2D **345.1 MB/層**で、pinned の実効 **14.0 GB/s
   [R]** (同じ 345.1 MB の D2H が d2h 区間 24.6 ms/層であることから逆算) なら
   **24.6 ms/層 = 394 ms/request**。tail 634 ms の **62 %** にしかならない。
   「tail 全体が 1 発の転送」と読むと **8.7 GB/s [R]** になり、pageable の実測
   13.9 GB/s (s68) より 1.6 倍遅い — 単一転送では説明できない。
3. しかも **att の H2D は既に pinned から出ている**。残差常駐経路は
   `chain_pf_att_inplace` (cuda.rs:13102) が D2H の宛先と H2D の転送元を同じ
   pinned (`pins[li*3+2]`, cuda.rs:13113/13119/13157) にしており、
   `chain_pf_upload_att` 経路 (cuda.rs:13044) も pinned へ写してから H2D する
   (cuda.rs:13058-13066)。**pageable な att の H2D は現行コードに存在しない。**
4. したがって task の条件 (「tail が pageable な att H2D に支配されている場合のみ
   P47 パターンを適用」) は**成立しない**。実装は無し (§6)。

## 2. 前提 (与えられた実測値・測り直していない)

| 区間 | 16 層合計 | 1 層あたり | 区間の定義 (cuda.rs:12361-12370) |
|---|---|---|---|
| prep | 1,094 ms | 68.4 ms | FULL_CORE 境界 → attention カーネル発行直前 |
| attn | 2,089 ms | 130.6 ms | 発行直前 → 発行直後 (カーネル実行) |
| d2h | 394 ms | 24.6 ms | 発行直後 → att の D2H 完了 |
| tail | **634 ms** | **39.6 ms** | D2H 完了 → FULL_OPROJ 境界 |

形状はコードから確定 [R]: `e = 5,120` / att の列数 `heads*dh = 6,144`
(qwen35.rs:15 のコメント「27B は非正方形 6144→5120」・`pf_layer_geom` の
`qn == att*2` 検査 cuda.rs:12946) / q 射影の列数 `heads*2*dh = 12,288` /
`kv_dim = 1,024` (cuda.rs:12945)。dtype は全経路 **f32** (要素数×4 バイトで
`chain_ensure_buf` / `cuMemcpy*` を呼んでいる)。

| バッファ | 要素数 | バイト数 (n=14,043) |
|---|---|---|
| att (`CH_ATT` / pinned ステージング) | 14,043 × 6,144 = 86,280,192 | **345.12 MB** (329.1 MiB) |
| q 射影出力 qg (gate 込み) | 14,043 × 12,288 = 172,560,384 | 690.24 MB |
| うちゲート半列 (ゲートが読む分) | 86,280,192 | **345.12 MB** |
| q の pinned ステージング (`pin_pf`) | 86,280,192 | 345.12 MB |
| KV ミラー H2D (k + v の n 行) | 2 × 14,043 × 1,024 | 115.08 MB |

task にあった「att は約 230 MB/層」は**誤り**。`heads*dh = 6,144` の f32 なので
**345.1 MB/層** (230 MB になるのは列数 4,096 のとき)。

## 3. tail 区間の中身 (コード順・残差常駐経路)

呼び出しは qwen35.rs:3096-3124。`n=14,043 > max_m()=16` なので残差常駐経路が選ばれ
(`pf_res_gate` cuda.rs:12864)、`full_attn_host_side` には `host_att_needed = false`
(qwen35.rs:3108) が渡る = `allow_dev = true`。

```
chain_qkv_prefill_q4_dev            q/k/v 射影 → q/k を pinned へ D2H、v は KV 行へ  (full_qk 相)
pf_phase(PF_PH_FULL_CORE)           → FC_KPROF mark(0)
  prepare (ホスト)                   QK-Norm + RoPE + KV 行書き込み → q を pin_pf へ直接
  chain_attention_prefill_pinned     KV ミラー H2D 115.1 MB + q H2D 345.1 MB
                                     → mark(1) → カーネル → mark(2)          [prep / attn]
  chain_pf_att_inplace               D2H 345.1 MB を 6 チャンクで発行 → mark(3)  [d2h]
                                     各チャンク: ev 待ち → sigmoid ゲート → H2D  [tail]
chain_pf_upload_att                  呼ばれない (att_place == GpuDev, qwen35.rs:3116)
chain_oproj_prefill_q4               prologue (weights_for_q4 + chain_ensure_buf)
                                     → pf_phase(PF_PH_FULL_OPROJ) → mark(4)
```

tail = mark(3) → mark(4) の間にホストがやることは 3 つだけ:

1. **sigmoid ゲート** (qwen35.rs:1710-1720 のクロージャ)。チャンクごとに
   `att /= 1 + exp(-gate)`。要素あたり read 4 B + read 4 B + write 4 B。
   1 層 = 86,280,192 要素 × 12 B = **1,035.36 MB**。
2. **att の H2D** (cuda.rs:13317-13324)。pinned → `CH_ATT`、1 層 345.12 MB。
   6 チャンクに割れているので最後の 1 つ = 57.52 MB。
3. **o_proj の prologue** (cuda.rs:13626-13640)。`weights_for_q4` はレジストリ
   ヒット (warm)、`chain_ensure_buf(CH_O)` は grow-only の no-op → **µs 級**。

アロケートは **0 件** (pinned は層ごとに grow-only で既に在り、`CH_ATT`/`CH_O` も
`chain_pf_reserve` で確保済み)。変換はゲート 1 つだけ。

### 3.1 実効レート

| 対象 | バイト数 | 区間 | 実効 |
|---|---|---|---|
| D2H (pinned 宛) | 345.12 MB | 24.6 ms/層 | **14.0 GB/s [R]** (s68 の pinned 実測 14.3 GB/s と一致) |
| att H2D (pinned 元) | 345.12 MB | — | 14.0 GB/s とみなすと **24.6 ms/層 [R]** |
| ゲート 3 パス | 1,035.36 MB | 39.6 ms/層 (tail) | **26.1 GB/s [R]** |
| attention カーネル | 2.42 TFLOP (causal の対 ≈ n²/2 = 9.86e7 行、QK と PV で各 9.86e7 × 6,144 MAC = 1.21e12 MAC) | 130.6 ms/層 | 18.5 TFLOP/s [R] (fp32 融合カーネルとして妥当) |

### 3.2 このコンテナでのゲート実測 (CPU のみ・GPU 不使用)

`att[i] /= 1 + exp(-gate[i])` を 86,280,192 要素 (345.12 MB × 2 本) で、
16 スレッド (このコンテナの `nproc`) で回した [M]:

```
pure-read  n=86280192  12.0〜15.4 ms  22.4〜28.8 GB/s
gate-3pass n=86280192  40.1〜40.9 ms  25.3〜25.8 GB/s (traffic 1.035 GB)
```

1 層 40.1〜40.9 ms × 16 層 = **642〜654 ms**。実測 tail **634 ms の 101〜103 %**。
このコンテナは 16 コアなので、GPU ホストのほうがコア数が多い分だけ速い
(実測 tail から逆算すると 1 層 ≈ 35 ms = 29.6 GB/s [R]) とみるのが自然で、
どちらの数値でも「tail ≈ ゲート」で一致する。

### 3.3 P46 の前後が同じモデルで説明できる

- P46 前: `tail = ゲート + H2D 全体` = 35〜39 + 24.6 = 60〜63 ms/層
  → 実測 63.4 ms/層 [M] と一致。
- P46 後: `tail = ゲート + 最後のチャンクの H2D` = 35 + 4.1 = 39.1 ms/層
  → 実測 39.6 ms/層 [M] と一致。
- P46 が予測した「tail → 394 ms」は「ゲートが D2H に完全に隠れる」場合の下限で、
  実際にはゲート (35〜40 ms/層) のほうが D2H (24.6 ms/層) より長いので隠れ切らない。
  残った 634 ms は**ゲートそのもの**であって、転送の取り残しではない。

## 4. データパス (変更前 = 変更後)

```
D2H  : CH_ATT (device) ──cuMemcpyDtoHAsync──> pins[li*3+2] (pinned, 345.1 MB)
ゲート: pins[li*3+2] を in-place で書き換え (qg のゲート半列を読む)
H2D  : pins[li*3+2] ──cuMemcpyHtoDAsync──> CH_ATT (device)
```

- **変更なし**。pinned ステージングもチャンク分割も P46 のまま。
- フォールバック (`DENNO_PF_ATT_INPLACE=0` / pinned 確保失敗 / イベント作成失敗):
  `chain_download(CH_ATT)` が pageable な `Vec` へ D2H (qwen35.rs:1746) →
  ホストでゲート (qwen35.rs:1765) → `buf_att` へ `copy_from_slice` (qwen35.rs:1777) →
  `chain_pf_upload_att` が **pinned へ写してから** H2D (cuda.rs:13058-13066)。
  この経路でも H2D の転送元は pinned。

## 5. pinned バッファの大きさと寿命 (現行のまま)

| 項目 | 値 |
|---|---|
| 実体 | `ChainScratch.pins[li*3+2]` (cuda.rs:13113) — **full 層ごとに 1 本** |
| 大きさ | `n × heads*dh × 4` = **345.12 MB** (n=14,043)。n に比例 |
| 確保 | `cuMemHostAlloc(bytes, CU_MEMHOSTALLOC_PORTABLE)` (cuda.rs:1080-1091)。WRITECOMBINED なし |
| 寿命 | スレッド寿命 (grow-only・解放しない)。**内容の契約**は「同じ層の次の `chain_pf_att_inplace` まで」= 層ループが直列なので 1 層の中で使い切る |
| 65k の常駐量 | 16 層 × 345.12 MB = **5.52 GB** (ほかに `pin_pf` 345.1 MB / `pin_pf_qkv` 747.8 MB / `pin_h2` 287.6 MB) |
| VRAM | 0 (増えない) |

## 6. なぜ実装しないか

task の条件は「tail が **pageable な** att H2D に支配されている場合のみ P47 パターン
(pinned ステージング + fail-closed) を適用」。実際は

- tail の 9 割以上がホストのゲート (転送ではない) [R]+[M]、
- tail に入る唯一の転送は att の H2D で、**既に pinned から出ている** (§3, §4)、
- pageable な att H2D は現行コードのどこにも無い (§4)。

なので pinned 化しても**削れるバイトが 0**、置き換える対象が存在しない。数値不変の
主張も不要 (何も変えない)。削減見積りは **0 ms (推定)**。

## 7. 次にやるなら (このタスクの範囲外・実装していない)

1. **ゲートをデバイス側へ** — `dno_attn_gate_mul` の n 行版 (`kernels_elem.cu`) を足せば
   D2H/H2D の往復 690 MB/層が消える。ただし **`.cu` + PTX 再生成 = 人間対応**
   (AGENTS.md の禁止事項)。削減見積り: ゲート 35〜40 ms/層 + 転送 49.2 ms/層の
   うち、カーネル時間 (≈ att の 3 パス相当をデバイスで ≈ 690 MB / 2 TB/s = 0.35 ms/層)
   を除いた **≈ 80 ms/層 = 1.3 s/request (推定)**。数値は
   `dno_attn_gate_mul` と同一の式・同一順なら bit 一致 (要検証)。
2. **att の pinned を層ごとに持つのをやめる** — 層ループは直列で、次の層が同じ
   スロットを触る前に必ず `ctx.sync()` が挟まる (qkv チェーン)。1 本の使い回しで
   5.52 GB → 345.1 MB に減る見込み。wall は変わらない (ホスト RAM だけ)。
   P46 が層ごとに専用にした理由 (cuda.rs:13041-13043 のコメント「非同期 H2D の完了前に
   次の層が同じ領域を上書きしないため」) を壊さないことを先に確認する必要がある。
3. **`chain_pf_upload_att` の失敗が prefill 全体のエラーになる** (qwen35.rs:3118-3121)。
   pinned 確保に失敗したときの pageable H2D への fail-closed が無い。P47 の
   「pinned の失敗は致命的でない」規約と揃っていない (今回の tail とは無関係)。

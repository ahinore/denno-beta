ADOPT: no t291a t291b t291d

# REPORT P291 — UD-Q4_K_XL の prefill: MoE 作業域の使い回しと down の VNNI GEMM

branch: br291 (int-auto c5cb293 = br282 の merge から。br286 は未 merge だったので
最初に merge した — `d975f50`)。lab: t291a / t291b / t291d (t291c は COND の
環境を間違えたため判定に使わない。下の 5.3)。

## 0. 結論

- **UD-Q4_K_XL の 3,826 cold prefill は 51.0 -> 36.8 s (-27.9%)。** 内訳は
  expert 39.4 -> 24.0 s (-39%)。decode は 24.1 -> 22.1 ms/tok (-8.4%) で、
  目標の「2% 悪化まで」に反する方向ではない。errors 0、text sane、md5 は
  c2000_cold だけ DIFF (Q5_1 の累積順を変えたため。下の 3.)。
- 30% には届いていない (-27.9%)。同じ run で `RAYON_NUM_THREADS=16` にすると
  35.04 s (-31.4%) で target に届くが、これは serve_flash の既定スレッド数
  (task 119b が「decode 14 が折れ点」として `論理 CPU - 2` に決めた) を変える話で、
  t291d の既定モデルでは損得を切り分けられず、119b の記録 (+3.5%) と合わせると
  既定モデルの decode を 2% 以上悪くする危険がある。採否規則に従い **ADOPT: no**。
  ゲートは 2 本とも既定 off (`DENNO_MOE_SCRATCH_KEEP_MB=2048` と
  `DENNO_Q51_BLK4=1` で on)。main session が既定モデル込みで 16 スレッドを
  測り直せるなら、この 2 本を on にするだけで -28% が載る。

## 1. 前提 (測り直していない。int-auto と lab の実測)

lab t286c の b0 (int-auto、PROFILE=1) の 3,826 cold prefill (wall 51.99 s):

| 段 | 実測 | 割合 |
| --- | --- | --- |
| experts (CPU、flat MoE) | 40.01 s | 77.0% |
| mix (GPU、hc mix) | 8.48 s | 16.3% |
| other | 2.20 s | 4.2% |
| ssm | 0.65 s | 1.3% |
| full | 0.33 s | 0.6% |
| shared | 0.25 s | 0.5% |
| router | 0.07 s | 0.1% |

- `[pf-host] experts_wall=40012 ms | plan=16.4 prefetch=0.7 | other=39995` =
  expert の 40 s はほぼ全部が GEMM の計算。
- `[q4ledger-pf] experts cpu_ms=40010.8 launches=0` = prefill の選択 expert は
  GPU を 1 度も使っていない。UD の 1 expert = 3,072,000 B
  (gate/up Q4_K 2560x640、down Q5_1 640x2560、REPORT_P278 の実測)。
- UD は `[pf-lm] 層 0: dev MoE を諦めた (理由: gate)` で層優先のデバイス MoE に
  入らない (`lm_dev_moe_ok` が `iq_expert_mode` を要求し、K 型の device GEMM が
  無いため)。IQ3_S が 3.5 s なのはこの device 経路に乗るから。

## 2. 変えたこと (全部 env ゲート付き)

### 2.1 MoE の作業域を層で使い回す (`DENNO_MOE_SCRATCH_KEEP_MB`、既定 2048)

`moe_scratch_put` は作業域が 512 MiB を超えると捨てていた。実形
(tot = 3826 × 8 = 30,608、e 2560 / ff 640) の作業域は
xin 313 + act 78 + up 78 + h 313 = 782 MB で **毎層捨てられ**、次の層で
`vec![0.0; ...]` のゼロ埋めを払い直していた。上限を env 可変 (既定 2048 MiB) に
して 48 層で使い回す。lab t291a の実測: `[moe-flat] resize` 149 -> 0 ms/層、
minor fault 14.8 M -> 2.5 M、`[exp-ram-probe]` の majflt 152k -> 139k。

lab t291a (同じ commit、`DENNO_MOE_SCRATCH_KEEP_MB=512` が b0/b1):
b0 52.43 / keep 40.30 / b1 50.85 s -> **-20.7% (vs b1)**。prefetch
(`DENNO_EXPERT_PREFETCH=1`) は 41.03 s で追加の効果なし。decode は
23.6 / 23.0 / 21.4 ms/tok で悪化なし。md5 は全 COND SAME。

### 2.2 down (Q5_1) の VNNI GEMM を 4 ブロックまとめ食いに (`DENNO_Q51_BLK4`、既定 on)

Q5_1 は 1 ブロック = 32 要素なので、旧カーネルは **32 MAC ごと**に水平加算
(256 ビット `vpdpbusd` 1 本 + 8 レーン縮約) と累積の読み書きをしていた。
2 ブロックを 1 本の 512 ビット `vpdpbusd` に載せ、縮約を 128 要素に 1 回へ
減らした (`gemm_q5_1_vnni4`)。`d * xd` (重みスケール × 活性スケール) は
8 レーンごとに違うのでレーンへブロック単位で配る。式は旧カーネルと同じ
`Σ_b d_b * xd_b * (Σ q x)`、最小値項 `Σ_b m_b (Σ x)` は 4 本まとめて足す。

### 2.3 索引付き VNNI (`DENNO_MOE_FLAT_IDX`、既定 off) — 効果なし

タイル全体を 1 回だけ量子化し、カーネルが `VnniAct::tid` で活性の行を引く
(gather 無し)。ローカル実測は 1 層 441 -> 441 ms (差なし)。gather 13 ms が
消える代わりに活性の局所性が落ちて相殺した。既定 off で残した
(ビット一致テスト付き)。

## 3. 数値の扱い

- 2.1 は作業域の再利用だけで値は不変。2.3 もビット一致 (テストあり)。
- 2.2 は prefill の丸め契約の外 (2026-10-28 のオペレーター指示) で、
  int32 の積和を 128 要素まとめて f32 に落とすため累積順が変わる。
  lab t291b の md5 は a / b と c2000_turn2 で SAME、c2000_cold だけ DIFF
  (a:3a25e36e -> 71a7197c)。text は "looks ok"。
- 既定モデル (IQ3_S) は IQ 型でこのカーネルを通らない。

## 4. 見積り (ローカル実測)

`moe_ud_prefill_microbench` (ignored、実形 ne 512 / k 8 / nt 3826、
gate/up Q4_K・down Q5_1、14 スレッド、rep の min):

| 構成 | 1 層 experts | 48 層 |
| --- | --- | --- |
| 変更前 (旧上限・旧カーネル) | 617.8 ms | 29.7 s |
| + 2.1 (作業域の使い回し) | 442.7 ms | 21.3 s |
| + 2.2 (down 4 ブロック) | 377.9 ms | 18.1 s |

2.1+2.2 の内訳: gather 13 / phaseA (gate+up) 149 / phaseB (down) 84 /
scatter 10 / router+shared 121 ms。phaseB は 152 -> 84 ms (-44%)。
lab はローカルの 1.33 倍 (ページキャッシュのミスぶん)。

## 5. lab

### 5.1 t291a (作業域の使い回し)

| cond | 3,826 cold prefill | decode ms/tok (短) | md5 |
| --- | --- | --- | --- |
| b0 (旧上限 512) | 52.43 s | 23.6 | SAME |
| keep (既定 2048) | **40.30 s** | 23.0 | SAME |
| kpfx (+PREFETCH=1) | 41.03 s | 24.0 | SAME |
| b1 (b0 の繰り返し) | 50.85 s | 21.4 | SAME |

`VERDICT_DEPTH keep c2000_cold vs b0 decode -2.7% prefill -23.1%`

### 5.2 t291b (down 4 ブロック)

| cond | 3,826 cold prefill | decode ms/tok (短) | md5 |
| --- | --- | --- | --- |
| b0 (旧上限 + 旧カーネル) | 51.03 s | 24.1 | 3a25e36e |
| new (既定) | **36.84 s** | 22.1 | 71a7197c (cold のみ DIFF) |
| newt16 (+RAYON=16) | **35.04 s** | 21.8 | 同上 |
| b1 (b0 の繰り返し) | 51.09 s | 24.6 | SAME |

`VERDICT_DEPTH new c2000_cold vs b0 decode +10.0% prefill -27.8%`
`VERDICT_DEPTH newt16 c2000_cold vs b0 decode -2.0% prefill -31.3%`

new の ledger: experts 24.0 s (b1 39.4 s)、mix 8.47 s、other 2.13 s、
ssm 0.64 s、full 0.31 s、shared 0.23 s、router 0.06 s。wall 35.9 s。

### 5.3 t291c (無効) と t291d (既定モデル)

t291c は COND に baseline の 4 つの env (`DENNO_EXPERT_IQ_GPU=1` など) を
入れ忘れて走らせた。既定モデル (IQ3_S) は `DENNO_EXPERT_IQ_GPU` が無いと
iq ゲートが閉じて device MoE を諦め、CPU の flat 経路になるため
prefill 37.45 s / decode 40.5 ms/tok という**別物の構成**を測っていた
(md5 も c2000_cold だけ違う)。判定には使わず、正しい env で t291d を
取り直した。t291d は r14 / r16 / r14b の 3 COND。

t291d (既定モデル swift15-abl-IQ3_S、depth、PROFILE=1、baseline の 4 env 付き):

| cond | decode ms/tok (短) | 3,826 cold prefill | md5 |
| --- | --- | --- | --- |
| r14 (`RAYON_NUM_THREADS=14`) | 19.0 | 6.14 s | b2057e53 |
| r16 (=16) | 19.2 | 3.35 s | b2057e53 |
| r14b (=14 の繰り返し) | 19.1 | 4.81 s | b2057e53 |

- md5 は int-auto の参照 (tint_c5cb2939c / t287a の `b2057e53` / `3852ffec` /
  `82ab0c06` / `624e0d4d`) と **全項目一致** = 既定モデルの値は変わっていない。
- decode 19.0 / 19.2 / 19.1 は差が無い (host load 5.17 で、参照の 18.1〜18.8 とは
  その分のずれ)。prefill 6.14 / 3.35 / 4.81 は同じ構成の繰り返しでも振れる
  (最初の COND が遅い harness の偏り) ので、この run からスレッド数の損得は
  言えない。task 119b は同じモデルの decode で 16T が +3.5% と記録しており、
  **16 スレッドを既定にするのは既定モデルに対して 2% 枠を超える危険がある**。

## 6. テスト

- `cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1`
  1173 passed / 0 failed。
- `cargo test --release -p denno-engine --features cuda --test kernel_invariants`
  95 passed / 0 failed。
- 追加テスト: `vnni_indexed_rows_match_gathered_rows_bitwise` (索引付き VNNI が
  詰め直し経路とビット一致)、Q5_1 の n_in を 256 (nb=8) にして 4 ブロック経路を
  f64 参照式と突き合わせる形へ強化。
- `--features cuda` を付けないビルドは **int-auto の時点で落ちる**
  (`crate::cuda` を無条件に使う箇所が 36 errors)。関門は cuda 付きで回した。

## 7. GPU で確認してほしいこと

- 残りの最大の塊は mix (8.47 s、全体の 23%) で、GPU の bf16 GEMM が
  約 2.5 TMAC/s しか出ていない。ここは今回触っていない。
- expert の 24.0 s のうち約 4 s はページキャッシュのミス (majflt 139k) と推定。
  `DENNO_EXPERT_PREFETCH=1` では改善しなかった (t291a)。

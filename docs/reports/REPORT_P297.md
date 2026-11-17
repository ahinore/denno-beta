ADOPT: yes t297a t297c

# REPORT P297 — UD-Q4_K_XL の prefill: task 291 の 2 本を既定 on (3,826 cold prefill -28.9%)

branch: br297 (int-auto 7c2e83f から。br291 は int-auto に未 merge だったので最初に
merge した — br291 が 7c2e83f を既に取り込んでいたため早送り。衝突なし。作業中に
int-auto は動かず (最後にも fetch して 7c2e83f のまま)、merge は不要だった)。
lab: t297a (UD、採否の判定) / t297b・t297c (既定モデル。t297b は host の状態が
壊れていて判定に使わず、t297c を最終 SHA で取り直した)。

## 0. 結論

- **UD-Q4_K_XL の 3,826 cold prefill は 55.30 -> 39.30 s (-28.9%)、39.79/38.80 s の
  2 本** (lab t297a)。target の 20% を満たす。分母は同じ構成の繰り返し b1 (54.52/
  56.08、median 55.30)。最初の COND b0 は harness の偏りで 62.17 s なので使わない
  (b1 は b0 より -11.1% = task 文の「最初の COND は偏る」どおり)。
- **UD の warm short decode は 22.0/21.0 -> 20.4 ms/tok** (b0/b1 -> new)。悪化なし。
- errors 0、text sane (下に引用)。md5 は b0/b1 内で SAME、new は `b` と
  `c2000_turn2` だけ DIFF (`c2000_cold` は SAME)。Q5_1 の積和を 128 要素まとめて
  f32 に落とす丸め順変更 (prefill の丸め契約外、2026-10-28 のオペレーター指示) で、
  値の差は 1e-7 台、token 選択が数か所で変わる。
- **既定モデル (swift15-abl-IQ3_S) は md5 4 本すべて一致** (lab t297c の off / on /
  scr / offb が同一: a:65ededc3 b:f18c75fc c2000_cold:4e4cf144 c2000_turn2:07dc79d2)。
  decode も on が off より -5.4% (39.6 対 41.9 ms/tok)、cold prefill は -19.3%。
- 採った change は 2 つだけ:
  1. task 291 のゲート 2 本を既定 on (作業域の使い回し 2048 MiB / Q5_1 down の
     4 ブロックまとめ食い)。task 291 では target 30% に届かず既定 off だった。
  2. prefill の終端で MoE の作業域を OS へ返す (decode 中に 782 MB を抱えない)。
     t297b で既定モデルの warm short decode が on だけ +12% だった原因がこれで、
     返すようにしたら t297c で -5.4% に戻った。

## 1. 前提の測り直し (task の Do 1) — int-auto 7c2e83f + ゲート off、PROFILE=1

lab t297a の b1 (b0 の繰り返し。3,826 cold prefill 54.52 s / 2 本目 56.08 s):

| 段 | 実測 (median) | 割合 |
| --- | --- | --- |
| experts (CPU、flat MoE) | 41.90 s | 76.8% |
| mix (GPU、Q8_0 quant GEMM) | 8.48 s | 15.5% |
| other (embed / ple / combine / layout) | 2.90 s | 5.3% |
| ssm | 0.66 s | 1.2% |
| shared | 0.27 s | 0.5% |
| full | 0.25 s | 0.5% |
| router (GPU) | 0.07 s | 0.1% |

`[pf-host] experts_wall=41895.7ms | plan=24.9 prefetch=1.0 | other=41895.0 (100.0%)`。
ゲートを入れた new は experts 27.0 s (-35.6%)、mix 8.47 s (不変)、other 3.11 s、
wall 39.8 s。**残りの最大の塊は依然 experts (68%) で、次が mix (21%)。**

## 2. 変えたこと

### 2.1 task 291 の 2 本を既定 on にした (`6359cf0`)

- `DENNO_MOE_SCRATCH_KEEP_MB` の既定 512 -> **2048 MiB**。実形の作業域 782 MB が
  毎層捨てられ、`vec![0.0; ...]` のゼロ埋めを 48 回払っていた (lab t291a: 149 ms/層)。
- `DENNO_Q51_BLK4` の既定 off -> **on**。Q5_1 (down) の VNNI GEMM を 4 ブロック
  (128 要素) まとめ食いにして水平加算を 1/4 に。
- どちらも `=512` / `=0` で切り戻せる。解釈は純関数 (`moe_scratch_keep_mb_from_value`
  / `q5_1_blk4_from_value`) に切り出し、既定をテストで固定した。

### 2.2 prefill の終端で作業域を OS へ返す (`1ee4025`)

- `moe_scratch_release()` を足し、`PfPoolGuard` の drop (= prefill の終端) から呼ぶ。
  使い回しが効くのは 1 回の prefill の中 (48 層 × 同じチャンク) だけなので、終端で
  `MOE_SCRATCH` を `None` に戻す。次の prefill は先頭で 1 回だけ確保し直す。
- 根拠: lab t297b (この SHA の前、既定モデル) で warm short decode が
  off 40.0 / on 45.1 / offb 38.8 ms/tok。Q5_1 blk4 は既定モデルの decode 経路を
  通らない (expert GEMV は VNNI GEMM を通らない) ので、原因は作業域の持ち越しと
  推定した。返すようにした t297c では off 41.9 / on 39.6 / scr 39.5 / offb 42.4 で
  **on が off より速い**。t297b の +12% は再現しない (推定どおり)。
- UD の prefill は 1 タイル = 2 チャンクで、48 層の使い回しはタイルの中で完結する。
  終端で返しても prefill の反復 (149 ms/層) は消えない。

## 3. 数値の扱い

- 2.1 の作業域の使い回しは**値不変** (lab t291a の md5 全 COND SAME、t297c の既定
  モデルも SAME)。
- 2.1 の Q5_1 blk4 は prefill の丸め契約の外。int32 の積和を 128 要素まとめて f32 に
  落とすため累積順が変わる。lab t297a の md5 は `c2000_cold` SAME、`b` と
  `c2000_turn2` が DIFF。既定モデルは Q5_1 の expert を持たないので影響しない
  (t297c で 4 COND 同一 md5)。
- 2.2 はメモリの返却だけで**値不変** (t297c の md5 4 本一致)。
- 決定性・行チャンク不変・par/seq 一致: 既存テスト (moe_flat_matches_legacy_bitwise、
  gemv_batch_matches_gemv_bitwise など) は緑のまま。Q5_1 blk4 の許容比較テストも緑。

## 4. 次の 2 大塊を触らなかった理由 (測ってから決めた)

### 4.1 CPU の expert GEMM は「この定式の上限」に居る (実測)

`moe_ud_prefill_microbench` (実形 ne 512 / k 8 / nt 3826、16 スレッド、rep の min。
今の HEAD で測り直した):

| 構成 | 1 層 | 48 層 |
| --- | --- | --- |
| ゲート off (`DENNO_MOE_SCRATCH_KEEP_MB=512 DENNO_Q51_BLK4=0`) | 633.1 ms | 30.4 s |
| ゲート on (既定) | 387.5 ms | 18.6 s |
| ゲート on + 水平加算を全部削った実験版 | 361.6 ms | 17.4 s |

つまり Q4_K VNNI カーネルの水平加算 (hsum16 + hsum8、1 (行, トークン, ブロック) に
2 回) を**丸ごと消しても -6.7%** しか出ない。1 層 = 150.5 GMAC (3,826 トークン ×
8 expert × 3 行列 × 640×2560) を 387.5 ms で回していて 388 GMAC/s、1 単位
(行, トークン, ブロック) は 256 MAC。残りは 4 vpdpbusd + 4 cvt + 4 fmadd という
命令構成そのものが律速 (推定)。CPU 側で抜けるのは「定式を変える」こと (例: 活性の
量子化を 1 トークン 1 スケールにして、トークン内のブロック累積を 1 回の縮約に
する) だけで、精度を落とすので採らなかった。→ CPU の expert はここで打ち止め。

### 4.2 GPU の batched expert GEMM / expert upload は既に測って負けている

- P40 の grouped GEMM (`DENNO_MOE_GROUPED_GPU`) はこのモデルの型 4 種
  (Q8_0 / Q5_1 / Q4_K / Q5_K) を既に受けるが、lab t266r2 で **4.45 倍遅い**
  (2,000 token cold: 41.98 -> 186.83 s)。原因は「選択 expert の重みを 1 本に詰める」
  pack で、実測 414 MiB/層。このタスクの時間内に pack を廃した転送カーネル
  (重みを stream しながら m 行 GEMM) を書いて実機で当てるのは、このコンテナに
  GPU / nvcc が無く (検証は lab の 3 ラウンドだけ)、リスクに見合わないと判断した。
- 台帳でも experts は `launches=0` = prefill の選択 expert は GPU を 1 度も使って
  いない。次の担当者への申し送りは下の 7。

### 4.3 mix (8.47 s) の正体 — 数字だけ出す

lab t297a の new_1 の台帳: `mix wall=8466.0 gpu=8372.8 launches=6720 syncs=192`,
`[mix-qchain] on calls=2689 hits=0` (Q8_0 の融合 chain は decode 専用で prefill は
発火しない)。prefill の mix は 1 層 2 回 (attn 入口 / ffn 入口) × 2 チャンク = 192 回、
1 回 43.6 ms で 1,913 トークン × 13.1 M MAC = 12.5 GMAC → **288 GMAC/s**
(既定モデルが 14k トークンで出している 3.5 TMAC/s の 1/12)。重みは Q8_0 なので
`PfWKind::Quant` の量子 GEMM (16 行ブロック / 128 トークン 1 起動、down 80 ブロック /
up 160 ブロックで SM 170 に足りない) を通る。ここを触るには .cu が要る。

## 5. lab

### 5.1 t297a (UD-Q4_K_XL、depth、PROFILE=1、SHA 6359cf0)

最終 SHA 1ee4025 との差は 2.2 (prefill 終端の free) だけで、prefill 中の経路は同一。
UD の prefill は 1 タイル (チャンク 1913 × 2) なので、終端の free は 1 回だけ
(推定 50 ms 未満 = 0.1% 未満)。下の数字はそのまま最終 SHA の推定値として使える。

| cond | warm short decode ms/tok | 3,826 cold prefill (2 本) | md5 (a / b / cold / turn2) |
| --- | --- | --- | --- |
| b0 (ゲート off) | 22.0 | 70.55 / 53.80 (median 62.17) | 737a73af / 87d8a97c / 73510e1f / 9e32c460 |
| new (既定) | **20.4** | **39.79 / 38.80 (median 39.30)** | 737a73af / 1938059f / 73510e1f / fbaf5055 |
| b1 (b0 の繰り返し) | 21.0 | 54.52 / 56.08 (median 55.30) | b0 と同一 |

`VERDICT new vs b0 OK decode -7.3% prefill -30.7% md5 DIFF errors 0`
`VERDICT b1 vs b0 OK decode -4.4% prefill -36.7% md5 SAME errors 0`
`VERDICT_DEPTH new c2000_cold vs b0 decode +31.5% prefill -36.8%`
`VERDICT_DEPTH b1 c2000_cold vs b0 decode +5.8% prefill -11.1%`

- **目標対比は b1 を分母にする: 55.30 -> 39.30 s = -28.9%** (2 本ずつの実測で、
  new の 2 本は 39.79/38.80 と揃っている)。
- md5 の DIFF は 2.1 の Q5_1 blk4 の丸め順変更による (3.)。`c2000_cold` は SAME。
- text (TXTD、new、3,826 token の cold の続き):
  「ご提示いただいた文書は、LLM推論エンジン（`denno-engine）における **Prefill
  （事前計算フェーズの性能最適化の歴史と教訓を詳細に記録したものです。」
  TXT/TXTB も b0 と同じ書式で sane (B-tree の説明 / Rust の所有権)。
- `c2000_cold` の列の decode (± 同じ 3,826 token のあとの decode) は b0 75.9 /
  new 99.8 / b1 80.4 ms/tok で、**同じ構成の b1 自身が 62〜99 と振れる**指標
  (ページキャッシュと expert の VRAM 当たりの状態で決まる)。この指標では 2% を
  判定できない。warm short decode (採否規則の対象) は -7.3% で悪化していない。

### 5.2 t297b (既定モデル、SHA 6359cf0) — 判定に使わない

off 40.0 / on 45.1 / offb 38.8 ms/tok、cold prefill 47.38 / 38.40 / 48.22 s。
直前の UD の run (104 GB モデル) がページキャッシュを占拠していて、既定モデルの
絶対値が参照 (tint_7c2e83f: 17.9 ms/tok、8.5 s) の 2〜6 倍に落ちた回。この +12% が
2.2 を入れる動機。md5 は 3 COND 同一 (4e4cf144 など。参照 tint の d4b6eefb とは
違うが、この run の中では全部同じ = ゲートは値を変えない)。

### 5.3 t297c (既定モデル、depth、PROFILE=1、SHA 1ee4025 = 最終 SHA)

| cond | warm short decode | 3,826 cold prefill | md5 |
| --- | --- | --- | --- |
| off (ゲート off) | 41.9 | 48.06 | a:65ededc3 b:f18c75fc cold:4e4cf144 turn2:07dc79d2 |
| on (既定) | **39.6** | **38.77** | 同上 (完全一致) |
| scr (blk4 だけ off) | 39.5 | 38.95 | 同上 |
| offb (off の繰り返し) | 42.4 | 48.59 | 同上 |

`VERDICT on vs off OK decode -5.4% prefill +5.3% md5 SAME errors 0`
`VERDICT scr vs off OK decode -5.8% prefill -4.7% md5 SAME errors 0`
`VERDICT offb vs off OK decode +1.1% prefill +4.7% md5 SAME errors 0`
`VERDICT_DEPTH on c2000_cold vs off decode +21.7% prefill -19.3%`
`VERDICT_DEPTH offb c2000_cold vs off decode +9.9% prefill +1.1%`

- **既定モデルの md5 は 4 COND すべて同一** = 値は動いていない。
- 2.2 の後は on が off より速い (t297b の +12% は再現しない)。cold prefill も
  -19.3% (作業域の使い回しは IQ3_S の prefill にも効く)。
- この回も host は遅いまま (decode 40 ms/tok、参照 17.9)。絶対値は参照と比べない。
  同一 run 内の off/on/scr/offb の比較だけを使う。

## 6. テスト

- `cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1`
  **1187 passed / 0 failed** (task 291 の 1186 から +1)
- `cargo test --release -p denno-engine --features cuda --test kernel_invariants`
  **95 passed / 0 failed**
- 追加: `q5_1_blk4_gate_defaults_on` (既定 on と偽値の真理値表)、
  `moe_scratch_keep_mb_defaults_to_2048`、`moe_scratch_release_drops_retained_workspace`
  (prefill 終端の返却)。ゲートの発火そのものは既存の
  `moe_flat_vnni_fires_and_matches_within_tolerance` /
  `vnni_indexed_rows_match_gathered_rows_bitwise` (Q4_K/Q5_1 の VNNI 経路) が押さえる。
- 既知の赤 (`qwen4exp_decode spec_batch_stops_at_eos`、`qwen4exp_params` の
  validate 2 件) はこのツリーでは緑 (task 291 と同じ状態)。

## 7. GPU で確認してほしいこと / 次の手

1. **mix (8.47 s、prefill の 21%)**: 288 GMAC/s。1 層 2 回の mix を 1 回にまとめる
   (attn 入口と ffn 入口を 1 カーネル鎖に)、または Q8_0 の mix 重みを bf16 に
   展開して `mix_pf` (bf16、3.5 TMAC/s) に載せると、推定 8.5 -> 2.5 s (-15%)。
   bf16 展開は精度上も Q8_0 以上 (8 ビット量子化を戻すだけ)。VRAM +1.3 GB。
2. **experts (27.0 s、68%)**: CPU は定式の上限 (4.1)。次は「pack しない」
   grouped GEMM (Q4_K/Q5_1 を stream しながら m 行) のみ。P266 の 4.45 倍遅い原因は
   pack なので、そこを外せば初めて勝負になる。
3. `c2000_cold` のあとの decode は同じ構成でも 62〜99 ms/tok と振れる (5.1)。
   2% 判定に使うなら同一 COND を 4 本以上並べた run が要る。

## 8. 未追跡のまま残したもの

- なし (作業ツリーは clean。REPORT は最終 commit に含める)。

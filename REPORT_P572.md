ADOPT: yes t572b t572c

# REPORT P572 — 既定モデルの decode: **full 層の 56% がデバイス経路から落ちていた**。
# KV ミラーの 1 行追記が「ちょうど 1 行遅れ」のときしか通らないため、一度でも
# 落ちるとその層は永久にホスト経路 (qkv の D2H/H2D + ホスト gate_mul + 同期) に
# 固定されていた。足りない行だけをホスト K/V から埋めて追記をやり直すだけで
# **full 段 -23% / 短い decode -3.8%**、md5 は全 COND SAME。

branch: **br572** = 開始時の `int-auto` **`99d3858e2` (merge: br562)**。
br568 は既に merge 済み。終端で int-auto を fetch し直したが動いていなかった
(追加 merge なし)。

数値はすべて GPU ラボの実測 (lab 名と測った SHA):
**t572a = 99d3858e2** (短い文脈の attention カーネル A/B)、
**t572b = eb282353c** (ミラー埋め戻しの A/B + 診断。fill 既定 off)、
**t572c = 028cea3a7** (既定 on の検証。off/on/off2)。3 run とも
`model=flash-next-UD-IQ3_XXS ptx=none profiler=none`、`PROFILE=1`、SCENARIO=depth8k。
土台の前提は lab t564a / t568a (同じ 99d3858e2)。

## 0. 結論 (6 行)

1. **採る (`ADOPT: yes`)。** 既定 `DENNO_FULL_DEC_MIRROR_FILL` を on にした
   (切り戻しは `=0`)。**短い decode 10.4 -> 10.0 ms/tok (-3.8% / -4.0%) を 2 ラウンド**、
   **full 段 1.23-1.26 -> 0.94-0.96 ms/token (-23% / -24%)**、14k 後と prefill は
   不変、md5 は全 COND SAME、errors 0、text 健全 (§5)。
2. **狙いは「段で 2% 以上・他が 1% 超悪化しない」を満たす。** 他段は ±1% 以内
   (experts -1.0%、ssm ±0%、mix -2.5%、shared/router/head 不変)。
3. **選んだ理由**: 新しい段プロファイル (§1) で **full 1.25 ms = decode の 12%** が
   「短い文脈では一度も測られていない唯一の大きい段」だった。P520 は 14k の
   flashdec だけ、P427/P500/P504 はカーネル選択だけを見ており、**短い文脈で
   flashdec が門 (MIN_CTX=1024) で切られて split 経路 + 層ごとのホスト往復が
   走っていること**は誰も測っていなかった。
4. **t572a (カーネル選択) は空振り**: 短い文脈で flashdec を強制 (`MIN_CTX=0`) は
   +0.8%、単発カーネル (`Q4_GPU_ATTN_KERNEL=decode`) は ±0%。sing は GPU を
   0.79 -> 0.66 ms にしたが sync が 0.87 のまま増えて壁が動かない = **犯人は
   カーネルではなく経路**だった (§4)。
5. **正体はミラーの 1 行遅れ**。`mirror_append` は「ミラーがちょうど len-1 行」の
   ときだけ通り、ホスト経路は自分の push 後の長さまでしかミラーを同期しないので、
   1 行足りないとその層は**永久に**ホスト経路へ落ちる。t572b の診断 COND は
   **mirror=1537/1537 (全呼び出しがミラーで落ちた)**、fill on では 1 行も落ちない (§3)。
6. 本タスクの lab は 3 ラウンドで終わり (t572a/b/c)。**次の一手は .cu**:
   expert の silu+sum 融合 (推定 -3.5%) と full 層 wo の 553 GB/s (§6)。

## 1. 現在地の段プロファイル (warm short decode、99d3858e2)

出処: lab t572b の `b` / `b2` と t572c の `off` / `off2` (どちらも `PROFILE=1`、
2 番目の `[q4ledger]` ブロック、ms/token・0.1 ms 刻み)。4 本とも 10.30-10.36。

| 段 | ms/tok (4 本) | 何が入っているか | 過去に触った REPORT |
| --- | --- | --- | --- |
| experts | **2.87-2.89** | GPU 2.30 (gate+up 22.2 / up 1.6 / silu 3.2 / down 16.6 / sum 4.2 us/層) + ホスト糊 0.57 | P388 (ミス CPU を起動に重ねる)、P526 (seg 融合 -0.9/-4.4%)、P534 (down 16B -5.9%)、P562 (gu16)、P564/P568 (silu+sum は .cu) |
| ssm | **2.77-2.82** | 36 層 x 4 起動。qkv dn8 33.6 us (1.33 TB/s) / out dn7 16.0 (1.04) / scan 12.4 / ba 7.4 | P504 (out を dn7 -2.1%)、P496 (dn8 rows 掃引)、P392c (入力をデバイスから)、P568 (中身の内訳) |
| mix | **1.95-1.98** | hc mix 97 回。cpu 1.38-1.40 = null ストリームの H2D 待ち | P431 (stream -0.7%)、P464 (NOWAIT)、P282/P292、P568 (待ちであることの整理) |
| **full** | **1.23-1.26** | 12 層。gpu 0.77-0.81 / sync 0.84-0.86 (15 回) / cpu 0.24 | P520 (14k の flashdec 命令削減)、P427 (dn6 -10.5%)、P500 (qkv dn8 負け)、P504 (dn7 負け)、P460 (14k の cbg 融合) |
| shared | 0.58-0.59 | 4 GEMV/層 (Q8_0 gate/up/down)。sh_begin 0.41 = 発行のホスト仕事 | P319 (v5 へ -2.4〜-2.7%)。以後誰も触っていない |
| head | 0.46 | Q8_0 語彙 GEMV 0.43 (674 MB = 1.57 TB/s = 天井) + 0.99 MB D2H | P427 (dn6 rows=4 -16.4%)、P504 (dn7 -23%) |
| router | 0.16-0.17 | gate_inp のホスト GEMV 48 回 | P307/P364 (ROUTER_IN_MIX) |

**なぜ full を選んだか**: 大きい順に experts (2.88) / ssm (2.80) / mix (1.96) /
full (1.25)。experts と ssm は P568 が内訳まで出して「残りは .cu」で閉じており、
mix は P568 が「待ち」と説明済み。**full だけが「短い文脈の経路」を誰も
プロファイルしていない** (flashdec は `ATTN_FLASHDEC_MIN_CTX=1024` で切られ、
128 token では split 経路が走る)。しかも decode の 12% で、1 段で 2% を
取れる唯一の現実的な的だった。

## 2. 段の中身 (full、t572b `d` = fill off の `[dec-kprof-avg] full band=0`)

12 層 x 64 窓、us/層 (cuEvent の固定費 2-4 us/op を含むので絶対値ではなく比で読む):

| 項目 | 値 | 備考 |
| --- | --- | --- |
| wall | 187-203 | |
| kern | 87-99 | |
| copy | 14-19 | ホスト往復の H2D/D2H |
| gap | 44-56 | |
| wait | 94-113 | 層末の同期 |
| launch | 16-24 | |
| steps | 71-86 | ホスト仕事 |

窓の中で **fd_attn (デバイス attention) に到達したのは 768 窓中 336**、残り 432 は
`gate_mul` (ホスト 11.2 us) / `rope_q` / `kv_split` を払う**ホスト経路**だった。
`[full-dec-dbg]` (t572b `d`) は `mirror=1537/1537` = **device 経路に入った全呼び出しが
ミラーで落ちた**ことを出している。

## 3. 直したこと (cuda.rs 2 箇所 + attn 1 箇所)

1. `dec_full_dev::Call` に `k_host` / `v_host` (= `cache.k(layer)` / `cache.v(layer)`、
   `len-1` 行ぶん) を足した。**値はデバイス経路が D2H して push したものと同じバイト列**。
2. `forward_inner` の 3) ミラー追記: `mirror_append` が `None` を返したら、ミラーが
   何行持っているかを読み (`q4_attn_mirror_have_rows`)、遅れが 1..=64 行なら
   **足りない行だけ** `q4_attn_mirror_prepare` で H2D し、追記をやり直す。
   1 回で復帰し、以降は普通の D2D 追記に戻る。上限 (`DENNO_FULL_DEC_MIRROR_FILL_MAX`、
   既定 64) を超える遅れは従来どおり断る (巨大な H2D を撃たない)。
3. 発火を数える: `MIRROR_FILLS` / `MIRROR_FILL_ROWS` を `[full-dec-dbg]` に足した
   (`DENNO_FULL_DEC_DBG=1` のときだけ)。テストは純関数 2 本
   (`mirror_fill_gap_truth_table` / `mirror_fill_gate_truth_table`)。
   `task468_full_mix_nowait_wiring` の文字列一致が `cur,\n x_dev,` を見ていたので
   `k_host`/`v_host` の配線検査を足す形に直した。

**数値の扱い**: 埋める行はホスト K/V のコピーで、デバイスが書く行と同一バイト列。
attention の丸め順・カーネル・発行順は 1 つも変えていない。だから md5 は
全 COND・全 6 文脈で SAME (狙って変えた丸めは無い)。

## 4. 落とした手 (t572a、99d3858e2)

| COND | decode ms/tok | full 段 | 判定 |
| --- | ---: | ---: | --- |
| b / b2 (基準) | 10.4 / 10.4 | 1.26 / 1.26 (gpu 0.79, sync 0.87/15) | |
| fdec (`ATTN_FLASHDEC_MIN_CTX=0`) | 10.5 | 1.33 (gpu 0.77, sync 0.95/15, 起動 57) | **+0.8% = 負け** |
| sing (`Q4_GPU_ATTN_KERNEL=decode`) | 10.4 | 1.28 (**gpu 0.66**, sync 0.87/15) | ±0% = 壁が動かない |

`VERDICT fdec vs x0 OK decode +0.8% prefill -0.9% md5 SAME errors 0`
`VERDICT sing vs x0 OK decode +0.0% prefill -1.8% md5 SAME errors 0`

sing が GPU を 0.79 -> 0.66 ms に減らしても壁が動かない = **full 段の壁は
カーネルではなく「層ごとのホスト往復と同期」**。ここから §3 に進んだ。
8k 後 (c8000_turn2) はどの COND も 11.2 (b だけ 11.6 が外れ値) で、門の外なので
当然変わらない。

## 5. 採否の本体 (t572b / t572c、2 ラウンド)

段 (ms/token、`[q4ledger]`、同じ run の中の対):

| run | COND | full | step | experts | ssm | mix | shared | head | router |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| t572b | b | 1.26 | 10.36 | 2.87 | 2.82 | 1.95 | 0.59 | 0.46 | 0.17 |
| t572b | **f (fill=1)** | **0.96** | **10.00** | 2.87 | 2.85 | 1.89 | 0.58 | 0.46 | 0.17 |
| t572b | b2 | 1.24 | 10.35 | 2.88 | 2.79 | 1.98 | 0.59 | 0.46 | 0.17 |
| t572c | off | 1.23 | 10.30 | 2.89 | 2.77 | 1.97 | 0.59 | 0.46 | 0.17 |
| t572c | **on (既定)** | **0.94** | **9.96** | 2.86 | 2.80 | 1.92 | 0.58 | 0.46 | 0.16 |
| t572c | off2 | 1.24 | 10.34 | 2.88 | 2.80 | 1.96 | 0.59 | 0.46 | 0.17 |

full 段の中身の変化 (t572c off -> on): gpu 0.77/0.81 -> 0.56、**sync 0.84/15 -> 0.72/12**、
**h2d 0.18 MB/10 -> 0.01 MB/0**、d2h 0.34 -> 0.17、cpu 0.25 -> 0.09。ホスト経路の
qkv 往復 (10 回/トークンの H2D) が消えたことが台帳に出ている。

decode の中央値 (RESULT.md の表):

| run | 基準 | 候補 | 差 |
| --- | ---: | ---: | ---: |
| t572b | b 10.4 / b2 10.4 | f 10.0 | **-3.8%** |
| t572c | off 10.4 / off2 10.4 | on 10.0 | **-3.8% / -4.0%** |

`VERDICT f vs x0 OK decode -4.1% prefill +0.0% md5 SAME errors 0` (t572b)
`VERDICT on vs x0 OK decode +0.4%` / `off +3.5%` / `off2 +3.9%` (t572c、x0 は既定 on の
捨て COND なので on と一致する = 既定が効いている証拠)

14k / prefill (t572c、同じ run): c8000_cold prefill 5.15 / **5.21** / 5.11 s、
c8000_turn2 decode 11.4 / 11.4 / 11.2、c2000_turn2 decode 11.3 / 11.6 / 11.4
= **位置ドリフト ±2% の内側で、悪化は無い**。prefill の中央値は 0.55 s で 3 COND 同値。

**text (t572c `on` の実出力、抜粋)**:
`# B-Trees: A Comprehensive Explanation / ## 1. What is a B-Tree? / A **B-tree
(Balanced Tree) is a self-balancing tree data structure that maintains sorted
data and allows searches, sequential access, insertions, and deletions in
logarithmic time.`
`TXTB: Rust の「所有権（Ownership）とライフタイム（Lifetime）は、Rust を学ぶ上で
最も重要であり、かつ最も難しい概念です。`
`TXTD: ご提示いただいた文書は、LLM推論エンジン(denno-engine)における Prefill
(事前計算)フェーズの性能最適化の歴史を詳細に記録した技術レポートです。`
md5 は a:edaa84ad b:1938059f c2000_cold:9b28e018 c2000_turn2:8b02d5c3
c8000_cold:672fec07 c8000_turn2:199f23b3 で **全 COND SAME**、errors 0。

## 6. 残っている手 (次のタスクへ)

- **expert の silu + 和の融合** (silu 3.2 + sum 4.2 us/層 = 0.36 ms/token、推定 -3.5%)。
  .cu が要る。P564/P568 と同じ結論。
- **full 層の wo GEMV が 553 GB/s** (k=6144 n=2560 Q8_0、dn6 で 30.2 us)。qkv は
  1.2 TB/s、ssm の dn8 は 1.33 TB/s なので 2 倍の開きがある。dn6/dn7/dn8 は
  P427/P500/P504 が試して dn6 が最良なので、**同じ形で速くするには .cu**。
- full 段の残りは `sync 0.72/12` (層あたり 60 us の GPU 待ち) = カーネルの中。
- shared の `sh_begin` 0.41 ms/token はドライバ呼び出し (4 本/層) が本体で、
  これも畳むには .cu。
- ミラーの埋め戻しが**効かなかった回の内訳**は今回取れていない (`fill` の
  カウンタは `note_fallback` 側にしか出ないので、fill が全部成功すると
  1 行も出ない)。次に測るなら「fill が何行埋めたか」を成功時にも 1 行出す。

## 7. テスト (このブランチ)

- `cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1`
  → **1414 passed / 0 failed / 103 ignored** (新規 2 本を含む)
- `cargo test --release -p denno-engine --features cuda --test kernel_invariants
  -- --test-threads=1` → **153 passed / 0 failed**
- `cargo build --workspace --release` (cuda feature 無し) → rc=0
- 既知の赤 (`qwen4exp_decode spec_batch_stops_at_eos`、`qwen4exp_params` の
  validate 2 件) は cuda feature 付きの lib では出ない (この木では緑)。

## 8. GPU で確認してほしいこと

1. **本番形 (serve_flash、長い文脈) で `[q4ledger]` の full 段と sync 回数**。
   今回の測定は depth8k の短い decode が主で、14k は 1 点だけ (§5)。
2. `[full-dec-dbg]` に `fill=<回数> fill_rows=<行数>` を**成功時にも**出せるように
   したうえで、長い文脈で埋め戻しが 1 回で済んでいるか (毎トークン埋めていないか)。
3. ミラーが 64 行より深く遅れる場面 (長い prefill の直後・spec の巻き戻し後) で、
   まだホスト経路に落ちていないか。

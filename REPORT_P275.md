ADOPT: no t275c

# REPORT P275 — IQ3_S decode の次の 2 枚 (expert GEMV の表 / mix の norm+inject)

(br275 = int-auto 2a7dbf71e + br269 の merge。br269 は int-auto にまだ入っていないので
 merge して土台にした。既定の挙動は int-auto と同じで、このタスクの変更はすべて
 既定 off のゲートの下にある。)

## 1. 前提と測り方 (Do の 1)

REPORT_P269 の表 (1.1 の台帳と 1.2 のイベント表) をそのまま前提にした。int-auto は
`2a7dbf71e` (br271 の merge) で、task 269 の lab `t269c` の base と同じ系列にある。
測り直しは lab の base COND で行い、P269 の値 (短い decode 19.4 / 14k の次 18.1) と
同じ土俵であることを確かめた (t275c の base2 = 19.4)。

`[q4ledger]` / `[exp-iq-kprof]` の読み方も P269 のまま。今回足したのは
`[exp-iq-kprof] 型ごと` の GB/s (表の幅を変えたので、そこが動いたかを見る)。

## 2. 変えたこと (Do の 2)

P269 の 1.3「どこが取れるか」の 1 番目と 2 番目を取った。どちらも既定 off のゲート。

### 2.1 expert GEMV のグリッド表を詰める (`DENNO_EXP_IQ_V3`)

`kernels_exp_iq_v3.cu` (新規)。`kernels_exp_iq_v2.cu` と同じ segment 表・同じ起動形
(4 行/ブロック)・同じアキュムレータ 4 本で、**shared に写す表だけ**を小さくする。

| 型 | v2 (いま) | v3 | 根拠 |
| --- | --- | --- | --- |
| IQ3_S | 512 本 x u64 = 4 KB、LDS.64 | 512 本 x **u32** = 2 KB、LDS.32 | 512 本の u64 の**上位 32 bit は全 0** で、どの経路も読まない (`kquant.rs` の IQ3S 分岐も `g1[j]` / `g2[j]` (j<4) だけ、v1/v2 も同じ) |
| IQ2_S | 1024 本 x u64 = 8 KB、LDS.64 | 1024 本 x **u16** = 2 KB、LDS.16 | 8 バイトすべて使うが、現れる値は **{8, 25, 43} の 3 種類だけ**。1 値 2 bit に詰める (code 3 は未使用) |

IQ2_S の復元は `p0/p1/p2 = db * {8,25,43}` を 1 度だけ作って code で選ぶ。8 は 2 の冪
なので `db*8` は厳密、25/43 は v2 と同じ 1 回の丸め。よって**値は v2 と一致する**。

`gen_exp_iq_tables.py` が詰めた表を出す。値の前提 (IQ2_S が {8,25,43} だけ / IQ3_S の
上位 32 bit が 0) が崩れたら**生成時に止まる**ようにした。

### 2.2 mix の norm+inject を 2 ブロック系に割る (`DENNO_DEC_MIX_NORM_PAR`)

P269 の 1.2 で `dno_mix_norm_inj` (mix 1 回 10.3 us、**grid = 1**) が mix の中で
いちばん大きい 1 本だった。inject は「正規化後の全ストリーム (k = hc*e)」を読むので
1 ブロックに閉じていたが、正規化した x は**以降の down / hc_mixed が読むだけ**で
書き換えない。したがって inject は down の epilogue へ移せる (順序を変えても値は同じ)。

    v2 (fold):        norm_inj (grid=1) → down → lo_apply → up → hc_mixed
    v3 (norm par):    dno_mix_norm (grid=hc) → down+inject → lo_apply → up → hc_mixed

`kernels_mix_dev3.cu` (新規) の `dno_mix_down_bf16_inj` は `kernels_mix_down.cu` の
`dno_mix_down_bf16` と同一の本体 + blockIdx.x < hc のとき inject の行を 1 本。
inject の木は `dno_mix_inj` (= `dno_mix_reduce256`) と同じ 256 スレッド固定。
**起動回数は増えない** (norm は kernels_mix_dev.ptx の既存カーネル)。
inject が読むのは `out` の**先頭** (ストリーム 0) で、v2 も 2 本版もそうなので
そこは変えていない (g でずらすと値が変わる)。

## 3. lab

### t275a — PTX ビルド失敗 (測定なし)

`gen_exp_iq_tables.py` が u16/u32 の値を `1033uu` と出していた (emit 側で接尾辞を
もう一度付けていた)。`kernels_exp_iq_tables.h` の生成を直した ((2/n) の commit)。

### t275b — SHA fad5cdcbc、PTX=exp_iq_v3 mix_dev3

toss / v3 / npar / both / bothba / base2。**この回は host の load が 1.75 で、
最初の COND が強く bias した** (同じ設定の toss と base2 が 14k の次で 19.7 と 18.2)。
窓ごとの内訳 (`[q4ledger]` の experts と mix) だけを読んだ:

| | base2 | v3 | npar | both |
| --- | --- | --- | --- | --- |
| experts ms/tok (14k の次) | 4.63 | 3.96 | 4.61 | 3.91 |
| mix ms/tok (14k の次) | 3.62 | 3.60 | 3.56 | 3.55 |
| IQ2_S GB/s | 265 | **425** | 265 | **428** |
| IQ3_S GB/s | 307 | **513** | 307 | **510** |

### t275c — SHA 90eeab485 (**commit 済み PTX**、PTX= なし) — 最終

toss / both / bothba / base2。

| COND | 短い decode | 14k cold | 14k の次 | prefill 14k | md5 | 判定 (vs toss) |
| --- | --- | --- | --- | --- | --- | --- |
| toss | 19.5 | 19.2 | 18.5 | 16.70 s | 基準 | (基準) |
| **both** | **18.9** | **18.2** | **17.5** | 16.30 s | SAME | **-3.0%** |
| bothba | 18.3 | 19.2 | 19.1 | 16.14 s | DIFF | -5.8% (深い点が悪化) |
| base2 | 19.4 | 18.9 | 18.2 | 16.27 s | SAME | -0.2% (ノイズ床) |

`VERDICT both vs toss OK decode -3.0% prefill -2.4% md5 SAME errors 0`
`VERDICT bothba vs toss OK decode -5.8% prefill -2.4% md5 DIFF errors 0`
`VERDICT base2 vs toss OK decode -0.2% prefill -0.8% md5 SAME errors 0`
`VERDICT_DEPTH both c2000_turn2 vs toss decode -3.9% prefill -2.4%`
`VERDICT_DEPTH both c8000_cold vs toss decode -5.7% prefill -2.4%`
`VERDICT_DEPTH both c8000_turn2 vs toss decode -5.5% prefill -2.0%`
`VERDICT_DEPTH bothba c2000_turn2 vs toss decode +1.1% prefill -2.4%`
`VERDICT_DEPTH bothba c8000_turn2 vs toss decode +3.0% prefill -4.0%`
`VERDICT baseline toss errors 0 text ok`

窓ごとの decode (ms/tok、`decode 秒 / 出力トークン`):

| 窓 | toss | both | bothba | base2 |
| --- | --- | --- | --- | --- |
| a1 (28 tok) | 19.45 | 18.87 (-3.0%) | 18.32 | 19.41 |
| a2 (同) | 17.11 | 16.25 (-5.0%) | 15.43 | 17.03 |
| c2000_turn2 | 17.81 | 17.11 (-3.9%) | 18.01 | 17.70 |
| c8000_turn2 (14k) | 18.52 | 17.50 (-5.5%) | 19.06 | 18.20 |

内訳の実測 (t275c、c8000_turn2 の `[q4ledger]` / `[exp-iq-kprof]`、base2 → both):

- experts 4.63 → **3.90 ms/tok** (-0.73)
- mix 3.61 → 3.56 (-0.05)
- ssm 4.20 → 4.21、full 2.23 → 2.24、router 1.28 → 1.27、head 0.74 → 0.74 (動かない)
- IQ2_S 263 → **426 GB/s** (7.5 → 4.6 us/本)、IQ3_S 307 → **516 GB/s** (7.6 → 4.5)
- IQ3_XXS 470 → 469、IQ4_XS 592 → 580 (v3 の担当外。動かないのが正しい)

本文は健全 (`TXT both`: B-tree / B+tree の説明、`TXTB both`: Rust の所有権とライフタイム)。
md5 は both が全 6 本 SAME、errors 0。bothba だけ DIFF (`DENNO_GDN_BA_GPU` の丸め。
P269 の 3 章と同じ理由)。

## 3.1 採否

**ADOPT: no t275c。** 目標「短い decode が 6% 以上」に対して **-3.0%** (14k の次は
-5.5% で目標側は満たすが、短い点が届かない)。両方のゲートは**既定 off** のまま残す
(コードとテストは残す。次のタスクが前提にしてよい)。

- 2 つとも**ビット一致** (md5 SAME)。切り戻しは env 1 本ずつ。
- 実測 -0.78 ms/token (19.5 → 18.7 相当)。6% (約 1.2 ms) には 0.4 ms 足りない。
- `DENNO_GDN_BA_GPU` (br269 から持ち込み、既定 off) を足すと短い点は -5.8% まで行くが、
  **14k の次が +3.0% 悪化**する (t269c の +2.6% と一致)。深い点の規則に反するので採らない。

**足りない 0.4 ms の居場所** (t275c の 14k の次、both の台帳): ssm 4.21 / experts 3.90 /
mix 3.56 / full 2.24 / router 1.27 / head 0.74 / shared 0.64。このタスクで触った 2 つを
除くと、残るのは「(a) ssm の GDN 3 カーネル待ち 3.2 ms」「(b) mix の down 9.3 us
(704 GB/s) + up 7.8 us (840 GB/s)」「(c) expert の down 15.1 us / d2h 9.5 us / silu 4.5 us」。
(a) は残差ストリームごとデバイス常駐にする必要があり (t245b で別案は 0%)、本タスクの
範囲外。(b) は 1 本 6.55 MB を 320 ブロックで読んでいて、k 方向に割れば 1 TB/s が
見えるが丸め順が変わる (推定 -0.3 ms)。(c) の silu は n=12,800 で 4.5 us なので
ほぼ起動固定費で、down に畳むと新しい down カーネル (6 型) が要る (推定 -0.2 ms)。

## 4. 数値の扱い

- `DENNO_EXP_IQ_V3=0` / `DENNO_DEC_MIX_NORM_PAR=0` (既定) では 1 bit も変わらない。
  どちらも**on でも値が変わらない**ことを lab の md5 (全 6 本 SAME) で確認した。
- **既定の経路が int-auto と 1 bit も変わらない**: int-auto `2a7dbf71e` 自身の lab
  (`done/tint_2a7dbf71e` の base) の md5 は c2000_cold:b2057e53 / c8000_cold:44178e66 /
  c8000_turn2:44634658 で、t275c の toss と base2 の 6 本は**すべてこれと一致**する
  (t269c の基準 1effbda4 / 0739f0d2 とは違うが、それは int-auto がこの間に動いた分で、
  このタスクの変更ではない)。
- 決定性・行チャンク不変・par/seq 一致・アンカーは触っていない。
- IQ2_S の `p0/p1/p2` は `db*8` (2 の冪なので厳密) と `db*25` / `db*43` (v2 と同じ 1 回の
  丸め)。`(k==0)?p0:((k==1)?p1:p2)` は v2 の `db * gv` と同じ値。

## 5. 見積り (推定) と実測

- 着手前の見積り: expert GEMV で -1.0 ms、mix の norm で -0.4 ms、合わせて -1.4 ms (-7%)。
- **実測: expert -0.73 ms (見積りの 7 割)、mix -0.05 ms (見積りの 1/8)**。合計 -0.78 ms で
  短い decode -3.0% / 14k の次 -5.5%。
- 見積りが外れた理由:
  - expert は IQ2_S の GB/s が 263 → 426 と 1.6 倍になったが、experts 区間には
    down 15.1 us / d2h 9.5 us / silu 4.5 us / sum 5.3 us が残り、gate の取り分は
    区間の 4 割弱だった (P269 の表の 35 us は 4 発のうちの 1 発)。
  - mix は `dno_mix_norm_inj` の 10.3 us のうち、1 SM に閉じていたのは正規化の 2 段だけで、
    inject (k = hc*e = 10,240 の内積 = 256 スレッドで 40 回のループ) が元から
    down と同じくらいの重さだった。移した先の down も同じ 40 回を持つので、
    差は「10.3 → 約 6.0 + down の +1.5」で -0.05 ms/token にしかならなかった。

## 6. テスト

- `cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1`
  → **1157 passed / 0 failed** (100 ignored)
- `cargo test --release -p denno-engine --features cuda --test kernel_invariants`
  → **92 passed / 0 failed**
- 追加したテスト:
  - `exp_iq::tests::v3_gate_and_kernel_names` (ゲートは既定 off / PTX の引数 8 本 /
    v3 が差し替えるスロットは 0 と 2 だけ)
  - `exp_iq::tests::compact_tables_match_originals` (**詰めた表が元の表と同じ値を与える**
    ことを 1024x8 と 512 本すべてで照合。GPU 無しで確かめられる部分)
  - `kernel_invariants::exp_iq_v3_uses_only_compact_tables` (v3 が元の 8 KB / 4 KB の
    表を読んでいないこと・build_ptx.sh に載っていること)
  - `kernel_invariants::mix_dev3_down_inj_is_wired` (down の本体が
    `dno_mix_down_bf16` と同一・inject が 256 スレッド固定・barrier が分岐しない)

## 7. GPU で確認してほしいこと

1. `DENNO_EXP_IQ_V3=1` で `[exp-iq-kprof]` の 0/0/4 と 2/2/4 の GB/s が 425 / 515 前後まで
   上がる (t275b / t275c の実測)。下がるようなら PTX の再生成か shared の bank 衝突を疑う。
2. `DENNO_DEC_MIX_NORM_PAR=1` で `[q4ledger]` の mix の sync が 2.65 → 2.59 ms/tok に
   わずかに減る (起動数は変わらない: 388/call のまま)。
3. 両方 on で md5 が SAME のまま (t275c で確認済み)。

## 8. merge

int-auto (2a7dbf71e) は br269 の merge で取り込み済み。このタスクのあと int-auto は
動いていない (t275c の時点)。`build_ptx.sh` の targets_all は 2 本追加した
(`exp_iq_v3` / `mix_dev3`)。int-auto 側がカーネルを足していたらこの行で衝突する。

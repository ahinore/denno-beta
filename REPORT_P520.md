ADOPT: yes t520a t520b t520c

# REPORT P520 — 既定モデル decode: flash decoding の**1 key あたりの命令**を削った
# (2 key 同時 + max が動かない key の乗算省略)。カーネル単体 14k 65.5 -> 61.4 us/層
# (-6.2%、3 run で同値)、**full 段 14k -2.9 to -4.1%**。decode 全体は 14k -0.4% で
# タスクの目標 (2%) は未達。演算子の段の規則 (段で 2 ラウンド以上 2% 以上、他は 1% 以内)
# で採る。値はビット一致 (md5 は 3 run × 全 COND × 全 6 文脈で SAME)。

branch `br520` = 開始時の `int-auto` **7510bc00a**。終端で int-auto を fetch し直したら
**ac0d7c1ad (merge: br561)** に動いていたので merge した (衝突なし。561 は新しい .cu を
足していないので `targets_all` の末尾は衝突しなかった)。lab が見たのは
**b2a64c179** / **9c7ecfefb** / **454d6acb8** (いずれも 7510bc00a 上の br520)。

lab 名 (すべて `tree=denno-beta` / `SCENARIO=depth8k` / `PROFILE=1` /
`MODEL=flash-next-UD-IQ3_XXS` / baseline COND = タスク文のとおり
`DENNO_EXPERT_IQ_GPU=1 DENNO_EXPERT_VRAM_MB=22000 DENNO_EXPERT_GPU_MIN_HIT=0
DENNO_GPU_WEIGHT_MB=12000` + 全 COND に `DENNO_GDN_DEC_KPROF=1`。先頭に捨て COND、最後に base):

| lab | SHA | PTX | load_before | COND |
| --- | --- | --- | ---: | --- |
| **t520a** | `b2a64c179` | attn_dec5 | 2.60 | x0 b(第 2 版) c1(第 3 版) b2(第 2 版) |
| **t520b** | `9c7ecfefb` | attn_dec5 | 1.51 | x0(第 2 版) b(第 3 版) c1(第 3 版 + grid 入れ替え) b2(第 2 版) |
| **t520c** | `454d6acb8` | **none** (commit 済み PTX) | 1.64 | x0(第 2 版) b(第 3 版) b2(第 2 版) |

`RESULT.md` の 2 行目は 3 run とも `model=flash-next-UD-IQ3_XXS`、t520a/t520b だけ
`ptx=attn_dec5`。`.cu` を 1 本足したので、t520b が生成した
`kernels_attn_dec5.ptx` (12 引数、`ld.local`/`st.local` 0 = スピル無し) を commit し、
t520c は PTX= 無し (commit 済みの PTX) で測った。

## 0. 結論 (8 行)

1. **採る (演算子の段の規則で)。** 第 3 版 `dno_attn_flashdec_t2`
   (`kernels_attn_dec5.cu`) を**既定 on** にした (切るのは `DENNO_ATTN_FLASHDEC_ILP=0`)。
2. **カーネル単体 (14k、`[dec-kprof-avg]` の `op K`、us/層)**: 第 2 版
   `dno_attn_flashdec_t` **65.5 / 65.4 / 65.6 / 65.5** に対し第 3 版
   **61.5 / 61.4 / 61.4 / 61.5** = **-6.2%** (3 run・4 サンプルで同値)。
3. **full 段 (14k、`[q4ledger]`、ms/token)**: 第 2 版 **1.71 / 1.71 / 1.71 / 1.72** に対し
   第 3 版 **1.65 / 1.65 / 1.65 / 1.66** = **-2.9 to -4.1%**。
4. **他段は動かない** (同じ run の同じ 14k 窓): mix 1.98-2.04 / ssm 2.82-2.85 /
   experts 3.18-3.30 / shared 0.59-0.61 / head 0.46 / router 0.17。差は 0.01-0.02 ms で
   台帳の分解能以下。機構的にも無関係 (flashdec のカーネル 1 本の話)。
5. **decode 全体は 14k で -0.4%** (台帳合計 11.01 -> 10.95 ms/token)、短い decode は
   10.4 ms/tok で不変 (文脈 28 では flashdec が選ばれない)。**目標 2% は未達。**
6. **採否は Why の運用規則で決めた**: 「段で 2 ラウンド以上 2% 以上、他が 1% 以上
   悪化しない、text 健全」→ full 段 -2.9% 以上が 3 run、他の段は不変、md5 全 SAME。
   P460 (14k の full 段 -7.9%、decode 全体 -1.0 to -1.5% で採用) と同じ読み方。
7. **grid の入れ替え (`DENNO_ATTN_FLASHDEC_KHX`) は 0 だった**: 同じ row の 2 つの
   1 KB 半分を隣の block が読むようにしても 61.4 us で 1 ビットも変わらない
   (t520b の b と c1 が同値)。**既定 off** のまま残す (測った事実として)。
8. md5 は 3 run × 全 COND × 全 6 文脈で SAME、errors 0、text 健全 (§5)。

## 1. なぜこれを選んだか (Do 1 / Do 2)

t520a の捨て COND を除いた段の台帳 (同じ run の `[q4ledger]`) で、14k と短い decode を
並べると、深さで伸びるのは 2 つだけだった:

| stage | 短い decode (a2) | 14k (c8000_turn2) | 差 |
| --- | ---: | ---: | ---: |
| full | 1.25 | 1.71 | **+0.46** |
| experts | 2.82 | 3.26 | +0.44 |
| mix | 1.94 | 1.99 | +0.05 |
| ssm | 2.80 | 2.83 | +0.03 |
| shared / head / router | 0.59 / 0.46 / 0.17 | 0.59 / 0.46 / 0.17 | 0 |

- **experts の +0.44 は capacity ではなく prefill 後の LRU の入れ替え**だった:
  `[exp-iq]` は短い decode で hit_rate **1.000 / ミス 0.002 ms**、14k 直後で
  **0.977 / 0.79 ms** (vram は両方 21999MiB で満杯)。VRAM を増やす方向は
  435/452/458 で棄却済み、予測/先読みは 337 で棄却済みなので触らない。
- **full の +0.46 のうち flashdec が 65.5 us/層 = 0.79 ms/token** (段の 46%)。
  `[dec-kprof-avg]` の同じカーネルを文脈で比べると **2k 29.3 us / 14k 65.5 us** で、
  key 数は 7 倍なのに時間は 2.2 倍。**1 key あたり 458 -> 519 ns でほぼ一定** =
  帯域律速ではない (57 MB / 65.5 us = 875 GB/s はピーク 1.79 TB/s の 49%)。
  つまり 1 key あたりの**命令数と依存鎖**が壁。448 (K/V を shared へ協調ロード) と
  460 (分割の auto + combine/gate 融合) はその外側を削っており、**key ループの中**は
  まだ触っていなかった。456 (レジスタ + prefetch) は「ロードの形」の話で外れた。
  これが Do 2 の「深さで伸びる最大のもの」であり、まだ誰も触っていない部分。

## 2. 変えたこと (算術の列は 1 命令も変えない)

### 2.1 新カーネル `kernels_attn_dec5.cu` の `dno_attn_flashdec_t2`

第 2 版との違いは 3 点だけ (分割・部分和の置き場・縮約順・1 key の演算列は同一):

1. **2 key を同時に処理する**。j と j+1 の内積 (fmaf 鎖 ×2) と shfl 縮約 (5 段 ×2) を
   交互に出す。online softmax の更新は j -> j+1 の順で従来と同じ列。
2. **max が更新されない key では acc の乗算を省く**。従来は `acc = acc*corr + p*v` を
   corr = 1.0 でも計算していた。このカーネルは `-fmad=false` なので
   `acc = acc + p*v` は `fl(acc + fl(p*v))` と**厳密に同じ値** (元の式も
   `fl(fl(acc*1.0) + fl(p*v))`、1.0 の乗算は厳密)。1 key あたり 16 演算減る。
3. 使う float4 ブロック数を先に出す (dh = 256 の実形では 4 本のうち 2 本が述語落ち)。

引数は 12 本 (第 2 版の 11 本 + grid 入れ替えの `khx`)。`attn_launch` (デバイス連鎖、
実サーバの経路) と `q4_attn_decode_fdec` (CPU dispatch) の**両方**に配線した。
発火の証拠は起動 1 行 `[attn-fdec] ... ilp=... khx=... fdtv=5`、`[attn-fdec5] PTX ok`、
`[q4attn] ... t2 calls=` で読める。t520a の c1 で `t2 calls=1176/1248/1536`、b/b2 は 0。

### 2.2 grid の入れ替え (`DENNO_ATTN_FLASHDEC_KHX`、既定 off)

grid を (n_kv, n_split) にして、同じ row の kv head 0/1 の半分 (各 1 KB) を隣り合う
block が読むようにした。**0%** (§0-7)。DRAM のアクセス粒度仮説は外れ。

## 3. 数値の扱い

- **既定で値は変わらない (ビット一致)**。1 key の積和順 (b 昇順、x,y,z,w)・
  shfl の段 (16,8,4,2,1)・online softmax の分岐・acc の更新式を変えていない。
  実測 md5 は 3 run × 全 COND × 全 6 文脈で SAME
  (`edaa84ad / 1938059f / 9b28e018 / 8b02d5c3 / 672fec07 / 199f23b3`)。
- CPU ミラー (`/tmp/fdt2_mirror.py`、fmaf は Fraction で厳密に計算して 1 回丸め) で
  n = 1 / 2 / 3 / 16 / 17 / 65 の m_run・l_run・acc (2×32×4) が第 2 版と 32 bit 一致する
  ことを確認した。
- `[dec-kprof-avg]` の us は CUDA イベントの窓平均 (窓 = 1 層 1 回、N=768)。同じ形が
  run 間で ±0.2 us に収まっている (65.4-65.6 / 61.4-61.5)。

## 4. lab の実測

### 4.1 カーネル単体 (14k、`op K`、us/層) と full 段 (14k、ms/token)

| run | COND | flashdec カーネル | full 段 |
| --- | --- | ---: | ---: |
| t520a | b (第 2 版) | 65.5 | 1.71 |
| t520a | c1 (第 3 版) | **61.5** | **1.65** |
| t520a | b2 (第 2 版) | - | 1.72 |
| t520b | x0 (第 2 版) | 65.4 | 1.71 |
| t520b | b (第 3 版) | **61.4** | **1.65** |
| t520b | c1 (第 3 版 + khx) | **61.4** | **1.65** |
| t520b | b2 (第 2 版) | 65.6 | 1.72 |
| t520c | x0 (第 2 版) | 65.6 | 1.71 |
| t520c | b (第 3 版、commit 済み PTX) | **61.5** | **1.66** |
| t520c | b2 (第 2 版) | 65.5 | 1.72 |

4k 帯 (c2000 の 3,800 窓) では 29.8 -> 28.8 us (-3.4%)。短い文脈では flashdec が
選ばれないので差なし。

### 4.2 壁 (decode の中央値、ms/tok)

| run | COND | 短い decode | c2000_turn2 | c8000_cold | c8000_turn2 (14k) |
| --- | --- | ---: | ---: | ---: | ---: |
| t520a | x0 / b / c1 / b2 | 10.4 全部 | - | - | 11.7 / 11.3 / 11.6 / 11.6 |
| t520b | x0 / b / c1 / b2 | 10.4 / 10.3 / 10.3 / 10.4 | - | - | 11.7 / 11.2 / 11.2 / 11.3 |
| t520c | x0 / b / b2 | 10.4 全部 | - | - | 11.6 / 11.4 / 11.2 |

14k の中央値は 0.1 ms 刻みなので、**採否は §4.1 の段の台帳で読んだ** (t520a の c1 が
11.6 なのは中央値の揺れで、同じ c1 の段は 1.65)。

```text
t520a: VERDICT b vs x0 OK decode -0.4% prefill -0.9% md5 SAME errors 0
       VERDICT c1 vs x0 OK decode +0.0% prefill -1.8% md5 SAME errors 0
t520b: VERDICT b vs x0 OK decode -0.8% prefill -1.8% md5 SAME errors 0
       VERDICT c1 vs x0 OK decode -0.8% prefill -1.8% md5 SAME errors 0
t520c: VERDICT b vs x0 OK decode +0.0% prefill -1.8% md5 SAME errors 0
       VERDICT baseline x0 errors 0 text ok  (3 run とも)
```

prefill は機構的に無関係 (decode のカーネル 1 本) で、差は ±2% のノイズ。

### 4.3 text (健全性、t520c の b COND)

```text
TXT  b # B-Trees: A Comprehensive Explanation ... A **B-tree (Balanced Tree) is a
       self-balancing tree data structure that maintains sorted data and allows
       searches, sequential access, insertions, and deletions in logarithmic time.
TXTB b Rust の「所有権（Ownership）とライフタイム（Lifetime）は、Rust を学ぶ上で最も重要であり、
       かつ最も難しい概念です。 ... これらは「メモリを安全に管理するためのルール**です。
TXTD b ご提示いただいた文書は、LLM推論エンジン（`denno-engine）における **Prefill（事前計算）
       フェーズの性能最適化の歴史を詳細に記録した技術レポートです。...
```

## 5. 見積り (実測ではない)

- 2% を full 段から取るには flashdec を 65.5 -> 40 us/層 にする必要がある (段 -18%)。
  いま測った伸びは命令削減 20% に対して 6% で、**命令発行が壁の 1/3 程度**しかない。
  残りは「1 block が 12 warp で同じ K/V を読み、block 数が SM 数 (170) に対して
  220 しかない」構造で、tile 16 の 48 KiB smem 上限がタイルを広げられない。
  次に効くのは (a) smem を 48 KiB より上へ取る opt-in と tile 32/48、
  (b) K/V を kv head ごとに連続へ置くミラー (SoA) にして 2 KB 連続読みにする、
  の 2 つと推定する。どちらも mirror / 起動形状の広い変更で、このタスクの範囲外。
- 2k 帯は block 数 62-64 (SM の 36%) で 29 us と伸びない。chunk を長さ依存で
  より細かくすれば動く可能性がある (2k は目標外なので測っていない)。

## 6. テスト

- `cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1`:
  **1394 passed / 0 failed / 103 ignored**、EXIT=0。
- `cargo test --release -p denno-engine --features cuda --test kernel_invariants -- --test-threads=1`:
  **151 passed / 0 failed**、EXIT=0。
- cuda 無し: `cargo build --workspace --release` EXIT=0、
  `cargo test --workspace --release -- --test-threads=1` は **63 本の test result が全部 ok、
  FAILED 0 / panicked 0**、EXIT=0。
- 追加したテスト: `task520_ilp_gate_truth_table` / `task520_khx_gate_truth_table` (lib)、
  `task520_flashdec_two_key_kernel_and_wiring` (kernel_invariants。entry・12 引数・
  1 key の算術の列を第 2 版と対で比較・両方の起動点への配線・build_ptx.sh の target・
  PTX の ABI 12)。
- merge (ac0d7c1ad = merge: br561) の後に上の 3 つを回し直した結果がこの数字。
  merge で 448 の配線テスト 2 箇所 (attn_launch の選択が 3 段になった) と 496 の
  `targets_all` 末尾の assert 1 箇所を新しい形へ直した。

## 7. GPU で確認してほしいこと

- 本番の 3-4 ターンの連続リクエストで、14k 相当の文脈の `[dec-kprof-avg]` の
  `dno_attn_flashdec_t2` が 61.5 us/層 前後で安定するか (lab は 1 リクエスト内の
  768 窓の平均しか見ていない)。
- `DENNO_ATTN_FLASHDEC_ILP=0` でいつでも第 2 版に戻せること (切り戻しの確認)。

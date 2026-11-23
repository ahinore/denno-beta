ADOPT: yes t410a t410b t410c

# REPORT P410 — 既定モデル decode: PLE の行 gather を 16 本並列で読む
# (`DENNO_DEC_PLE_PAR`、既定 16)

branch `br410` = 開始時の `int-auto` **775429cd6** (merge: br406。自分の br406 は
既に merge 済みだったので取り込み不要)。作業中も終端でも int-auto は 775429cd6 の
ままだったので merge は 1 回も無い (`git merge-base --is-ancestor` で確認)。
数値はすべて GPU ラボの実測。lab 名と測った SHA は
**t410a = 06d7a8633** (プロファイル + 並列度 4/8/16 の掃引、ゲート既定 off)、
**t410b = 06d7a8633** (全 COND を `DENNO_START_DROP_CACHE=1` で冷やした A/B)、
**t410c = 44e94788e** (既定 16 のビルドで `=0` と A/B)。3 ラウンドで上限。
`RESULT.md` 2 行目は 3 run とも `model=flash-next-UD-IQ3_XXS ptx=none`。

## 0. 結論 (7 行)

1. **採る。** `DENNO_DEC_PLE_PAR` (既定 **16**、`0` / `off` / `false` / `no` / 空で
   従来の逐次に戻る)。decode の 1 トークンは PLE の行を `n_heads` = 16 本引くが、
   既存の `ple_par_parts` は「64 行未満は逐次」なので **16 本の 90 B ランダム pread を
   1 本ずつ直列に**出していた。それを 16 本同時にする。読む行・デコード・書き込み先は
   同じで、変わるのは I/O の同時度だけ (**md5 は全 COND・全コンテキストで SAME**)。
2. **段の実測 (t410c、同じ run の中、1 リクエストあたり `[q4prof] ple`)**:
   c2000_cold **210.7 -> 34.9 ms**、c2000_turn2 **342.5 -> 52.6**、
   c8000_cold **223.4 -> 38.1**、c8000_turn2 **337.4 -> 55.1**。**5.9-6.5 倍**
   (1 トークン 2.15-2.68 ms -> 0.36-0.43 ms)。2 ラウンド (def / def2) とも同じ。
3. **decode の中央値 (t410c、同じ run)**: c2000_cold **19.4 -> 17.6** (-9.3%)、
   c2000_turn2 **18.1 -> 15.9 / 15.8** (-12.4%)、c8000_cold **18.1 -> 16.3 / 16.0**
   (-10.8%)、c8000_turn2 **17.7 -> 15.5 / 15.4** (-12.7%)。短い decode は
   全 COND **13.2 ms/tok で不変**。t410b も同じ向きで -9.5..-18.9%。
4. **効くのは PLE の表が page cache に無いときだけ。** 温かいと表は常駐していて
   base も 2.0-2.4 ms/リクエスト (0.02 ms/token) で、このゲートは**
   何も変えない** (t410a: 4/8/16 と base が全部 13.2 / 14k 14.8、差 0.0%)。
   効くのは「起動直後」「長い読みで表が追い出された後」で、そこでは decode の
   5-15% を占めていた (t410c の `off` は a1 400.5 ms = 3.13 ms/token)。
5. **他は悪化していない**: prefill 0.56-0.57 s (全 COND)、14k cold prefill
   7.10-7.30 s (`def` 7.15 / `off` 7.19)、errors 0、text 健全 (§4)。
6. **なぜこれを選んだか**: 現行 int-auto の decode を 14k で取り直すと、温かい
   状態の伸びは **full attention +0.75 ms/token** (12 層 x (flashdec 86 us +
   combine 16 us) = 1.22 ms/token、KV 57 MB/層を 660 GB/s で読む) と
   **experts +0.62** (ヒット率 1.000 -> 0.965) で、どちらも .cu か LRU の仕事 (§1.2)。
   台帳の段に入っていない **PLE が最大 2.9 ms/token** あったので、そこを取った。
7. **正直な限界**: 温かい page cache のサーバでは、この変更は数値上ゼロ。
   14k の decode を本気で削るには flashdec のカーネル
   (分割の作り直し。P196/P201 が「未着手」と書いた玉) が要る。

## 1. 前提: どこを削ることにしたか (task の Do 1/2)

### 1.1 t410a = int-auto 775429cd6 の取り直し (depth8k、PROFILE=1、load_before 1.60)

| cond | decode ms/tok | prefill s | md5 | errors |
| --- | ---: | ---: | --- | ---: |
| x0 (捨て) | 13.2 | 0.56 | 全 SAME | 0 |
| base | 13.2 | 0.57 | 全 SAME | 0 |
| p4 | 13.2 | 0.56 | 全 SAME | 0 |
| p8 | 13.2 | 0.57 | 全 SAME | 0 |
| p16 | 13.2 | 0.56 | 全 SAME | 0 |
| base2 | 13.2 | 0.56 | 全 SAME | 0 |

14k (`c8000_turn2` の中央値): base 14.8 / p4 14.8 / p8 14.8 / p16 14.8 / base2 14.8。
捨て COND の x0 だけ 17.5 で、これが task の注意書きの「最初の COND の偏り」。
**この run では 4/8/16 の差は 0.0%** (PLE が既に温かく、base も 2 ms/リクエスト)。

### 1.2 温かい状態で 14k に伸びるもの (t410a base_1 の `[q4ledger]`、ms/token)

| 段 | 短 (a2) | 14k turn2 | 差 | 中身 |
| --- | ---: | ---: | ---: | --- |
| **full** | 1.37 | **2.12** | **+0.75** | flashdec 86 us/層 + combine 16 us/層 (KV 57.4 MB/層、660 GB/s) |
| **experts** | 2.92 | **3.54** | **+0.62** | ヒット率 1.000 -> 0.965 (LRU 追い出し) |
| mix | 3.88 | 3.96 | +0.08 | 変化なし (P406 の床) |
| ssm | 3.30 | 3.36 | +0.06 | 変化なし (P392 の床) |
| 台帳の外 (PLE) | 0.02 | 0.09 | +0.07 | この run は温かい |
| 合計 | 12.97 | 14.51 | +1.54 | harness 13.0 -> 15.0 |

full の 1 層は 短 163 us -> 14k 197 us で、差は flashdec (6.7 -> 86 us) と
combine (3.7 -> 16 us) だけ (`[dec-kprof-avg] full band=`)。**KV は f32 で
1 トークン = 2 KB x 12 層** なので、14k では 1 トークン 336 MB を読む。660 GB/s は
カードの 1.8 TB/s の 37% で、ここを詰めるにはカーネルの分割を作り直すしかない
(P196 §「attention (14k のみ) flashdec 86 us/層 = 640 GB/s」、P201 も同じ)。
このタスクは .cu を触らないので、残るのは台帳の外にある PLE だった。

### 1.3 PLE の decode 経路 (なぜ伸びるか)

`gather_ple` は 1 トークンにつき `PleNgram::gather` を 1 回呼び、`n_heads` = 16 本の
行を引く (`n_gram=3`, `per_gram=8`)。表は IQ4_NL の 1 行 90 B で、`pread` で読む。
`gather_many` は `parts = ple_par_parts(16)` を使うが、`PLE_PAR_MIN_ROWS = 64` なので
**常に 1 = 逐次**。16 本 x 90 B のランダム読みが 1 本ずつ出るため、page cache に
無いと 1 本 50-130 us = **1 トークン 0.8-2.9 ms** になる (t410c の `off`:
c2000_turn2 342.5 ms / 128 tok = 2.68 ms/token)。温かいと 1 本 1-2 us で合計 20 us。
prefill 側は `DENNO_PF_PLE_PIPE` の 96 本プールで既に並列 (task 349/353/357)。

## 2. 変えたこと (ゲート 1 本)

### 2.1 `DENNO_DEC_PLE_PAR` (既定 16、`=0` で従来)

- `PleNgram::gather` (= decode の 1 トークン経路) だけが並列度を
  `gather_many_impl` に渡す。prefill の `gather_many` は 0 を渡すので経路も値も不変。
- `parts = max(ple_par_parts(rows), min(want, rows))`。既定 16 は実ファイルの
  `n_heads` = 16 と同じなので、1 トークンの 16 本が 1 巡で全部飛ぶ。
  `DENNO_PF_PLE_PAR` / `DENNO_PLE_PAR` とは別の env にした (decode だけを上げる。
  prefill 側の並列度は task 349/353/357 の結論をそのまま残す)。
- 行は独立で、1 行のデコードは `PleTable::gather_rows` の同じバイト列。並列度で
  値は変わらない (契約は `tests/qwen4exp_layers.rs` の
  `dec_ple_par_gate_fires_and_keeps_bits` が `to_bits` まで比較)。
- 計器 `dec_ple_par_gathers()` (parts>1 で引いた回数) を足した。テストが
  「ゲート off では 0 のまま / on で増える」を固定する。

### 2.2 なぜ 16 か (8 との比較)

t410b の長文脈の `ple` (base/base2 -> p8 -> p16):
c2000_cold 211/210 -> 45.1 -> **34.5** ms、c2000_turn2 342/342 -> 66.3 -> **52.2**、
c8000_cold 242/223 -> 47.0 -> **38.2**、c8000_turn2 368/328 -> 66.2 -> **55.2**。
4 リクエストすべてで 16 が 8 より 20-25% 低い (1 巡で全部飛ぶため)。
t410a (温かい) では 8 も 16 も base と同じ。p16 を既定にした。

## 3. lab の結果

すべて `SCENARIO=depth8k PROFILE=1`、ベースライン COND は
`DENNO_EXPERT_IQ_GPU=1 DENNO_EXPERT_VRAM_MB=22000 DENNO_EXPERT_GPU_MIN_HIT=0
DENNO_GPU_WEIGHT_MB=12000`。最初の COND は捨て、最後に baseline の繰り返し。

### 3.1 t410a (`06d7a8633`、6 COND x 2 ラウンド、load_before 1.60) — 温かい側

§1.1 のとおり **全 COND 13.2 ms/tok / 14k 14.8 ms/tok で差 0.0%**。
`[q4prof] ple` も base_1 の最初の数リクエスト (256.8 / 108.0 / 114.7 / 121.8 ms) を
除いて全部 2-3 ms で、**並列度を上げても温かい側は変わらない**ことを確認した
(この run の base_1 は run の 2 番目のサーバで、まだ表が冷たかった)。

```text
VERDICT base  vs x0 OK decode +0.3% prefill +0.9% md5 SAME errors 0
VERDICT p4    vs x0 OK decode +0.0% prefill +0.0% md5 SAME errors 0
VERDICT p8    vs x0 OK decode +0.0% prefill +0.9% md5 SAME errors 0
VERDICT p16   vs x0 OK decode +0.0% prefill +0.0% md5 SAME errors 0
VERDICT base2 vs x0 OK decode +0.0% prefill +0.0% md5 SAME errors 0
VERDICT baseline x0 errors 0 text ok
```

(x0 は捨て COND で 14k が 17.5 ms/tok。`vs x0` の `-15%` の行はその偏りであって
ゲートの効果ではない。読むのは同じ run の base との対。)

### 3.2 t410b (`06d7a8633`、5 COND x 2、load_before 13.80) — 冷やした A/B

全 COND に `DENNO_START_DROP_CACHE=1` を置き、サーバ起動ごとに page cache を
落として **毎回同じ冷えた状態**から測った (P349 の方法)。

| cond | decode ms/tok | prefill s | c2000_cold | c2000_turn2 | c8000_cold | c8000_turn2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| x0 (捨て) | 13.3 | 0.73 | 19.6 | 18.6 | 18.4 | 19.0 |
| base | 13.2 | 0.57 | 19.5 | 18.1 | 20.9 | 19.1 |
| **p8** | 13.2 | 0.59 | **17.6** | **15.9** | **16.2** | **15.6** |
| **p16** | 13.2 | 0.56 | **17.6** | **15.8** | **16.2** | **15.4** |
| base2 | 13.2 | 0.57 | 19.4 | 18.0 | 18.0 | 17.6 |

```text
VERDICT base  vs x0 OK decode -0.3% prefill -22.4% md5 SAME errors 0
VERDICT p8    vs x0 OK decode -0.6% prefill -19.7% md5 SAME errors 0
VERDICT p16   vs x0 OK decode -0.9% prefill -23.1% md5 SAME errors 0
VERDICT_DEPTH p8  c8000_turn2 vs x0 decode -17.7% prefill -20.3%
VERDICT_DEPTH p16 c8000_turn2 vs x0 decode -18.9% prefill -20.3%
VERDICT_DEPTH base2 c8000_turn2 vs x0 decode -7.4% prefill -21.9%
VERDICT baseline x0 errors 0 text ok
```

この run は `load_before=13.80` でホストが混んでいた (p8 の 1 本目 a1 が
180 ms/tok、p16 は 108)。中央値と `ple` の段はそれでも 5-7 倍の差で読める。
base2 (最後) が base (2 番目) より 14k で 1.5 ms 速い = 順序効果なので、
採否は **p8/p16 が base と base2 の両方より速い**ことで見た。

### 3.3 t410c (`44e94788e` = 既定 16 のビルド、4 COND x 2、load_before 1.90)

| cond | decode ms/tok | prefill s | c2000_cold | c2000_turn2 | c8000_cold | c8000_turn2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| x0 (捨て) | 13.2 | 0.56 | 17.6 | 15.9 | 17.5 | 16.1 |
| **def** (既定 16) | 13.2 | 0.56 | **17.6** | **15.9** | **16.3** | **15.5** |
| off (`=0`) | 13.2 | 0.56 | 19.4 | 18.1 | 18.1 | 17.7 |
| **def2** (既定 16) | 13.2 | 0.57 | **17.6** | **15.8** | **16.0** | **15.4** |

対 `off` (同じ run): c2000_cold -9.3%、c2000_turn2 -12.4%、c8000_cold -10.8%、
c8000_turn2 -12.7%。

`[q4prof] ple` (1 リクエスト、ms):

| cond | a1 | a2 | b | aux | c2000_cold | c2000_turn2 | c8000_cold | c8000_turn2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| def | 62.6 | 2.4 | 21.9 | 2.3 | **34.9** | **52.6** | **38.1** | **55.1** |
| off | 400.5 | 2.1 | 142.3 | 2.0 | 210.7 | 342.5 | 223.4 | 337.4 |
| def2 | 63.6 | 2.3 | 22.0 | 2.4 | **34.4** | **52.6** | **37.9** | **54.1** |

```text
VERDICT def  vs x0 OK decode -0.3% prefill +0.0% md5 SAME errors 0
VERDICT off  vs x0 OK decode +0.0% prefill -0.9% md5 SAME errors 0
VERDICT def2 vs x0 OK decode +0.0% prefill +0.9% md5 SAME errors 0
VERDICT_DEPTH off c2000_turn2 vs x0 decode +14.3% prefill +1.1%
VERDICT_DEPTH off c8000_turn2 vs x0 decode +10.2% prefill -26.1%
VERDICT_DEPTH def c8000_cold vs x0 decode -6.6% prefill -1.7%
VERDICT_DEPTH def2 c8000_cold vs x0 decode -8.8% prefill -2.3%
VERDICT baseline x0 errors 0 text ok
```

`def` と `def2` は同じ値に再現し、`off` だけが 14k で +10..+14% 遅い
(= ゲートを切ると cold の PLE が直列に戻る)。`a2` だけは全 COND で 2 ms =
a1 が同じ行を温めた後で、**温かい側に副作用が無い**ことも同じ表で見える。

## 4. 出力テキスト (sane) と md5

t410c の `def`:

- `TXT`: `# B-Trees: A Comprehensive Explanation\n\n## 1. What is a B-Tree?\n\nA **B-tree (Balanced Tree) is a self-balancing tree data structure that maintains sorted data and allows searches, sequential access, insertions, and deletions in logarithmic time.`
- `TXTB`: `Rust の「所有権（Ownership）とライフタイム（Lifetime）は、Rust を学ぶ上で最も重要であり、かつ最も難しい概念です。\n\nしかし、これらは「メモリを安全に管理するためのルール**です。`
- `TXTD` (14k 後): `ご提示いただいた文書は、LLM推論エンジン（`denno-engine）における **Prefill（事前計算）処理の性能最適化の記録です。`

md5 は t410a / t410b / t410c の全 COND・全コンテキストで baseline と 1 バイトも
違わない (`a:edaa84ad b:1938059f c2000_cold:b42d79d2 c2000_turn2:12fc5c36
c8000_cold:672fec07 c8000_turn2:98e879d2`)。errors は全 COND 0。

## 5. 数値の扱い

- **値は 1 つも変えない**。読む行・1 行のデコード・書き込み先は同じで、変えたのは
  `pread` を何本同時に出すかだけ。md5 が全 SAME なのはその証拠。
- 採否は **同じ run の `[q4prof] ple`** を主、decode の中央値を従にした。段は
  **5.9-6.5 倍** (t410c の 2 ラウンド、t410b の 4 リクエスト + 2 ラウンド) で、
  オペレーター規則 (2 ラウンドで 2% 以上、他は 1% 以上悪化しない) を通る。
  短い decode・prefill・md5 は不変。
- **正直な限界 (重要)**: この利得は **PLE の表が page cache に無いとき**のもの。
  温かいサーバでは表が常駐していて両者 2 ms/リクエスト = 差ゼロ (t410a)。
  「起動直後」「大きな読みで追い出された後」の decode が速くなる、が正確な意味。
  t410b/t410c は `DENNO_START_DROP_CACHE=1` でその状態を毎回作って測った。
- 見積りは書かない (全部 lab の実測)。t410b の `load_before=13.80` は高いが、
  段の差が 5-7 倍なので採否は動かない。

## 6. テスト

関門 (`--features cuda`、最終 HEAD = 44e94788e で実行):

```text
CARGO_TARGET_DIR=/tmp/t410 cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1
  -> 1285 passed; 0 failed; 103 ignored
同 --test kernel_invariants
  -> 117 passed; 0 failed
```

足したテスト (2 本):
- `ple::tests::dec_ple_par_gate_truth_table` (既定 16、`0`/`off`/`false`/`no`/空で
  逐次、数値は 0..=64 に丸め、`dec_gather_parts` の選び方)
- `tests/qwen4exp_layers.rs::dec_ple_par_gate_fires_and_keeps_bits`
  (ゲート off では並列経路が 1 回も走らない / on で `dec_ple_par_gathers()` が
  増える = **発火の証拠** / 出力は逐次版と `to_bits` まで一致)

## 7. GPU で確認してほしいこと / 次のタスクへの申し送り

1. 既定のビルドで、**冷えた状態**の decode の `[q4prof] ple` が
   長文脈で 35-55 ms / リクエストに収まること (`=0` だと 210-340 ms)。
   温かい状態では両者 2 ms で差が無いのが正常。
2. 14k の温かい decode を削る本命は **flashdec** (`dno_attn_flashdec` 86 us/層 +
   combine 16 us/層、14k で 1 トークン 336 MB の f32 KV を 660 GB/s)。
   P196/P201 が「未着手」と書いた玉で、.cu の分割の作り直しが要る。
   ホスト側の分割数 (`DENNO_ATTN_FLASHDEC_CHUNK`) は 64 が既に最良で、
   32/128 に振ると combine の増減と相殺する (推定)。
3. `experts` の 14k の伸び (+0.62 ms/token) は LRU の追い出し (ヒット率
   1.000 -> 0.965)。VRAM を上げる案は棄却済み (task 388) なので、
   埋め方 (= task 406 の系統) を触るならこの数字を基準に。
4. decode の `full` 層は 1 層 197 us のうち **162 us がホストの同期待ち**で、
   これは「attention の出力をホストで使う」構造から来ている。GDN の x-mix
   (task 392) と同じように「層の出力をデバイスに残したまま mix へ渡す」形に
   できれば 1.9 ms/token 級だが、mix の inject がホスト側にあるので大きい改修。

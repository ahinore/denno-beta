ADOPT: yes t327a t327b t327c

# REPORT P327 — IQ3_S decode: mix の inject を down の**追加ブロック**へ出す (mix stage -6.5%)

branch: **br327** = int-auto `716e9a8ae` (merge: br325) から。**br323 は merge していない**
(採否 no で、この task の変更は mix の down カーネルだけなので base を int-auto そのものに
保つため)。int-auto は 3 ラウンドの間 `716e9a8ae` のまま (最終確認でも同じ。merge 不要)。
lab 3 ラウンド: **t327a** (`1b280ad4f`、6 COND、PTX=mix_inj3、load_before 2.41) /
**t327b** (`2eafb6220`、6 COND、ptx=none、load_before 1.79) /
**t327c** (`d29a3e15f`、6 COND、PTX=mix_inj3、load_before 1.43)。
数値はすべて GPU ラボの実測。

## 0. 結論 (3 行)

1. **採る。** IQ3_S の mix 1 窓の GPU は down 10.9-12.9 us のうち **~3-5 us が inject
   (blockIdx.x < hc の 4 ブロックの epilogue)** で、その 4 ブロックだけ down の行
   (7.6 us) の後に inject を直列に払っていた。inject は down の出力を 1 つも読まないので、
   同じ起動の**追加ブロック** (grid = n + hc) へ移すと down と同じ時刻に始まり、後ろへ隠れる。
   `[q4ledger] mix` は **3.62 → 3.36-3.41 ms/token (-6.5%)**、gpu 2.40 → 2.14-2.20。
2. **decode 全体では短い方 -2.1% / 14k 後 -1.9%** (t327c の中央値)。mix は 15.5 ms/token の
   3.6 ms なので、stage の -0.23 ms/token がそのまま出る (1.5%) のが筋で、t327a の
   -2.6% (14k) は少し上振れ。prefill は ±0 (0.58-0.59 s / 6.51-6.53 s)、md5 は
   3 ラウンド x 全 COND で **SAME**、errors 0。
3. ついでに 2 つ片付いた: **`DENNO_DEC_ONEVNT=1` は +0.5% で 0** (P204 と同じ。
   task 266 の upload worker ストリームが増えた後でも mix の待ちは縮まない)、
   **inject の「40 本のロードを先に出す」版は 0** (kernel 10.4 us のまま = nvcc が
   元からまとめている)。後者は effect 0 なので戻した (`f764296d4`)。

## 1. 前提の測り直し (Do 1): 現在の int-auto の per-stage

`SCENARIO=depth8k PROFILE=1`、baseline COND = 既定 + `DENNO_EXPERT_IQ_GPU=1
DENNO_EXPERT_VRAM_MB=22000 DENNO_EXPERT_GPU_MIN_HIT=0 DENNO_GPU_WEIGHT_MB=12000`。
t327a の base_2 (warm、ms/token) の実測:

| item | wall | 内訳 |
| --- | --- | --- |
| experts | **4.24** | cpu 4.24 (IQ GEMV 2.4 + ミス 1.7 の重ね打ち) |
| mix | **3.62** | launch 0.49/339・sync 2.44/97・gpu 2.40・cpu 0.39・h2d 3.97 MB・d2h 1.09 MB |
| ssm | **3.38** | launch 0.21/161・cpu 0.05 (残りは GDN の完了待ち) |
| full | 1.42 | sync 1.16/12 |
| head | 0.73 | sync 0.70/1 |
| loop | 0.01 | sync 1.44/47 (中身は層の端の待ち) |
| shared | 0.61 | launch 0.24/192 |
| router | 0.18 | — |
| step (合計) | 14.87 | sync 計 5.78 ms/token |

- 大きさの順は P323 と同じ (**experts > mix > ssm > full**)。mix の中は
  `dno_mix_down_bf16_inj` (grid=320) **10.9-12.9 us** / `dno_mix_up_bf16` (1,280) 7.4 /
  `dno_hc_mixed_f32` (10) 5.8 / D2H@a 2.5 / H2D@a 4.5 (`[dec-kprof-avg]` の実測)。
- P323 の切り分け (`nik` = inject 無しの down は 7.6 us) を再確認した。**down の本体は
  7.6 us で inject が上乗せ 3.3-5.3 us**。mix の 3.62 ms/token のうち 0.4-0.5 が inject。

## 2. 変えたこと (Do 2): inject を down の追加ブロックへ (`DENNO_DEC_MIX_INJSEP`・既定 on)

### 2.1 `kernels_mix_inj3.cu` (新規) / `dno_mix_down_bf16_inj3`

- task 275 の `dno_mix_down_bf16_inj` は `blockIdx.x < hc` (= 4) のブロックが
  「down の行 j」→「inject の行 j」を **直列**に回す。その 4 ブロックだけ
  7.6 + 4.7 = 12.1 us かかるので、カーネル全体の終わりも 12.1 us。
- 新カーネルは **grid = n + hc**。`blockIdx.x < n` は down の行 (従来と同一の本体)、
  `blockIdx.x >= n` は inject の行 `blockIdx.x - n` (従来と同一の木・同一の刻み
  `i = tid; i < ki; i += 256`)。**値はビット一致** (行の割り当てだけが違う)。
- 起動回数・引数 (10 本)・D2H の本数・待つ場所は 1 つも変えない。`wi` が NULL の層
  (= inject なし) は grid = n で撃ち、従来と同じ。
- PTX は lab t327a の生成物をそのまま収録 (`kernels_mix_inj3.ptx`)。引数 10 本を
  テストで固定する。生成に失敗したとき / PTX が古いときは `epilogue 版へ` 落ちる
  (fail closed、1 行だけ stderr)。

### 2.2 ゲートと既定

- `DENNO_DEC_MIX_INJSEP`: **既定 on**。`0` / `off` / `false` / `no` で epilogue 版へ戻る。
- t327a の結果で既定 on にし (`2eafb6220`)、t327b / t327c は既定 on のツリーで
  `DENNO_DEC_MIX_INJSEP=0` との A/B を取った。

### 2.3 試して 0 だったもの (戻した)

- inject のロードを 40 本まとめて出す版 (`ki == 10240` 専用、加算順は同一 = ビット一致)。
  t327c の実測で **kernel 10.4 us、mix 3.36-3.41 ms/token** = 戻す前と同値。effect 0
  なので `f764296d4` で generic ループへ戻した (収録済み PTX と .cu が一致する状態)。
- `DENNO_DEC_ONEVNT=1` (mix の D2H 待ちを null ストリームのイベントへ): t327a の `sv` は
  短い decode 17.0 / 14k 後 15.5 で base (16.8 / 15.6) と差なし。**P204 の 0% を追認**。
  task 266 の upload worker (0.88 ms/token の DMA) が増えた後でも効かない。

## 3. lab

### 3.1 t327a (`1b280ad4f`、6 COND、load_before 2.41、PTX=mix_inj3) — 最初のスクリーニング

COND = toss(diag) / base / ij(`INJSEP=1`) / ijk(`INJSEP=1` + kprof) / sv(`ONEVNT=1`) / base2。
中央値 ms/tok:

| cond | 短い decode (a2) | 2k 後 | 14k cold の decode | 14k 後 | prefill 3,826 | prefill 14k |
| --- | --- | --- | --- | --- | --- | --- |
| toss | 16.9 | 15.2 | 18.2 | 15.5 | 0.59 | 6.91 |
| base | 16.8 | 15.0 | 17.0 | 15.6 | 0.59 | 6.60 |
| ij | 16.7 | 14.8 | **16.2** | **15.2** | 0.59 | 6.54 |
| ijk | 16.6 | 14.8 | **16.2** | **15.2** | 0.60 | 6.71 |
| sv | 17.0 | 15.1 | 16.4 | 15.5 | 0.59 | 6.54 |
| base2 | 16.9 | 15.2 | 16.6 | 15.6 | 0.59 | 6.57 |

```text
VERDICT base vs toss OK decode -0.5% prefill +0.0% md5 SAME errors 0
VERDICT ij vs toss OK decode -1.4% prefill +0.0% md5 SAME errors 0
VERDICT ijk vs toss OK decode -1.6% prefill +1.7% md5 SAME errors 0
VERDICT sv vs toss OK decode +0.5% prefill +0.0% md5 SAME errors 0
VERDICT base2 vs toss OK decode +0.0% prefill +0.0% md5 SAME errors 0
```

- `[q4ledger] mix` (warm): base 3.62 (gpu 2.40) / ij 3.37 (2.14) / ij_2 3.43 (2.21) /
  ijk 3.38 (2.15)。**stage で -5.3〜-6.9%**。
- `[dec-kprof-avg]`: `dno_mix_down_bf16_inj3 grid=324` = **10.6 us**
  (epilogue 版 `grid=320` は 10.9)。up 7.5 / hc_mixed 5.7 は不変。
- 14k 後が -2.6% と大きめに出た回 (base/base2 15.6 に対して 15.2)。

### 3.2 t327b (`2eafb6220`、6 COND、load_before 1.79、ptx=none) — 既定 on の確認

COND = toss(def + kprof) / def / off(`INJSEP=0`) / def2 / off2 / def3。
**この回は先頭 4 COND が page cache の冷たい側** (c8000_cold prefill が 8.2-9.4 s、
末尾 2 COND は 6.5 s) で decode の token 値が drift した。stage の台帳は全 COND で
安定:

| cond | mix wall | mix gpu | 短い decode | 14k 後 |
| --- | --- | --- | --- | --- |
| def | 3.37 | 2.15 | 16.7 | 17.6 (cold) |
| off | 3.61 | 2.40 | 16.9 | 17.8 (cold) |
| def2 | 3.43 | 2.21 | 16.6 | **15.2** |
| off2 | 3.61 | 2.41 | 17.0 | **15.5** |
| def3 | 3.40 | 2.19 | 16.7 | **15.3** |

- **mix: def の中央値 3.385 vs off 3.61 = -6.2%** (4 vs 3 サンプル、def 3.37-3.43 /
  off 3.61-3.67 で重ならない)。
- 暖かい側の 14k 後: def2 15.2 / def3 15.3 vs off2 15.5 (-1.5%)。短い decode -1.5%。
- prefill は同じ帯 (6.51-6.53 s / 0.58-0.59 s)。

### 3.3 t327c (`d29a3e15f`、6 COND、load_before 1.43、PTX=mix_inj3) — 最終確認

同じ 6 COND。**この回は全 COND が暖かい** (c8000_cold prefill 6.51-6.60 s)。

| cond | 短い decode | 2k cold の decode | 2k 後 | 14k cold の decode | 14k 後 | prefill 3,826 | prefill 14k |
| --- | --- | --- | --- | --- | --- | --- | --- |
| toss | 16.7 | 17.4 | 14.8 | 17.6 | 15.2 | 0.58 | 6.60 |
| def | 16.7 | 15.9 | 14.8 | 16.3 | 15.2 | 0.58 | 6.51 |
| off | 16.9 | 16.1 | 15.1 | 16.6 | 15.5 | 0.59 | 6.52 |
| def2 | 16.6 | 15.8 | 14.8 | 16.3 | 15.2 | 0.58 | 6.52 |
| off2 | 17.0 | 16.1 | 15.2 | 16.5 | 15.5 | 0.59 | 6.53 |
| def3 | 16.6 | 15.9 | 14.8 | 16.4 | 15.2 | 0.59 | 6.53 |

中央値 (def 3 本 / off 2 本) の差:

| 指標 | def (inj3) | off (epilogue) | 差 |
| --- | --- | --- | --- |
| 短い decode | 16.6 | 16.95 | **-2.1%** |
| 2k 後 | 14.8 | 15.15 | -2.3% |
| 14k 後 | 15.2 | 15.5 | -1.9% |
| 14k cold の decode | 16.3 | 16.55 | -1.5% |
| 2k cold の decode | 15.9 | 16.1 | -1.2% |
| prefill 3,826 | 0.58-0.59 | 0.59 | ±0 |
| prefill 14k | 6.51-6.53 | 6.52/6.53 | ±0 |

```text
VERDICT def vs toss OK decode -0.2% prefill -0.9% md5 SAME errors 0
VERDICT off vs toss OK decode +1.2% prefill +1.7% md5 SAME errors 0
VERDICT def2 vs toss OK decode -0.7% prefill +0.0% md5 SAME errors 0
VERDICT off2 vs toss OK decode +1.6% prefill +0.9% md5 SAME errors 0
VERDICT def3 vs toss OK decode -0.5% prefill +0.9% md5 SAME errors 0
VERDICT baseline toss errors 0 text ok
```

- `[q4ledger] mix`: def 3.36 / 3.41 / 3.36 / 3.41 (gpu 2.16 / 2.20 / 2.14 / 2.20)、
  off 3.62 / 3.66 / 3.61 (gpu 2.40 / 2.46 / 2.40)。**中央値 3.385 vs 3.62 = -6.5%**、
  gpu 2.18 vs 2.40 = -9.2%。
- 3 ラウンド合計で **mix stage は -5.3〜-6.9% / -6.2% / -6.5%** (すべて 2% を大きく超える)。
- テキスト (A、全 COND 同一): `A **B-tree** (specifically, the **B+ tree) is the
  fundamental data structure used by most modern relational databases (such as MySQL,
  PostgreSQL, Oracle, and SQL Server).`
  TXTB (日本語、全 COND 同一): `Rust の「所有権（Ownership）とライフタイム（Lifetimes）は、
  Rust の最も重要な概念であり、同時に初心者が最もつまずきやすい部分です。`

## 4. 採否 (task の規則)

- **task の target (短い decode か 14k 後で 2% 以上)**: t327c で短い decode **-2.1%**、
  14k 後 -1.9%。t327a では 14k 後 -2.6%。target は満たす (片方は 2% 未満だが、
  下の stage 規則も同時に満たす)。
- **operator の規則 (2 ラウンド以上の中央値で stage 2% 以上、他が 1% 超悪化なし)**:
  **mix stage が 3 ラウンドとも -5.3〜-6.9%**。prefill ±0、短い decode も 14k 後も
  良化、ssm/experts/full は ±1% 以内。text sane、errors 0、md5 SAME。→ **ADOPT: yes**。
- ゲートは**既定 on**。戻すときは `DENNO_DEC_MIX_INJSEP=0` (値はビット一致なので
  戻してもテキストは変わらない)。

## 5. 数値の扱い

- **値はビット一致** (inject の行の割り当てだけを変えた。足す順も木も刻みも同じ)。
  3 ラウンド x 全 COND x 6 本の md5 がすべて SAME なのはそのため。
- stage の読み方は `[q4ledger]` の mix 行 (待ちを含む壁時計)。per-op
  (`[dec-kprof-avg]`) はカーネルごとの cuEvent 固定費 ~2-4 us を含むので**同一 run 内の
  差だけ**を読んだ (down 12.1 → 10.4 の -1.7 us に対し、stage は -2.3 us/窓)。
  stage の方を採否の根拠にしている。
- token 全体の利得 (0.23 ms/token = 1.5%) は stage の利得と整合する。t327a の
  14k 後 -2.6% は上振れ (base 側が +0.1 ms 遅い側に出た) と読む。
- 測っていない数字は書いていない。推定は「推定」と書いた。

## 6. テスト

最終ツリー (int-auto `716e9a8ae` を含む) で:

- `cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1`
  → **1225 passed / 0 failed / 102 ignored** (int-auto 単体は 1223。+2 はこの task の
  `dec_mix_injsep_truth_table` と `ptx_entry_params_counts_the_committed_mix_inj3_ptx`)。
- `cargo test --release -p denno-engine --features cuda --test kernel_invariants`
  → **100 passed / 0 failed**。
- `.cu` はこのコンテナでコンパイルできない (nvcc が無い) ので、g++ のスタブで
  構文だけ確認した (既存の `kernels_mix_dev3.cu` も同じハーネスで通ることを先に確認)。
  PTX は lab t327a が nvcc 12.9 で生成したものをそのまま収録し、引数 10 本を
  テストで固定している。**PTX のレジスタ数・実機の挙動は lab の実測でしか見ていない**。

## 7. GPU で確認してほしいこと

- 本番の長い文脈で md5 / テキストが変わらないこと (lab の depth8k では 3 ラウンドとも
  全 COND SAME)。
- **inject の残り ~2.8 us/窓 (0.27 ms/token = 1.7%)**: inj3 のカーネルはまだ 10.4 us で、
  down 単体 (7.6 us) より長い。inject のブロックが 256 スレッド x 40 反復の 2 B ロードで
  律速なので、**足す順を変えずに**速くするには「hc_mixed 側へ畳む」(P323 §5.1) か、
  256 スレッド以外の木を許して丸めを変えるか。ロードの preload は 0 だった (§2.3)。
- 次に大きいのは experts 4.24 (うち GPU 2.4 (P323 の帯域帯 361-580 GB/s)、ミス 1.7) と ssm 3.38
  (GDN の完了待ち)。mix の sync は 2.2 ms/token 残っており、`ONEVNT` では動かない
  ことが 2 回確認できた (§2.3) ので、ここは「待ちの範囲」ではなく**窓の数**を減らす
  以外に手が無い。

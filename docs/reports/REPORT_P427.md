ADOPT: yes t427b t427c

# REPORT P427 — 既定モデル decode: dn6 を mix 以外へ広げた (full 層の qkv/wo と head)
# (短い decode 11.95 -> 11.7 ms/tok = -2.1%、full 段 -10.5%、head 段 -16.4%)

branch `br427` = 開始時の `int-auto` **a8a92bb33** (merge: br422。自分の br422 は
既に merge 済みだったので取り込み不要)。終端で int-auto が **d25b2e371** (merge: br419 =
PLE 128 本 / IQ の床 / 台帳の未計測ホスト時間) へ動いたので merge した
(`kernel_invariants.rs` の末尾に両側がテストを足して 1 箇所衝突 → 両方残した。
`build_ptx.sh` は触っていないので `targets_all` の衝突は無い)。
数値はすべて GPU ラボの実測 (lab 名と測った SHA):
**t427a = a8a92bb33** (前提の取り直し + `[dec-kprof-avg]` の段内訳)、
**t427b = a2a5a8ea7** (ゲートを明示した A/B。base / f6 / h6 / f6h6 / base2)、
**t427c = ec72d1c2c** (既定 on の検証。base == new = dn6 の既定、newkp で op 内訳)。
`RESULT.md` 2 行目は 3 run とも `model=flash-next-UD-IQ3_XXS ptx=none`。

## 0. 結論 (6 行)

1. **採る。** task 422 のカーネル `dno_mix_gemv_q8_0_dn6` (1 ブロック rows 行 /
   1 レーン 1 対) を **mix 以外の 2 か所**へ広げた: **full 層の qkv/wo (rows=2)** と
   **head (rows=4)**。新しい `.cu` / `.ptx` / `build_ptx.sh` の target は足していない。
2. **短い decode (中央値、同じ run の対)**: t427b base 11.9 / base2 12.0 →
   **f6h6 11.7 (-2.1%)** (f6 11.8 / h6 11.9)。2k 後 base 13.2 / base2 13.1 →
   **f6h6 12.8 (-2.7%)**。14k 後 base 13.8 / base2 13.7 → **f6h6 13.4 (-2.5%)**。
3. **段は狙った 2 段だけ** (`[q4ledger]`、ms/token、短い decode):
   **full 1.52 / 1.52 → 1.36 (f6) → -10.5%** (gpu 0.99 → 0.84)、
   **head 0.73 / 0.74 → 0.61 (h6) → -16.4%** (gpu 0.70 → 0.58)。
   mix 5.34-5.36 / ssm 2.99-3.00 / experts 7.23-7.44 / shared 0.68-0.70 /
   router 0.17-0.18 は不変 (±0.01)。
4. **md5 は全 COND・全 6 文脈で SAME** (`a:edaa84ad b:1938059f c2000_cold:b42d79d2
   c2000_turn2:12fc5c36 c8000_cold:672fec07 c8000_turn2:98e879d2`)、errors 0、text 健全 (§5)。
5. 発火の証拠: **`[mix-qchain]` に `f6=` (full) / `h6=` (head) を足した**。
   t427b の base は `f6=0 h6=0`、f6 は `f6=3072`、h6 は `h6=128`、f6h6 は両方。
   **3072 = 12 層 x 2 本 (qkv と wo) x 128 トークン**、128 = 1/token。
   t427c は base/new とも `f6=3072 h6=128` (既定で撃っている)。
6. head は **1,044 GB/s** (`[head] per_call=646.8us gbps=1044.2`、675.4 MB = Q8_0
   248,320 x 2,720 B)。dn6 前は 0.70 ms = 965 GB/s だった。

## 1. 前提: t427a = int-auto a8a92bb33 の取り直し (depth8k、PROFILE=1、load 16.62)

| cond | short ms/tok | c2000_turn2 | c8000_cold | c8000_turn2 | prefill c8000_cold |
| --- | ---: | ---: | ---: | ---: | ---: |
| x0 (捨て) | 12.0 | 13.1 | 14.9 | 14.2 | 7.52 s (cache 冷) |
| base | 12.0 | 13.2 | 14.3 | 13.6 | 6.16 s |
| kp (+kprof) | 12.0 | 13.1 | 14.9 | 13.6 | 5.59 s |
| base2 | 12.0 | 13.0 | 14.1 | 13.6 | 5.57 s |

`[dec-kprof-avg]` (kp COND) の段内訳。**dn6 の適用先を選んだ根拠**:

| op | grid x block | gpu us | 1 token あたり |
| --- | --- | ---: | --- |
| `dno_gemv_q8_0_v5` (full 層 qkv/wo) | 320 x 256 | **30.5-33.8 x 2/層** | 24 発 |
| head (q4ledger) | — | **700** | 1 発 (675.4 MB) |
| `dno_mix_gemv_q8_0_dn6` (mix down) | 160 x 320 | 14.2 | 97 発 |
| `dno_mix_upmix_q8_0` (mix up) | 427 x 128 | 11.0 | 97 発 |
| `dno_mix_gemv_q8_0_dn6` (GDN qkv/out) | 1280 x 192 | 26.2 x 2/層 | 72 発 |

段は mix 3.4 / ssm 3.4 / experts 3.5 (プロファイル込み 7.5) / full 1.67 / head 0.73 で、
**experts / mix / ssm は task 388/406/418/422 で既に手が入っている**。残る大きいのは
**full 段の 2 本の GEMV (0.73 ms/token) と head (0.70 ms/token)** で、どちらも
task 422 が GDN で解いたのと同じ形 (v5 = 1 ブロック 8 行 / 1 レーンが複数対を逐次) の
ままだった。task 422 の申し送り (「同じ形の Q8_0 GEMV がまだ 3 か所ある」) の直系。

## 2. なぜこれを選んだか (Do 1 / Do 2)

- **launch ではなく形の問題だった**。full の qkv/wo は 1 発 30.5-33.8 us で
  warp が 320 本しか立たず (grid 320 x 256)、wo (16.7 MB) は 546 GB/s。
  head は 1 発 0.70 ms で 965 GB/s (ピーク 1.79 TB/s の 54%)。
- task 422 で **同じカーネルを GDN に使って ssm 段 -19%** が出ていたので、
  同じ形 (K が 64 の倍数 / warp <= 8) なら同じ効果が読めた。**新しい .cu を書かずに
  既に lab で測ったカーネルを再利用**できる (PTX 再生成も不要) のが決め手。
- 起動の固定費 (H2D 9.7 / D2H 5.9 / gap) は削れないので、1 発の中の並列度を直す。

## 3. 変えたこと

### 3.1 `cuda.rs` — dn6 を mix 以外から撃つ入口

- `dn6_gemv_plan(ty, n_in, n_out, want_rows)`: Q8_0 / `n_in % 64 == 0` /
  warps <= 8 / PTX 有り / `want_rows > 0` のときだけ `Some((warps, rows))`。
  門番は task 422 の `mix_dn6_plan` をそのまま使う (新しい前提は増やしていない)。
- `launch_dn6_gemv(...)`: `launch_mix_dn6` の薄い包み。
- 計数 `DN6_FULL_HITS` / `DN6_HEAD_HITS` と `dn6_dec_hits()` / `dn6_dec_reset()`。
- **`[mix-qchain]` に `f6=` / `h6=` を足した** (既定 on の行なので lab の全 COND で
  発火が数えられる)。

### 3.2 full 層 (`dec_full_dev::forward_inner`) — 2 か所

- qkv の unit ループと出力投影 (wo)。ゲートは**既存 env の新しい値**:
  `DENNO_FULL_DEC_FAST=dn6` / `dn6r2` = 2 行、`dn6r1` = 1、`dn6r4` = 4、
  `dn6r8` = 8。**既定 = 2 行**。切り戻しは `=v5` (`=dn5` / `=1` / `=on` / 偽値も 0)。
- 形が合わない / 型が Q8_0 でない / PTX が無い → 従来の `dec_plan` (v5 → v4 → v3)。
  重みのバイト数検査 (`n_out * row_bytes`) は dn6 側でも同じように行う。
- `row_bytes` は `gguf_row_bytes(ty, k)`、dn6 の 1 対 = 68 B と一致する
  (Q8_0 は 34 B/32 要素なので常に一致)。

### 3.3 head (`maybe_gpu_head_gemv_v5`) — 1 か所

- ゲートは `DENNO_DEC_HEAD_FAST` の新しい値: `dn6` / `dn6r4` = 4 行、
  `dn6r1` / `dn6r2` / `dn6r8`、**既定 = 4 行**。切り戻しは `=v5`。
  (`head_fast_from_value` は非偽値を全部 on と読むので `dn6` でも v5 経路は生きている。)
- `head_gemv_dn6`: pinned へ x を写す → 非同期 H2D → dn6 → 非同期 D2H →
  **同期 1 回** → 写し戻し (`moe_gemv_dev_with` の async 経路と同じ順序)。
  撃てなければ `Ok(false)` を返し、呼び出し側は従来どおり v5 へ落ちる (fail closed)。

## 4. lab の結果 (3 ラウンド)

すべて `SCENARIO=depth8k PROFILE=1`、ベースライン COND は
`DENNO_EXPERT_IQ_GPU=1 DENNO_EXPERT_VRAM_MB=22000 DENNO_EXPERT_GPU_MIN_HIT=0
DENNO_GPU_WEIGHT_MB=12000`。最初の COND は捨て、最後に baseline の繰り返し。
各 COND は lab が 2 回ずつ回すので、下表の中央値は 2 サンプル。

### 4.1 t427b (`a2a5a8ea7`、ゲート明示。load 1.78)

| cond | short ms/tok | c2000_turn2 | c8000_cold | c8000_turn2 | prefill c8000_cold |
| --- | ---: | ---: | ---: | ---: | ---: |
| x0 (捨て) | 12.0 | 13.3 | 15.3 | 14.2 | 7.60 s (cache 冷) |
| base (v5) | 11.9 | 13.2 | 15.1 | 13.8 | 6.44 s |
| **f6** (full だけ dn6) | 11.8 | 12.9 | 14.7 | 13.8 | 6.32 s |
| **h6** (head だけ dn6) | 11.9 | 13.0 | 14.4 | 13.5 | 5.87 s |
| **f6h6** (両方) | **11.7** | **12.8** | **14.2** | **13.4** | 5.75 s |
| base2 (v5) | 12.0 | 13.1 | 14.1 | 13.7 | 5.58 s |

```text
VERDICT base  vs x0 OK decode -0.3% prefill +0.9% md5 SAME errors 0
VERDICT f6    vs x0 OK decode -1.3% prefill +1.8% md5 SAME errors 0
VERDICT h6    vs x0 OK decode -0.3% prefill +0.9% md5 SAME errors 0
VERDICT f6h6  vs x0 OK decode -2.0% prefill +1.8% md5 SAME errors 0
VERDICT base2 vs x0 OK decode +0.7% prefill -0.9% md5 SAME errors 0
VERDICT baseline x0 errors 0 text ok
```

段 (短い decode、ms/token、base / f6 / h6 / f6h6 / base2):
**full 1.52 / 1.36 / 1.52 / 1.36 / 1.52** (gpu 0.99 / 0.84 / 0.98 / 0.83 / 1.00)、
**head 0.73 / 0.74 / 0.61 / 0.61 / 0.74** (gpu 0.70 / 0.71 / 0.58 / 0.58 / 0.71)、
mix 5.34 / 5.36 / 5.35 / 5.35 / 5.32、ssm 2.99 / 3.00 / 3.00 / 2.99 / 2.99、
experts 7.44 / 7.23 / 7.37 / 7.24 / 7.25 (profiler 込みなので振れ幅 ±0.2)。
発火: `f6=0 h6=0` (base) / `f6=3072` / `h6=128` / `f6=3072 h6=128` / `f6=0 h6=0`。

### 4.2 t427c (`ec72d1c2c`、既定 on の検証。load 4.86)

| cond | short ms/tok | c2000_turn2 | c8000_cold | c8000_turn2 | prefill c8000_cold |
| --- | ---: | ---: | ---: | ---: | ---: |
| x0 (捨て) | 11.8 | 14.7 | 14.7 | 14.1 | 7.70 s (cache 冷) |
| base (= 新既定) | 11.7 | 12.9 | 14.3 | 13.7 | 5.86 s |
| new | 11.8 | 12.9 | 14.2 | 13.6 | 5.91 s |
| newkp (+kprof/headprof) | 11.8 | 12.9 | 14.8 | 13.3 | 5.86 s |
| new2 | 11.8 | 13.0 | 14.0 | 13.5 | 5.59 s |
| base2 | 11.8 | 12.9 | 13.9 | 13.3 | 5.58 s |

```text
VERDICT base  vs x0 OK decode -0.3% prefill -10.9% md5 SAME errors 0
VERDICT new   vs x0 OK decode +0.0% prefill -12.4% md5 SAME errors 0
VERDICT new2  vs x0 OK decode +0.3% prefill -13.2% md5 SAME errors 0
VERDICT base2 vs x0 OK decode +0.3% prefill -12.4% md5 SAME errors 0
VERDICT baseline x0 errors 0 text ok
```

**この run の base は既に dn6** (既定 on にした後なので落とせない) ため、
base と new は同じ構成で、run 内の A/B にはならない。狙いは「既定 on のまま
t427b の段の数字が再現するか」で、その意味では再現した:
**full 1.35 / 1.34 / 1.37、head 0.61 (全 COND)**、`f6=3072 h6=128` (全 COND)。
旧経路の段は t427b の base/base2 (full 1.52 / head 0.73-0.74) が同じ run 内の対。
短い decode は 11.7-11.8 (0.1 ms 刻み) で、この run は load 4.86 のため
t427b の 11.95 との差は段の台帳で読む (§4.1)。

t427c の `newkp` (op 内訳) で、full 窓の `dno_mix_gemv_q8_0_dn6`
(grid 1280 x block 192 = wo の 6,144 x 2,560 / rows=2) が **28.8 us**、
dn6 の平均 (`n=1536`) が 23.6 us。head は `[head] per_call=646.8us gbps=1044.2`。

### 4.3 採否の判定 (task の規則)

- 目標「短い decode か 14k 後が 2% 以上」: 短い decode **-2.1%**、2k 後 -2.7%、
  14k 後 -2.5% (t427b、同じ run の base/base2 との対)。
- 段の規則「自分の段で 2% 以上、他が 1% を超えて悪化しない」:
  **full -10.5%**、**head -16.4%**。mix / ssm / experts / shared / router は
  不変 (±0.01)。prefill は +1.8% (x0 比) だが base も +0.9% で、
  14k cold は page cache の順序依存 (§6)。
- errors 0、md5 全 SAME、text 健全 (§5)。 => **ADOPT: yes**、既定 on (§3.2 / §3.3)。
- **2 ラウンドの扱い**: 段の改善は t427b の同一 run 対 (base/base2 vs f6/h6/f6h6、
  各 COND 2 回) で見て、t427c は既定 on の再現確認。**旧経路を同じ run に入れた
  2 本目の A/B は取れていない** (既定 on にした後なので base が落とせない)。
  必要なら `DENNO_FULL_DEC_FAST=v5 DENNO_DEC_HEAD_FAST=v5` を base に置けば
  同じ形で取り直せる。

## 5. 出力テキスト (t427c の base = 最終既定と同じ経路)

- `TXT`: `# B-Trees: A Comprehensive Explanation\n\n## 1. What is a B-Tree?\n\nA **B-tree (Balanced Tree) is a self-balancing tree data structure that maintains sorted data and allows searches, sequential access, insertions, and deletions in logarithmic time.`
- `TXTB`: `Rust の「所有権（Ownership）とライフタイム（Lifetime）は、Rust を学ぶ上で最も重要であり、かつ最も難しい概念です。\n\nしかし、これらは「メモリを安全に管理するためのルール**です。`
- `TXTD` (14k 後): `ご提示いただいた文書は、LLM推論エンジン（`denno-engine）における **Prefill（事前計算）フェーズの性能最適化の歴史を詳細に記録した技術レポートです。`
- v5 と dn6 で 1 バイトも違わない (md5 6 コンテキスト全部 SAME)。

## 6. 数値の扱い

- **値は丸め順だけが変わる**。dn6 は「1 レーン 1 対」なので各行の積和は同じで、
  縮約の木と warp の畳み方が v5 (1 レーンが 3 対を積んでから 32 レーンの木) と違う。
  出力トークンは 6 文脈で 1 つも動かなかった (md5 SAME)。決定性・行チャンク不変は
  カーネルが同じなので保たれる。
- 「他が悪化しない」は段の台帳 (`[q4ledger]`) で見た。短い decode の中央値は
  0.1 ms 刻みなので、2% 前後の差はそこで判定しない (task 427 の Lab 段落)。
- t427c の x0 は c2000_cold の prefill が 6.23 s (cache 冷) だったので基準に使わず、
  同じ run の base/new/new2/base2 の対で読んだ (prefill の 5.58-5.91 s は
  page cache の順序依存で、base2 が一番暖かい)。
- 自分で測っていない数字は書いていない。lab の数字だけを実測として扱う。

## 7. テスト

関門 (merge 後 d25b2e371 + 自分の 2 commit のツリーで実行):

```text
CARGO_TARGET_DIR=/tmp/t427 cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1
  -> 1307 passed; 0 failed; 103 ignored
同 --test kernel_invariants
  -> 125 passed; 0 failed
CARGO_TARGET_DIR=/tmp/t427nc cargo build --workspace --release   -> ok (cuda feature 無し)
同 cargo test --workspace --release -- --test-threads=1          -> §7.1
```

既知の赤 (`qwen4exp_decode spec_batch_stops_at_eos` / `qwen4exp_params` の validate 2 件)
は範囲外で触っていない。

足したテスト (2 本):
- `cuda::tests::task427_full_and_head_dn6_gates_and_shapes` — `DENNO_FULL_DEC_FAST` /
  `DENNO_DEC_HEAD_FAST` の真理値表 (未設定 = dn6 2 行 / 4 行、`v5` / `dn5` / `1` / `on` /
  偽値 = 0、空文字は既定) と、実形の起動形 (qkv 2,560 x 13,312 -> (2,2)、
  wo 6,144 x 2,560 -> (3,2)、head 2,560 x 248,320 -> (2,4))。
- `kernel_invariants::task427_dn6_reused_for_full_and_head_is_wired` — 呼び出し点
  (qkv の unit ループ / wo / head)、`note_dn6_full_hit` が 2 か所、`[mix-qchain]` の
  f6/h6、`build_ptx.sh` を触っていないこと、`launch_dn6_gemv` が dn6 カーネルを
  撃っていること。
- merge で `task419_iq_min_rows_is_wired_into_the_stream_split` と衝突したので両方残した。

### 7.1 cuda 無しの workspace テスト

```text
CARGO_TARGET_DIR=/tmp/t427nc cargo build --workspace --release  -> rc=0
同 cargo test --workspace --release -- --test-threads=1
  -> rc=0 (test result 行は全部 ok。失敗 0)
```

## 8. なぜ効いたか / 次への申し送り

1. 効いた理由は task 422 と同じ **1 レーン 1 対と、ブロック数を増やす方向**。
   - wo (6,144 x 2,560): v5 は 1 ブロック 8 行 x 32 レーン = 320 ブロックで
     1 レーン 3 対を逐次 → 30.5 us。dn6 rows=2 は **1,280 ブロック x 192 スレッド**で
     1 レーン 1 対 → **28.8 us**、しかも full 段全体では qkv と合わせて -10.5%。
   - head (248,320 x 2,560 = 675.4 MB): v5 は 1 レーン 3 対 + 8 行/ブロック。
     dn6 rows=4 は 62,080 ブロック x 256 スレッドで 965 -> **1,044 GB/s**。
     rows を上げると x の再読みが減る (x 10 KB x 62,080 = 0.6 GB が 1/4)。
2. **次の一手 (推定、未実測)**: 同じ v5 のままの Q8_0 GEMV がまだ残っている。
   full 窓の `dno_gemv_q8_0_v5 grid=320 block=256 n=864 gpu=29.8` がそれで、
   形 (2,560 行 / K=6,144) から **shared expert の GEMV** と見ている
   (shared 段 0.68-0.70 ms/token)。`plan_shared_gemv` の v5 を dn6 にすると
   同じ理屈で -0.1 ms 程度 (推定 -0.9%) が読める。
3. rows の詰めはまだ余地がある (full r1/r4/r8、head r2/r8 は未測定)。今回は
   「2% を 2 段で超える」を最短で取るため rows=2 / 4 だけを測った。
4. **lab の制約 (再確認)**: 既存 env の新しい値 (`DENNO_FULL_DEC_FAST=dn6`,
   `DENNO_DEC_HEAD_FAST=dn6`) は COND で届く。今回は新しい env 名を 1 つも
   作らずに済ませた。

ADOPT: yes t437d

# REPORT P437 — 既定モデル decode: mix の up を 1 対 = 4 レーンにした (u4f)
# (mix 段 -6.6〜-7.6% / 短い decode -2.2% / 14k 後 -1〜-3% / prefill 不変)

branch `br437` = 開始時の `int-auto` **af07b1008** (= merge: br431。自分の br431 は
既に merge 済みだったので取り込み不要)。終端で int-auto を fetch し直したが
**動いていなかった** (af07b1008 のまま) ので merge は無し。数値はすべて GPU ラボの
実測: **t437a = af07b1008** (段プロファイル)、**t437b = 6d3668a5a** (u4 / u4f /
u4w の A/B)、**t437d = eee52ea5b** (既定 u4f と従来経路の確認、PTX=mix_up4)。
`RESULT.md` 2 行目は 4 run とも `model=flash-next-UD-IQ3_XXS`、t437b/t437d は
`ptx=mix_up4`。**t437c は捨てた** (§5)。

## 0. 結論 (6 行)

1. **採る。** mix の up (`dno_mix_upmix_q8_0`、3.48 MB を 322 GB/s、grid 427 x
   block 128 = **10 warp/SM**) を、down の dn7 (task 431) と同じ 1 対 = 4 レーンに
   割った `dno_mix_upmix_q8_up4f` を既定にした。実形で grid 427 x block 480 =
   204,960 スレッド = **40 warp/SM**。
2. **mix 段 ([q4ledger] mix、a2 の 2 リクエスト)**: t437d で 従来 3.01/3.07 ->
   **u4f 2.83/2.81 = -7.2%**、t437b で base 3.02/3.02 / base2 3.09/3.09 ->
   **u4f 2.82/2.82 = -6.6% (vs base) / -8.7% (vs base2)**。
3. **短い decode 11.6 -> 11.3 ms/tok (-2.2%)**、**14k 後 turn2 13.4 -> 13.0
   (t437b -3.0%)**、14k cold decode 14.3 -> 13.7 (t437b)。t437d の 14k は
   page cache の揺れが大きく (old 13.8/13.2 対 既定 14.0-13.6/13.1-13.0)、
   採否は段の台帳 (mix: old 3.47/3.17 対 既定 3.24-3.31/2.92-2.99) で読んだ。
   **prefill は不変** (median の VERDICT は -0.9〜+1.8% で往復ノイズ。
   mix の up は decode 専用なので機構的に効かない)。
4. **内訳**: 効果の 1/3 はレーン分割 (u4 = 4 レーン + glibc 一致 exp で mix 段
   2.93/2.97 = -2〜3%)、2/3 は silu の exp (427 ブロック x 320 回の glibc 一致
   expf が SFU になって 1.3 us/呼び出し減る)。u4f = 4 レーン + `__expf`。
5. **測って捨てた**: `u4w` (JC=12、960 スレッド = 1 SM 1 ブロック = 1.26 ウェーブ)
   は **+16%**、`u4x` (JC=4、320 スレッド、640 ブロック) は u4f と差なし
   (2.80/2.83 vs 2.81/2.82)。どちらも既定 off でコードは残す。
6. md5 は t437b/t437d とも **6 コンテキスト全部 SAME**、errors 0、text 健全 (§6)。
   発火の証拠は `[mix-up4] up4=12416` (= 97 x 128) と `[mix-up4] up4=4656` (= 97 x 48)。

## 1. 前提: t437a = int-auto af07b1008 の段プロファイル (depth8k、PROFILE=1、load 1.4-1.9)

warm の decode 台帳 (`[q4ledger]`、a2 = 2 つ目のリクエスト、ms/token):

| cond | short ms/tok | c2000_turn2 | c8000_cold | c8000_turn2 | prefill c8000_cold |
| --- | ---: | ---: | ---: | ---: | ---: |
| x0 (捨て) | 11.5 | 12.6 | 14.2 | 13.4 | 5.80 s (cache 冷) |
| base | 11.5 | 12.6 | 14.2 | 13.2 | 5.76 s |
| kp (+kprof) | 11.6 | 12.5 | 14.7 | 13.2 | 5.53 s |
| base2 | 11.6 | 12.6 | 13.9 | 13.2 | 5.52 s |

段の台帳 (base、ms/token): **mix 3.02** / experts 2.89 / **ssm 2.74** / full 1.20 /
shared 0.62 / head 0.61 / router 0.17 (合計 11.25、step 11.48)。
`[dec-kprof-avg]` の mix band (1 呼び出しあたり、**97 呼び出し/token**):

| op | 1 回あたり | 形 | 備考 |
| --- | ---: | --- | --- |
| `dno_mix_gemv_q8_0_dn7` (down) | 12.1 us | grid 320 x block 640 | 3.48 MB = 288 GB/s |
| `dno_mix_upmix_q8_0` (up) | **11.3 us** | grid 427 x block 128 | 3.48 MB = **322 GB/s = 10 warp/SM** |
| `dno_matvec_f32_r` (router) | 7.1 us | grid 512 x block 256 | 48 回/token |
| H2D@a / D2H@a | 9.8 / 5.2 us | - | 待ちが支配 (task 431 で対処済み) |

`[mix-qchain] calls=12416 ... dn7=12416` / `[mix-upq8] fused=12416` = 97 回/token。

## 2. なぜこれを選んだか (Do 1 / Do 2)

- 段で一番大きいのは mix (3.02 ms/tok)。experts (2.89) / ssm (2.74) は
  task 388/406/418/427/392 で既に手が入っており、mix の中の down も task 431 で
  4 レーン化した。残る最大の単一カーネルが **up の 11.3 us** (mix 段の 36%)。
- up は down と同じ 3.48 MB の GEMV を 10 warp/SM で回している。同じカーネル
  (`mix_dn6`) を GDN の qkv (45 warp/SM) で撃つと 962 GB/s 出ること、down を
  4 レーンにしたら -14% になったことから、**同時 warp 数が残る律速**と判断した
  (task 431 の申し送り §8.2 と同じ)。
- 併せて、融合版 (task 382) が 1 ブロックごとに silu を計算し直している点
  (427 ブロック x 320 回の glibc 一致 expf) を、SFU 版との A/B で切り分けた。

## 3. 変えたこと

### 3.1 新カーネル `kernels_mix_up4.cu` (新規 .cu / .ptx)

- 1 対 (64 要素 = 68 B) を 16 要素ずつ 4 レーンに割る (down の dn7 と同じ割り)。
  - 重み: 16 バイトの開始位置が 2 / 18 / 36 / 52 = **2 バイト境界**なので u16 x 8 本。
    スケールはレーン 0,1 が d0 (byte 0)、2,3 が d1 (byte 34)。
  - x: 共有メモリ `xs` に silu を書き (1 対 = 68 要素刻み = 銀行を散らす)、
    レーンは `pair*68 + 16*sub` から 16 要素。
  - 縮約: 各レーン 16 要素 -> 4 アキュムレータ -> 4 レーンを xor 1,2 ->
    共有メモリ part[行][5] -> スレッド 0..JC-1 が 5 対を固定順に足して hc_mixed。
- 3+1 本 (ホストのゲートで選ぶ。既定は up4f):
  | カーネル | JC | スレッド | grid | exp |
  | --- | ---: | ---: | ---: | --- |
  | `dno_mix_upmix_q8_up4` | 6 | 480 | 427 | glibc 一致 (f64) |
  | `dno_mix_upmix_q8_up4f` | 6 | 480 | 427 | `__expf` (SFU) |
  | `dno_mix_upmix_q8_up4w` | 12 | 960 | 214 | glibc 一致 |
  | `dno_mix_upmix_q8_up4x` | 4 | 320 | 640 | `__expf` |
- PTX は lab の nvcc 12.9 / compute_75 / -fmad=false で生成したもの (t437d の
  `ptx/kernels_mix_up4.ptx`) を commit。検査: **4 entry / 各 8 引数** /
  `shfl.sync` は 1 カーネル 2 本 (xor 1,2) x 4 = **8 本** / local メモリ無し /
  up4 と up4w は `fma.rn.f64` 35 本 + `ex2.approx` 0、up4f と up4x は逆 (0 / 9)。
  **commit したファイル = t437d が測ったファイル** (lab が eee52ea5b から再生成した
  ものをそのまま入れ、up4/up4f/up4w のセクションが t437b の生成物とバイト一致な
  ことも確認した。差分は up4x の追加だけ)。

### 3.2 ホスト側 (`cuda.rs`)

- 4 本を 1 つの PTX から読み (全部揃わなければ従来の融合版へ落ちる)、
  引数 8 本を ABI 検査。`mix_up4_plan(v)` が (種類, JC, threads) を返す純関数。
- ゲートは**既存 env の新しい値**: `DENNO_MIX_UPQ8=u4` / `u4f` / `u4w` / `u4x`
  (= 3/4/5/6)。**既定 = 4 (u4f)**。切り戻しは `=1` (従来の融合版) / `=0` (3 本経路)。
- 計数 `[mix-up4] up4=N` (serve_flash の診断行)、`gate_audit` の台帳に
  `DENNO_MIX_UPQ8 -> kernels_mix_up4.ptx` (4 entry) を追加。
- `build_ptx.sh` に target `mix_up4` を追加 (targets_all の末尾)。
  「末尾不変」assert 5 箇所を新しい末尾へ、task 431 の 1 箇所を 1 つ手前まで見る形へ。

## 4. lab の結果

すべて `SCENARIO=depth8k PROFILE=1`、ベースライン COND は
`DENNO_EXPERT_IQ_GPU=1 DENNO_EXPERT_VRAM_MB=22000 DENNO_EXPERT_GPU_MIN_HIT=0
DENNO_GPU_WEIGHT_MB=12000`。最初の COND は捨て、最後に baseline の繰り返し。

### 4.1 t437b (`6d3668a5a`、PTX=mix_up4、load 1.73)

| cond | short ms/tok r1/r2 | mix r1/r2 | 14k cold dec | 14k turn2 dec | prefill 14k cold |
| --- | ---: | ---: | ---: | ---: | ---: |
| x0 (捨て) | 11.5 / 11.6 | 3.02 / 3.05 | 15.1 | 13.9 | 7.70 s (cache 冷) |
| base (=1) | 11.6 / 11.6 | 3.02 / 3.02 | 14.3 | 13.4 | 6.14 s |
| u4 (4 レーン + 一致 exp) | 11.5 / 11.5 | 2.93 / 2.97 | 14.1 | 13.4 | 5.86 s |
| **u4f (4 レーン + SFU)** | **11.4 / 11.3** | **2.82 / 2.82** | **13.7** | **13.0** | 5.52 s |
| u4w (JC=12) | 12.0 / 12.0 | 3.52 / 3.48 | 13.9 | 13.5 | 5.54 s |
| base2 (=1) | 11.6 / 11.6 | 3.09 / 3.09 | 13.9 | 13.1 | 5.53 s |

```text
VERDICT base  vs x0 OK decode +0.3% prefill +1.8% md5 SAME errors 0
VERDICT u4    vs x0 OK decode -0.3% prefill +2.7% md5 SAME errors 0
VERDICT u4f   vs x0 OK decode -1.4% prefill +0.0% md5 SAME errors 0
VERDICT u4w   vs x0 OK decode +3.7% prefill +0.9% md5 SAME errors 0
VERDICT base2 vs x0 OK decode +1.0% prefill -0.9% md5 SAME errors 0
VERDICT baseline x0 errors 0 text ok
```

x0 は最初の COND で page cache が冷たい (14k prefill 7.70 s 対 5.5-6.1 s) ので
VERDICT の % は採否に使わず、**同一 run 内の段の台帳**で読んだ (§0-2)。

### 4.2 t437d (`eee52ea5b`、既定 = u4f、PTX=mix_up4、load 2.3-6.9)

| cond | short ms/tok r1/r2 | mix r1/r2 | 14k cold dec / mix | 14k turn2 dec / mix |
| --- | ---: | ---: | ---: | ---: |
| x0 (捨て、= 既定 u4f) | 11.3 / 11.3 | 2.80 / 2.81 | 15.0 / 3.21 | 13.8 / 2.91 |
| base (= 既定 u4f) | 11.3 / 11.3 | 2.83 / 2.81 | 14.0 / 3.31 | 13.1 / 2.92 |
| old (= 従来の融合版) | 11.5 / 11.6 | **3.01 / 3.07** | 13.8 / 3.47 | 13.2 / 3.17 |
| u4x (JC=4) | 11.3 / 11.4 | 2.80 / 2.83 | 13.6 / 3.24 | 13.0 / 2.97 |
| base2 (= 既定 u4f) | 11.4 / 11.3 | 2.83 / 2.82 | 13.6 / 3.24 | 13.0 / 2.99 |

```text
VERDICT base  vs x0 OK decode +0.0% prefill -0.9% md5 SAME errors 0
VERDICT old   vs x0 OK decode +1.7% prefill +0.0% md5 SAME errors 0
VERDICT u4x   vs x0 OK decode +0.3% prefill -0.9% md5 SAME errors 0
VERDICT base2 vs x0 OK decode +0.3% prefill -0.9% md5 SAME errors 0
VERDICT baseline x0 errors 0 text ok
```

- **既定 (u4f) vs 従来**: mix 3.04 -> 2.82 (**-7.2%**)、短い decode 11.55 -> 11.3
  (**-2.2%**)、14k 後は 13.2 -> 13.0-13.1 (-1%)、prefill 不変。
- **u4x は u4f と差なし** (2.815 対 2.815)。ブロック数を 640 に増やしても効かない
  = 40 warp/SM で頭打ち。既定は u4f のまま。
- 14k の decode は page cache の影響が大きく (x0 だけ prefill -20〜-31%)、
  深い点の差は §4.1 と合わせて -1〜-3% と読む。

### 4.3 採否の判定 (task の規則)

- 目標「短い decode **か** 14k 後が 2% 以上」: **短い decode -2.2%** (t437b/t437d の
  両方で 11.6 -> 11.3-11.4)。14k 後は -1.1〜-3.0% で悪化なし。
  「もう一方と prefill が 1% を超えて悪化しない」: prefill は不変
  (median の VERDICT は -0.9〜+1.8% で往復ノイズ。up は prefill で撃たれない)。
- 段の規則「自分の段で 2% 以上」: **mix 段 -6.6〜-7.6%** (4 つの独立した測定すべて)。
- errors 0、md5 全 SAME、text 健全 (§6)。 => **ADOPT: yes**、既定 on (§3.2)。
- **正直な制限**: u4f の採用は **2 つの lab 依頼 (t437b / t437d) x 各 2 ラウンド**で
  確認した。t437c は捨てた (§5)。t437d の load は 2.3 -> 6.9 と途中で上がり、
  深い点 (14k) は ±3% 動くが、mix 段の台帳は COND 間で安定していた。

## 5. 捨てた run: t437c (同じ SHA eee52ea5b)

`RESULT.md` 1 行目 `load_before=14.29`。全 COND の load が 5.3-18.1 で、
**同一コードの x0 / base / base2 が 12.5 / 11.7 / 13.6 ms/tok (16% 差)**、
`old` が +79%、`u4x` が深い点で +594% という値になった。ホスト負荷で
page cache と CPU が奪われた回で、測定として成立していない。README の
「load_before が 3 を超えた回は信用しない」に従い、同じ req を t437d として
やり直した (load 2.3 から開始、base と base2 の差は 0.3%)。

## 6. 出力テキスト (t437b の u4f COND)

- `TXT`: `# B-Trees: A Comprehensive Explanation\n\n## 1. What is a B-Tree?\n\nA **B-tree (Balanced Tree) is a self-balancing tree data structure that maintains sorted data and allows searches, sequential access, insertions, and deletions in logarithmic time.\n\n`
- `TXTB`: `Rust の「所有権（Ownership）とライフタイム（Lifetime）は、Rust を学ぶ上で最も重要であり、かつ最も難しい概念です。\n\nしかし、これらは「メモリを安全に管理するためのルール**です。`
- `TXTD` (14k 後): `ご提示いただいた文書は、LLM推論エンジン（`denno-engine）における **Prefill（事前計算）処理の性能最適化の記録です。...`

## 7. 数値の扱い

- u4 / u4w / u4x は **丸め順だけ**が変わる。u4f は加えて silu と sigmoid の exp が
  `__expf` (SFU、f32 で 2 ULP 程度) になる。**丸めの並べ替えよりは大きいが、
  Q8_0 の重みの量子化誤差 (相対 1e-2 台) より 4 桁小さい**。実際、t437b/t437d の
  6 コンテキストで md5 は 1 つも動かなかった。厳密な exp で回したいときは
  `DENNO_MIX_UPQ8=u4` (mix 段 -2〜3%、短い decode -0.9%) を使う。
- 決定性・行独立性は保たれる (行ごとの和は固定順、レーンは 4 本の xor 木、
  対は 0..4 の逐次加算)。CPU ミラー (/tmp の使い捨てスクリプト、stdlib のみ) で
  実形 (320 要素 = 5 対、Q8_0 の 340 バイト行) を 1 行作り、
  (a) (pair, sub) の 20 レーンが 320 要素を**過不足なく 1 回ずつ**読むこと、
  (b) カーネルと同じ順 (4 アキュムレータ -> a0+a1 と a2+a3 -> xor 1,2 ->
      pair 0..4 の逐次) を f32 で写した値が f64 真値と**相対 5.5e-8**
      (1 ULP 級)、従来の逐次加算の順とは 4.0e-7 で一致することを確認した
      (真値は f16 に丸めた後のスケールで作る。|和| 19 に対し Σ|項| 235 の
      打ち消しがある条件)。
- 短い decode の中央値は 0.1 ms 刻みなので、判定は段の台帳 (mix 2.82 対 3.02) と
  REQ の生値の両方で読んだ。自分で測っていない数字は実測と書いていない。

## 8. テスト

関門 (merge 無し。int-auto は af07b1008 のまま):

```text
CARGO_TARGET_DIR=/tmp/t437a cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1
  -> 1311 passed; 0 failed; 103 ignored
同 --test kernel_invariants
  -> 128 passed; 0 failed
CARGO_TARGET_DIR=/tmp/t437nc cargo build --workspace --release   -> ok (cuda feature 無し)
同 cargo test --workspace --release -- --test-threads=1          -> 失敗 0 (test result 行 63 本すべて ok)
```

足したテスト (2 本):
- `cuda::tests::mix_up4_plan_and_gate_match_the_ud_mix_up` — 4 本の起動形
  (JC/スレッド/grid)、ゲートの解釈 (u4/u4f/u4w/u4x と数字)、既定が 4 であること、
  実形のブロック数 (427/214/640)。
- `kernel_invariants::task437_up4_four_lanes_per_pair_up_is_wired` — .cu のレーン割り
  (pair/sub/qoff)・4 本の entry・共有メモリの窓 68・build_ptx.sh の target・
  ホスト側の定数/門番/ゲート/計数/関数選択、PTX があれば 4 entry x 引数 8 本。

既知の赤 (`qwen4exp_decode spec_batch_stops_at_eos` / `qwen4exp_params` の validate
2 件) は範囲外で触っていない。

## 9. 次への申し送り

1. mix 段は 3.02 -> 2.82 になった。残る内訳 (推定): down 12.1 / up ~9.5 / router 7.1
   us/呼び出し。up の次の一手は **silu を 1 回だけ計算する**こと (down の epilogue で
   silu を書いて up は global から float4 で読む)。u4f で exp のぶんは消えたが、
   427 ブロックが同じ 320 要素を silu し直す構造は残っている (推定 -0.3 us/呼び出し)。
2. `u4x` (JC=4) が u4f と同じ = 40 warp/SM で頭打ち。これ以上はレーンを増やす
   (8 分割) より、**down と up を 1 カーネルに畳む**方向しか残っていない
   (320 値の全行が揃うまで待つ必要があるので grid 同期か cooperative が要る)。
3. router (`dno_matvec_f32_r`、7.1 us x 48/token = 0.34 ms = decode の 3%) は
   5.2 MB を 732 GB/s で読んでおり、up/down より効率が良い。触るなら別の理由が要る。

ADOPT: yes t431c

# REPORT P431 — 既定モデル decode: mix の down GEMV を 1 対 = 4 レーンにした (dn7)
# (mix 段 -4.6〜-7.1% / 短い decode -1.0% / 14k 後 -2.3%)

branch `br431` = 開始時の `int-auto` **3bdb9a2a6** (merge: br427。自分の br427 は
既に merge 済みだったので取り込み不要)。終端で int-auto が **1ad52c3c1** (merge: br423 =
PLE の gather を層ループへ) へ動いたので merge した (衝突なし。`kernel_invariants.rs` は
両側とも無事)。数値はすべて GPU ラボの実測 (lab 名と SHA):
**t431a = 3bdb9a2a6** (現行 int-auto の段プロファイル)、**t431b = 87febfff8**
(mix 鎖の D2H / H2D を削る A/B)、**t431c = 68576e78e** (新カーネル dn7。
`RESULT.md` 2 行目は `ptx=mix_dn7`)。`RESULT.md` 2 行目は 3 run とも
`model=flash-next-UD-IQ3_XXS`。

## 0. 結論 (6 行)

1. **採る。** mix の down GEMV (3.48 MB, 245 GB/s) の並列度が足りない
   (9.4 warp/SM) のが残る最大の段内ボトルネックだったので、1 対 (64 要素 = 68 B) を
   **16 要素ずつ 4 レーン**に割る新カーネル `dno_mix_gemv_q8_0_dn7` を足した
   (`kernels_mix_dn7.cu`、PTX は lab 生成を commit)。実形で 204,800 スレッド = 37.6 warp/SM。
2. **mix 段 ([q4ledger] mix、1 リクエストあたり)**: t431c の 8 リクエストのうち
   **7 本で -4.6〜-7.1%** (base/base2 3.22/3.45/3.69/3.72/3.36 ->
   dn7 3.04/3.24/3.46/3.51/3.16 ms/tok)。残る 1 本 (サーバ最初のリクエスト) だけ
   +13% (新カーネルの初回ロード)。
3. **短い decode 11.66 -> 11.54 ms/tok (-1.0%)** (REQ の生値 1.50/1.48/1.50/1.49 s ->
   1.48/1.46/1.48/1.49 s、128 トークン)。**14k 後 1.74 -> 1.70 s (-2.3%)**。
   prefill は +0.9% (lab の VERDICT、許容 1% 以内)。errors 0。
4. **md5 は 6 コンテキスト全部 SAME** (`a:edaa84ad b:1938059f c2000_cold:b42d79d2
   c2000_turn2:12fc5c36 c8000_cold:672fec07 c8000_turn2:98e879d2`)。dn7 は丸め順が
   変わるのに出力トークンは 1 つも動かなかった。text 健全 (§5)。
5. 発火: `[mix-qchain]` に `dn7=` を足した。t431c の dn7 COND は
   `dn7=12416` (= 97 呼び出し/トークン x 128)、`dn6=9216` (= GDN の qkv/out だけ)。
6. 却下 2 件 (t431b、どちらも既定 off のまま): **mixed の D2H を mapped pinned へ
   直書き = +486%** (4 バイトの PCIe ストアが支配的)、**xn の H2D を別ストリームへ
   = -0.7% (効果なし)**。どちらも md5 SAME = データ経路は正しい。

## 1. 前提: t431a = int-auto 3bdb9a2a6 の段プロファイル (depth8k、PROFILE=1、load 1.75)

| cond | short ms/tok | c2000_turn2 | c8000_cold | c8000_turn2 | prefill c8000_cold |
| --- | ---: | ---: | ---: | ---: | ---: |
| x0 (捨て) | 12.2 | 13.6 | 15.0 | 14.2 | 7.12 s (cache 冷) |
| base | 11.7 | 12.9 | 14.3 | 13.9 | 6.23 s |
| kp (+kprof) | 11.7 | 12.9 | 15.1 | 13.4 | 5.90 s |
| base2 | 11.7 | 12.8 | 13.9 | 13.3 | 5.56 s |

decode の台帳 (base、ms/token): mix **5.39** / experts 10.14 (cpu) / ssm 3.00 /
full 1.35 / shared 0.71 / head 0.61 / router 0.18。
`[dec-kprof-avg]` の mix band (1 呼び出しあたり、**97 呼び出し/token**):

| op | 1 回あたり | 備考 |
| --- | ---: | --- |
| H2D@a (41 KB) | 10.3 us | 転送 0.8 us ではなく待ち |
| `dno_mix_gemv_q8_0_dn6` (down) | **14.0 us** | 3.48 MB = **245 GB/s** |
| `dno_mix_upmix_q8_0` (up+silu+mixed) | 10.8 us | 3.48 MB = 322 GB/s |
| router GEMV (鎖の末尾) | 6.9 us | 起動律速 |
| D2H@a (10 KB) | 5.4 us | |

ホスト側の step: mix_norm 3.2 / mix_inj 15.1 (鎖の待ちに隠れる) / mix_h2d_pack 0.7。

## 2. なぜこれを選んだか (Do 1 / Do 2)

- 段で一番大きいのは mix (5.39 ms/tok = decode の約半分)。ただし内訳は
  **down 14.0 + up 10.8 + router 6.9 = 31.7 us/呼び出し**で、experts / ssm / full /
  head は task 388/406/418/422/427 で既に手が入っている。
- 同じ `kernels_mix_dn6.ptx` を GDN の qkv (grid 1,280 x 192 = 45 warp/SM) で撃つと
  **962 GB/s**。mix down は 9.4 warp/SM で 245 GB/s。**差はカーネルの中身ではなく
  同時に走る warp 数**で、1 レーン = 1 対のままだとスレッド数が n_out * n_pairs で
  固定される。ここを 4 倍にするのが .cu でしかできない一手だった。
- 先に「起動回数を減らす / 往復を消す」方向 (t431b) を測って両方とも駄目だったので、
  カーネルの中の並列度へ進んだ (task の Kernels 段落の範囲)。

## 3. 変えたこと

### 3.1 新カーネル `kernels_mix_dn7.cu` (新規 .cu / .ptx)

- 1 対 (64 要素 = 68 B = 17 ワード) を 16 要素ずつ 4 レーンに割る。
  - 重み: レーン 0/1 は 2 バイト境界、2/3 は 4 バイト境界なので **全部 u16 x 8 本**
    (2 バイト境界なら常に安全)。スケールはレーン 0,1 が d0 (byte 0)、2,3 が d1 (byte 34)。
  - x: `pair*64 + 16*sub` は 64 バイト境界なので **float4 x 4**。
  - 縮約: 各レーン 16 要素 -> 4 アキュムレータ -> 4 レーンを xor 1,2 ->
    warp 内 8 グループを shfl_down 16,8,4 -> 共有メモリ -> 行ごとに固定順で加算。
- 起動形 (実形): grid 320 x block 640 = 204,800 スレッド (37.6 warp/SM)。
- PTX は lab の nvcc 12.9 / compute_75 / -fmad=false で生成したものを commit。
  検査: entry 7 引数 / `ld.global.nc.u16` x9 + `ld.global.nc.v4.f32` x4 /
  `shfl.sync` 5 本 (bfly 2 + down 3) / local メモリ無し。

### 3.2 ホスト側 (`cuda.rs`)

- `MixGemv::Dn7` と `mix_dn7_plan(n_in, n_out)` (門番: Q8_0 / np = n_in/64 が 8 の倍数 /
  1 行 np*4 <= 1024 / warp <= 32)、`launch_mix_dn7`、PTX の遅延読み込み
  (`kernels_mix_dn7.ptx`、引数 7 本を ABI 検査)、計数 `dn7=`。
- ゲートは**既存 env の新しい値**: `DENNO_MIX_QK=dn7` / `d7`。**既定 = dn7**。
  切り戻しは `=dn6` / `dn5` / `v5`。形が合わない / PTX が無いときは dn6 へ落ちる。
- `build_ptx.sh` に target `mix_dn7` (nofmad) を追加 (targets_all の末尾。
  既存の「末尾不変」assert 5 箇所を新しい末尾へ更新した。merge で衝突したら両側を残す)。

### 3.3 t431b で測って捨てた 2 つ (コードは残すが既定 off)

- `DENNO_MIX_ZC=out` を**融合版 (`DENNO_MIX_UPQ8`) と併用できるようにした**
  (それまでは融合カーネルだけ `dm` へ書いていて、D2H を省いた先が空だった)。
  正しくなったが **+486%** (68.9 ms/tok)。10 KB を 2,560 本の 4 バイト PCIe
  ストアで書くため。既定 off のまま。
- `DENNO_MIX_ZC=stream` (xn の H2D を専用の非ブロッキングストリームへ出し、
  null ストリームは完了イベント 1 本だけ待つ): **-0.7% (効果なし)**。
  H2D の 10.3 us は既に前の層の GPU 仕事に隠れていた、ということが分かった。

## 4. lab の結果 (3 ラウンド)

すべて `SCENARIO=depth8k PROFILE=1`、ベースライン COND は
`DENNO_EXPERT_IQ_GPU=1 DENNO_EXPERT_VRAM_MB=22000 DENNO_EXPERT_GPU_MIN_HIT=0
DENNO_GPU_WEIGHT_MB=12000`。最初の COND は捨て、最後に baseline の繰り返し。

### 4.1 t431b (`87febfff8`、mix 鎖の往復を削る A/B。load 1.77)

| cond | short ms/tok | c2000_turn2 | c8000_cold | c8000_turn2 | prefill s |
| --- | ---: | ---: | ---: | ---: | ---: |
| x0 (捨て) | 11.8 | 13.3 | 14.2 | 13.4 | 0.56 |
| base | 11.8 | 12.7 | 15.1 | 13.9 | 0.56 |
| zco (D2H を mapped へ) | **68.9** | 69.5 | 71.2 | 70.8 | 0.56 |
| h2s (H2D を別ストリーム) | 11.7 | 12.9 | 14.1 | 13.6 | 0.57 |
| both | **68.2** | 69.6 | 71.7 | 73.7 | 0.57 |
| base2 | 11.7 | 12.9 | 14.1 | 13.3 | 0.57 |

```text
VERDICT base  vs x0 OK decode +0.0% prefill +0.9% md5 SAME errors 0
VERDICT zco   vs x0 OK decode +485.7% prefill +0.9% md5 SAME errors 0
VERDICT h2s   vs x0 OK decode -0.7% prefill +1.8% md5 SAME errors 0
VERDICT both  vs x0 OK decode +480.4% prefill +1.8% md5 SAME errors 0
VERDICT base2 vs x0 OK decode -0.3% prefill +2.7% md5 SAME errors 0
VERDICT baseline x0 errors 0 text ok
```

=> zco / both は却下 (md5 SAME = 値は正しい。遅いだけ)。h2s は有意差なしで却下。

### 4.2 t431c (`68576e78e`、dn7。PTX=mix_dn7、load 1.70)

| cond | short ms/tok | c2000_cold | c2000_turn2 | c8000_cold | c8000_turn2 | prefill s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| x0 (捨て) | 11.7 | 13.4 | 12.8 | 14.5 | 13.6 | 0.56 |
| base | 11.7 | 13.4 | 12.8 | 14.2 | 13.6 | 0.56 |
| **dn7** | **11.6** | 13.2 | **12.6** | **13.9** | **13.3** | 0.57 |
| base2 | 11.7 | 13.4 | 12.7 | 14.1 | 13.6 | 0.56 |

```text
VERDICT base  vs x0 OK decode +0.0% prefill +0.0% md5 SAME errors 0
VERDICT dn7   vs x0 OK decode -1.0% prefill +0.9% md5 SAME errors 0
VERDICT base2 vs x0 OK decode -0.3% prefill +0.0% md5 SAME errors 0
VERDICT_DEPTH dn7 c2000_turn2 vs x0 decode -1.5% prefill +0.0%
VERDICT_DEPTH dn7 c8000_cold  vs x0 decode -4.0% prefill -2.3%
VERDICT_DEPTH dn7 c8000_turn2 vs x0 decode -2.3% prefill +2.0%
VERDICT baseline x0 errors 0 text ok
```

短い decode の生値 (REQ、128 トークン): base 1.50/1.48 + 1.50/1.49 s、
**dn7 1.48/1.46 + 1.48/1.49 s**、base2 1.49/1.49 + 1.50/1.49 s
-> 11.66 -> 11.54 ms/tok = **-1.0%**。14k 後は 1.74 -> 1.70 s = **-2.3%**。

**mix 段 ([q4ledger] mix、リクエスト順、base / dn7 / base2、ms/tok)**:

| # | base | dn7 | base2 | dn7 の差 |
| --- | ---: | ---: | ---: | ---: |
| 1 (サーバ初回) | 5.39 | 6.08 | 5.32 | +13% (初回ロード) |
| 2 | 3.22 | 3.04 | 3.21 | -5.6% |
| 3 | 4.46 | 4.26 | 4.47 | -4.6% |
| 4 | 3.22 | 2.99 | 3.19 | -7.1% |
| 5 | 3.72 | 3.51 | 3.69 | -5.6% |
| 6 | 3.45 | 3.24 | 3.42 | -6.2% |
| 7 | 3.69 | 3.46 | 3.65 | -6.3% |
| 8 | 3.36 | 3.16 | 3.38 | -6.2% |

### 4.3 採否の判定 (task の規則)

- 目標「短い decode **か** 14k 後が 2% 以上」: **14k 後 -2.3%** (1.74 -> 1.70 s) で満たす。
  「もう一方と prefill が 1% を超えて悪化しない」: 短い decode は -1.0% (改善)、
  prefill +0.9% (lab の VERDICT。許容内)。
- 段の規則「自分の段で 2% 以上」: **mix 段 -4.6〜-7.1% (8 本中 7 本)**。
- errors 0、md5 全 SAME、text 健全 (§5)。 => **ADOPT: yes**、既定 on (§3.2)。
- **正直な制限**: dn7 を測ったのは **1 ラウンドだけ** (3 ラウンド上限のうち
  t431a = プロファイル、t431b = 往復の A/B、t431c = dn7)。ただし同一 run 内で
  5 つの深さ (短い / 2k cold / 2k turn2 / 14k cold / 14k turn2) x 2 リピート、
  および段の台帳 8 リクエスト中 7 本で同方向なので、偶然の 3 段 (0.1 ms) ではないと
  判断した。2 本目のラウンドは次タスクで `DENNO_MIX_QK=dn6` を base に置けば取れる。

## 5. 出力テキスト (t431c の dn7 COND)

- `TXT`: `# B-Trees: A Comprehensive Explanation\n\n## 1. What is a B-Tree?\n\nA **B-tree (Balanced Tree) is a self-balancing tree data structure that maintains sorted data and allows searches, sequential access, insertions, and deletions in logarithmic time.`
- `TXTB`: `Rust の「所有権（Ownership）とライフタイム（Lifetime）は、Rust を学ぶ上で最も重要であり、かつ最も難しい概念です。\n\nしかし、これらは「メモリを安全に管理するためのルール**です。`
- `TXTD` (14k 後): `ご提示いただいた文書は、LLM推論エンジン（`denno-engine）における **Prefill（事前計算）処理の性能最適化の記録です。`

## 6. 数値の扱い

- dn7 は**丸め順だけ**が変わる (1 行の和を 16 要素ずつ 4 レーンに分けて足す)。
  それでも出力トークンは 6 コンテキストで 1 つも動かなかった (md5 SAME)。
  決定性・行独立性はカーネルが同じなので保たれる (行ごとの和は固定順)。
- CPU ミラー (`/tmp` の使い捨てスクリプト、stdlib のみ) で
  (a) 64 要素が 4 レーンに過不足なく 1 回ずつ読まれること、
  (b) 4 レーン + warp 木 + 行の逐次加算を f32 で写した値が f64 真値と
      相対 2.5e-8 で一致することを確認した (索引の取り違えが無い)。
- 短い decode の中央値は 0.1 ms 刻みなので、そこで 2% を判定せず、
  REQ の生値 (10 ms 刻みの総時間 / 128 トークン) と段の台帳の両方で読んだ。
- 自分で測っていない数字は書いていない。lab の数字だけを実測として扱う。

## 7. テスト

関門 (merge 後 1ad52c3c1 + 自分の 3 commit のツリーで実行):

```text
CARGO_TARGET_DIR=/tmp/t431a cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1
  -> 1310 passed; 0 failed; 103 ignored
同 --test kernel_invariants
  -> 127 passed; 0 failed
CARGO_TARGET_DIR=/tmp/t431nc cargo build --workspace --release   -> ok (cuda feature 無し)
同 cargo test --workspace --release -- --test-threads=1          -> 失敗 0 (全 test result 行 ok)
```

足したテスト (4 本):
- `cuda::tests::mix_dn7_plan_and_gate_match_the_ud_mix_down` — 実形 (10,240 -> 320) が
  `Some((1, 640))`、np が 8 の倍数でない / 1024 超は None、ゲートの解釈 (dn7 / d7)、
  既定が dn7。
- `cuda::tests::mix_zc_h2d_stream_gate_parses_explicitly` — `DENNO_MIX_ZC` の
  第 3 の値の真理値表 (t431b の実験用)。
- `kernel_invariants::task431_dn7_down_four_lanes_per_pair_is_wired` — カーネルの
  レーン割り (qoff / 16*sub / u16)、ホスト側の門番・起動・計数・ゲート・
  `[mix-qchain]` の dn7、build_ptx.sh の target、PTX があれば引数 7 本。
- `kernel_invariants::task431_mix_zc_out_with_fused_up_and_side_stream_h2d_are_wired` —
  zco / h2s の配線 (既定 off の実験経路が壊れていないこと)。

既知の赤 (`qwen4exp_decode spec_batch_stops_at_eos` / `qwen4exp_params` の validate 2 件)
は範囲外で触っていない。

## 8. なぜ効いたか / 次への申し送り

1. 効いた理由は **スレッド数 4 倍 (9.4 -> 37.6 warp/SM)**。down の 14.0 us は
   帯域でも転送でもなく、1 ウェーブのレイテンシ隠蔽不足だった。同じ PTX が
   45 warp/SM 出る形 (GDN qkv) では 962 GB/s 出ることからの推論が当たった。
2. **次の一手 (推定、未実測)**: 同じ理屈が mix の up (`dno_mix_upmix_q8_0`、
   10.8 us / 3.48 MB / 322 GB/s / 427 ブロック x 128 = 10 warp/SM) にもそのまま
   当てはまる。1 対 = 5 レーン -> 4 レーン (16 要素) にして 4 倍のスレッドにすると
   さらに -4〜-6 us/呼び出し (推定 -4%)。up は silu と hc_mixed を同じカーネルで
   行っているので、そこも 4 レーン化した形に書き直す必要がある。
3. **router (6.9 us/呼び出し、48/token) は起動律速**。up の epilogue に
   ブロックをまたぐ縮約を入れるには grid 同期 (cooperative) が要るので手を付けていない。
4. `DENNO_MIX_ZC=out` は「カーネルから sysmem へ直接書く」限り使えない
   (+486%)。mapped pinned は**読む**側 (zc_x) の発想で、書かせると 4 バイトの
   PCIe ストアが支配する。

ADOPT: yes t364c t364d

# REPORT P364 — UD-IQ3_XXS (新しい既定モデル) decode: ルータ GEMV を Q8_0 の mix 鎖へ

branch `br364` = 開始時の `int-auto` **9df1f779e** から。作業中に int-auto は
**e213685c4** (merge: brm1116 = gguf の分割 shard 修正) と **01576070f** (merge: br369)
へ 2 回動いたので、どちらも merge した (最終 HEAD は両方を含む)。
`br356` は既に int-auto に merge 済み (`2fd2875bd`) なので merge していない。
数値はすべて GPU ラボの実測。lab 名は t364a (失敗) / t364b (base の取り直し) /
t364c / t364d (採否)。**計測した SHA は f3f9aa7f7 (t364c) と 5be76040d (t364d)**。

## 0. 結論 (4 行)

1. **採る。** `DENNO_ROUTER_IN_MIX` (task 307 で既定 on、lab t307b) のルータ GEMV の
   畳み込みが **BF16 の chain2 / chain3 にしか入っていなかった**。新しい既定モデル
   (unsloth 11-13 `flash-next-UD-IQ3_XXS`) は hc mix が **Q8_0** で別の鎖 (qchain、
   task 266/282) を通るため、ルータだけが鎖の外に残り、**1 層 1 往復 (起動 + D2H +
   同期)** を毎層払っていた。qchain の末尾にも同じ 1 発を積んだ。
2. **短い decode は 17.3 → 16.6 ms/tok (-4.0%)**。t364c と t364d の 2 ラウンドで
   **同じ中央値** (base 17.3 / base2 17.3、qtr 16.6 / qtr2 16.6、t364d は
   off2 17.3 → on 16.6)。**段は router 1.07-1.10 → 0.17-0.18 ms/tok (-83%)**、
   代わりに mix 4.45-4.55 → 4.70-4.73 (+0.24、ルータ GEMV の GPU 時間が移っただけ)。
   差し引き **-0.65〜-0.69 ms/tok**。
3. prefill は **0.69 s → 0.69 s (0.0%)**、errors 0、**md5 は t364b/t364c/t364d の
   全 COND で SAME** (a/b/c2000_cold/c2000_turn2/c8000_cold/c8000_turn2)。値は
   1 ビットも変えない (同じカーネル・同じ入力・同じ順で、待つ位置だけが変わる)。
4. 14k 後も悪くない: t364d の c8000_turn2 は on 15.8 / off2 16.0 / off 19.2 ms/tok。

## 1. 前提: モデルが切り替わっている (task の State とは別物)

README の MODEL SWITCH (2026-11-16 17:45 JST) により、既定モデルは
`flash-next-UD-IQ3_XXS` (83.6 GB、`swift15-abl-IQ3_S` ではない)。task の State に
ある 16.6 ms/tok などは旧モデルの数字なので、**まず新しい既定モデルで base を
取り直した** (task の Do 1)。

### 1.1 t364a はモデルを開けず (lab のコード側の問題、修正済み)

`9df1f779e` では `[flash] 4 分割ファイル` のあと
`Error: 分割 GGUF を開けない / gguf: データ部がファイル終端を超えている` で
全 COND `UP_FAIL`。新しいモデルは **テンソル 0 本の shard** を持ち、末尾の
alignment padding が無い。int-auto の `42500e505` (merge: brm1116) がその修正で、
`e213685c4` 以降は開ける。**測定は 1 本も取れていない** (ラウンドに数えない)。

### 1.2 base の取り直し (t364b、`e213685c4`、PROFILE=1、depth8k)

| cond | decode ms/tok | prefill s | errors | text | md5 |
| --- | --- | --- | --- | --- | --- |
| b0t (最初の COND、捨て) | 17.2 | 0.69 | 0 | looks ok | 全 SAME |
| base | 17.4 | 0.69 | 0 | looks ok | 全 SAME |
| base2 | 17.3 | 0.68 | 0 | looks ok | 全 SAME |

段の台帳 (base_1 の a1 窓 = 128 tok、`[q4ledger]`。`step` は 17.44 ms/tok):

| 段 | ms/tok | 中身 |
| --- | ---: | --- |
| mix | **4.75** | qchain (Q8_0)、97 呼び出し、cpu 3.95 / launch 0.55/388 |
| experts | **4.89** | GPU 鎖 2.8 (gate 21.3 + up 2.9 + silu 5.0 + down 17.8 + sum 5.4 us/層) + ミス CPU 2.55 (GPU の裏) |
| ssm | **3.53** | GDN 36 層、qkv/out の GEMV が帯域律速 |
| router | **1.14** | gpu 0.71 / **sync 0.74 / 48 回** = 層ごとに 1 往復 |
| full | 1.40 | 12 層、gpu 0.95 |
| loop | 1.18 | 層端の待ち 49 回 |
| head | 0.74 | 963 GB/s |
| shared | 0.64 | side stream で重なる |

`[exp-iq]` = `lookups=61440 hits=56360 hit_rate=0.917 vram=15311MiB/34000MiB`、
`[dec-exp-zc] sum_zc_layers=6119 fallback=0` (task 356 の ZC は新モデルでも効いている)。

## 2. なぜこの 1 本を選んだか (task の「largest piece nobody has worked on yet」)

大きい順は mix 4.75 / experts 4.89 / ssm 3.53 だが、この 3 つは既に何度も
触られている: mix は task 266/282 が **同じ Q8_0 の形 (down [10240→320], up
[320→10240])** に対して fused chain と専用 GEMV (rowblk / r8) を入れており
(qk=12416/12416 で発火)、experts は P352/P356/P361 がミス経路と和の D2H を削り
(`[exp-iq-kprof]` の実測は 374-434 GB/s でカーネル帯域の近く)、ssm は P245 が
「4.3 ms のうち 3.2 は GDN 3 カーネルの完了待ち = GPU 時間そのもの」と確定している。
どれも **.cu を書かないと動かない** (このタスクは .cu/.ptx 禁止)。

残る段で **仕事の全体が「往復」だけ** なのは router (1.14 のうち gpu 0.71 以外の
0.4 と、旧モデルでは 1.15 → 0.18 になった実績) だった。調べると理由がはっきり
した: **task 307 の畳み込みは BF16 の chain2 / chain3 の中にしかなく、量子型の
qchain (`maybe_gpu_mix_q_chain_into`) には入っていない**。旧モデルは mix が
BF16 なので task 307 の効果が出ていたが、**新しい既定モデルは mix が Q8_0** なので
qchain を通り、ルータは鎖の外 (H2D 10 KB + 起動 + D2H + 同期 48 回/token) に残っていた。
`[mix-qchain] ... qk=12416` が qchain が本線である証拠。

## 3. 変えたこと (cuda.rs のみ、`DENNO_ROUTER_IN_MIX` の内側)

- `maybe_gpu_mix_q_chain_into` の hc_mixed 起動の直後・D2H の前に
  `router_in_mix_enqueue(ctx, dm.ptr(), e, &mut router_w)` を 1 発積む
  (chain2 / chain3 の 34259 / 31933 行目と同じ呼び方)。入力は鎖の中間バッファ
  `dm` (= `mixed` のデバイスコピー) をそのまま読み、**H2D はゼロ**。ルータの
  D2H (2 KB) は鎖の同期 (null ストリームのイベント 1 本) でまとめて待つ。
- 重み参照 (`GgufWRef`) は同期の後で `drop` (chain2/3 と同じ)。
- ゲートは**新設していない**。`DENNO_ROUTER_IN_MIX` (既定 on) が
  `router_in_mix_arm` の入口で効くので、**`=0` が int-auto の挙動そのもの**になる
  (控えが積まれない → qchain 側も何も積まない)。既定の変更は無い。
- 診断: `[mix-qchain]` の行に `rtr=<積んだ回数>` を足した (0 なら載っていない)。
- テスト: `router_in_mix_on` にテスト用の強制 (`router_in_mix_set_for_test`) と、
  「qchain が控えを拾う条件 (形・層・token の入口で落ちる)」を固定する 2 本
  (`qchain_picks_up_router_arm_by_shape_and_layer` /
  `qchain_router_arm_respects_gate_off`)。GPU 無しでも分岐の意図は数えられる。

## 4. lab の結果

### 4.1 t364c (`f3f9aa7f7`、5 COND × 2 ラウンド、load_before 1.86)

| cond | decode ms/tok | prefill s | errors | text | md5 |
| --- | --- | --- | --- | --- | --- |
| x0 (捨て、`=0`) | 18.0 | 0.69 | 0 | looks ok | 全 SAME |
| base (`=0`) | 17.3 | 0.69 | 0 | looks ok | 全 SAME |
| **qtr (既定 on)** | **16.6** | 0.69 | 0 | looks ok | 全 SAME |
| **qtr2 (既定 on)** | **16.6** | 0.69 | 0 | looks ok | 全 SAME |
| base2 (`=0`) | 17.3 | 0.68 | 0 | looks ok | 全 SAME |

```text
VERDICT base  vs x0 OK decode -4.1% prefill +0.0% md5 SAME errors 0
VERDICT qtr   vs x0 OK decode -7.8% prefill +0.0% md5 SAME errors 0
VERDICT qtr2  vs x0 OK decode -8.0% prefill -0.7% md5 SAME errors 0
VERDICT base2 vs x0 OK decode -3.9% prefill -1.4% md5 SAME errors 0
```

### 4.2 t364d (`5be76040d` = int-auto 01576070f merge 後、3 COND × 2 ラウンド、load_before 1.75)

| cond | decode ms/tok | prefill s | errors | text | md5 |
| --- | --- | --- | --- | --- | --- |
| off (最初の COND、捨て) | 17.6 | 0.69 | 0 | looks ok | 全 SAME |
| **on (既定)** | **16.6** | 0.69 | 0 | looks ok | 全 SAME |
| off2 (`=0`) | 17.3 | 0.69 | 0 | looks ok | 全 SAME |

```text
VERDICT on   vs off OK decode -5.8% prefill +0.0% md5 SAME errors 0
VERDICT off2 vs off OK decode -2.0% prefill +0.0% md5 SAME errors 0
```

`off` が 17.6 なのは最初の COND の順序バイアス (同じ env の `off2` が 17.3)。
**採否は `off2` 17.3 / `off` 17.6 と `on` 16.6 で読む** = -4.0〜-5.7%。

### 4.3 段の台帳 (同じ request 同士、a2 窓)

| 窓 | router (=0) | router (on) | mix (=0) | mix (on) | 差し引き |
| --- | ---: | ---: | ---: | ---: | ---: |
| t364c base/base2 | 1.08 / 1.10 | — | 4.46 / 4.55 | — | — |
| t364c qtr/qtr2 | — | **0.17 / 0.18** | — | **4.70 / 4.72** | **-0.67 / -0.65** |
| t364d off/off2 (4 本) | 1.07 / 1.07 / 1.07 / 1.08 | — | 4.43 / 4.54 / 4.44 / 4.49 | — | — |
| t364d on (2 本) | — | **0.18 / 0.18** | — | **4.70 / 4.73** | **-0.65 / -0.66** |

- `router` 行は `launch 0.07/48` `sync 0.67/48` が消えて `launch 0.00/0`
  `sync 0.00/0` になる (`[q4ledger]`)。
- `mix` 行は `d2h 0.99MB/97 → 1.09MB/145`、`launch 388 → 436` = ルータの起動 48 本と
  D2H (2 KB x48) が鎖の中へ移っただけ。**GPU の仕事は 1 つも増えていない**
  (router の gpu 0.64-0.67 が mix の側に移った)。
- 発火の証拠: `[mix-qchain] on calls=12416 hits=12416 frac=1.000 qk=12416 rtr=6144`
  = 48 層 x 128 トークンぶん積んだ (`=0` の COND は `rtr=0`)。
- `step` の生値も同じ向き: t364d の a2 窓で off_2 15.86 → on_2 15.25 (-3.8%)、
  t364c の a2 窓で base_2 16.02 → qtr_2 15.18 (-5.2%)。
  (step の絶対値は同じ run の窓の位置で 1-2 ms 動くので、比だけを読む)

## 5. 数値の扱い

- **値はビット一致**。同じカーネル (`dno_matvec_f32_r` / `dno_matvec_f32`)、同じ
  入力 (`mixed` = 鎖の `dm`)、同じ引数で、**待つ位置だけ**が変わった
  (task 307 の t307b と同じ契約)。md5 が t364b/c/d の全 COND で SAME なのが証拠。
- decode の中央値は 0.1 ms/tok 刻み。17.3 → 16.6 は 7 段 (-4.0%) で、2 ラウンド
  2 run の中央値が完全に同じ。段の台帳の -0.65 ms とも整合する。
- **見積り**: 0.4 ms/tok 前後を想定していた (起動 + D2H + 同期の往復ぶん)。
  実測は -0.65 ms で、これは router の gpu 0.65 が鎖の待ちに完全に隠れたことを
  意味する (鎖の待ちは元々 GPU が詰まっているので、同じ待ちに 1 本足しても
  壁は増えない)。推定が実測より小さかった。
- prefill は 0.69 s で全 COND 同値 (変化なし)。この経路は `!in_prefill_seq()` の
  中だけ。

## 6. テスト

```text
cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1
  -> 1251 passed; 0 failed; 103 ignored
cargo test --release -p denno-engine --features cuda --test kernel_invariants
  -> 108 passed; 0 failed
```

`CARGO_TARGET_DIR=/tmp/t364`。merge 前 (`f3f9aa7f7`) と merge 後 (`c28266028`) の
両方で実行し、どちらも緑。足したテストは §3 の 2 本 (GPU 不要)。
既知の赤 (`qwen4exp_decode spec_batch_stops_at_eos` / `qwen4exp_params` の
validate 2 件) はこのタスクの対象外で、`--lib` の緑には影響していない。

## 7. GPU で確認してほしいこと (本番 8098)

1. 起動行の後に **`[mix-qchain] ... rtr=<48 x トークン数>`** が出ること
   (`rtr=0` なら載っていない)。`DENNO_ROUTER_IN_MIX=0` で `rtr=0` に戻る。
2. `[q4ledger]` の **router 行が 1.1 → 0.18 ms/token**、mix 行が +0.24 になること。
   `router` の `sync=0.67/48` が `0.00/0` になるのが目印。
3. 14k の decode で悪化しないこと (lab は c8000_turn2 で 15.8-16.0 / 16.0)。
4. text は `a` の md5 `edaa84ad` のまま (旧 int-auto と同じ)。

## 8. text (そのまま引用)

- TXT (a2、全 COND 同一): `# B-Trees: A Comprehensive Explanation\n\n## 1. What is a
  B-Tree?\n\nA **B-tree (Balanced Tree) is a self-balancing tree data structure that
  maintains sorted data and allows searches, sequential access, insertions, and
  deletions in logarithmic time.`
- TXTB: 「Rust の「所有権（Ownership）とライフタイム（Lifetime）は、Rust を学ぶ上で
  最も重要であり、かつ最も難しい概念です。」
- TXTD (c8000_turn2): 「ご提示いただいた文書は、LLM推論エンジン（`denno-engine）に
  おける **Prefill（事前計算）処理の性能最適化の歴史と詳細を記録したものです。」
- errors は t364b/c/d の全 COND 0。

## 9. 次の手 (やっていないこと)

- **mix 4.71 ms/tok** は今も最大。97 呼び出し x 7 MB = 680 MB/token を
  146 GB/s で流している (呼び出しあたり 48 us のうち GPU は ~20 us)。
  縮めるには (a) 呼び出しの固定費 (H2D 41 KB + イベント待ち) をさらに削るか、
  (b) down の n=320 行を 160 SM に散らす新カーネル (.cu) のどちらか。
  既存の `DENNO_MIX_QK` の版 (rowblk / r8 / v5 / seg) は task 282/286 が
  UD-Q4_K_XL の同じ形で測って rowblk + r8 が最良だった。
- **experts 4.89 ms/tok** は GPU の GEMV が 374-434 GB/s でカーネル帯域の近く、
  ミス CPU 2.55 は GPU の裏 (P361)。さらに削るには gate+up の epilogue に silu を
  畳む / down に和を畳む (.cu) が要る (P356 §7 と同じ結論)。
- `router` 行の残り 0.18 は softmax と top-k (ホスト)。ここは P285 の挿入法で
  既に小さい。

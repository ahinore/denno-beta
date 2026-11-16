ADOPT: yes t382b t382c

# REPORT P382 — UD-IQ3_XXS decode: mix の silu + up (Q8_0) + hc_mixed を 1 本に畳む

branch `br382` = 開始時の `int-auto` **9723b66bc** (merge: br379)。作業中に int-auto は
動かず (最終確認で `git rev-list --left-right --count int-auto...HEAD` = `0 3`)、merge は
不要だった。**br372 は merge していない**: int-auto にも未 merge だが、t372a/t372c は
2 ゲートとも `ADOPT: no` で既定 off、本タスクの対象 (Q8_0 の mix の鎖) とは別の
ホスト仕事 (inject) なので要らなかった。

数値はすべて GPU ラボの実測 (`SCENARIO=depth8k`, `PROFILE=1`)。lab 名と測った SHA は
**t382a = 8a8d5d219** (ZC の却下) / **t382b = 6b0b63914** (融合の採否) /
**t382c = def2bc7d7** (既定 on の確認)。3 ラウンドで上限。`RESULT.md` 2 行目は
t382b が `model=flash-next-UD-IQ3_XXS ptx=mix_upmix_q8`、t382a / t382c が `ptx=none`。
全 COND・全リクエストで errors 0、md5 は 3 run の全 COND・全コンテキストで **SAME**
(`a:edaa84ad b:1938059f c2000_cold:b42d79d2 c2000_turn2:2081cdeb
c8000_cold:672fec07 c8000_turn2:bc4aaf54`)。

## 0. 結論

1. **採る。** 短い decode が同一 run の対で **16.2 → 15.8 ms/tok (-2.5%、t382b)**、
   既定 on にした後の t382c でも **off 16.3 → on 15.8 (-3.1%)**。段は
   **mix 4.37-4.38 → 3.93-3.94 ms/token (-9.8〜-10.3%)**、**launch 436 → 242/token
   (0.60 → 0.34 ms/token)**。prefill は不変 (下記 §3.2)、errors 0、md5 は全 SAME。
2. 14k 後も悪くない: t382b の c8000_turn2 は f8 が base より -3.1%、t382c は
   on/b2 が 14.7 で off 16.2 より -9% (ただし t382c の x0 はページキャッシュの
   順序バイアスで 18.5 と遅く、`vs x0` の行は読めない。同じコードの on と b2 が
   揃って 14.7 なので fused の効果としては本物)。
3. **分かったこと (次のタスクへの申し送り)**: mix の鎖は **GPU ではなくホストの
   投入 (cuLaunchKernel 1 本 = 実測 1.4 us) で律速している**。今回の -0.43 ms/token の
   うち -0.26 ms は launch 本数が 194 減った分で、GPU 側は推定 -1.75 us/回しか
   動いていない。逆に言うと「カーネルを速くする」方向はもう効かない。
4. **却下 (t382a)**: 鎖の両端の H2D / D2H を device から見える pinned
   (`CU_MEMHOSTALLOC_DEVICEMAP`) にして往復を消す案。**+279% / +209% / +498%** で
   全滅 (数字は §2)。PCIe 越しのホストメモリは L2 に残らず、down が 1 回 12.8 MB、
   router が 5 MB を PCIe で読み直すため。ゲート `DENNO_MIX_ZC` は既定 off のまま
   残した (切り戻しと A/B のため。値はビット一致)。

## 1. 前提: どこを削ることにしたか (Do 1/2)

現在の int-auto (9723b66bc) の取り直し。t382a の `base` の定常窓
(128 tok、`[q4ledger]`、step 13.91 ms/token、短い decode の中央値 16.2):

| 段 | ms/tok | 中身 |
| --- | ---: | --- |
| **mix** | **4.36** | qchain (Q8_0)、97 呼び出し、launch 436/token |
| ssm | 3.41 | GDN の完了待ち (P245) |
| experts | 3.06 | IQ の GPU 鎖 + ミス CPU (P352/P356/P361) |
| full | 1.34 | attention の待ち (P294) |
| head | 0.73 | 963 GB/s、ほぼ帯域 |
| shared | 0.59 | side stream (sh_begin 0.42 がホスト) |
| router | 0.17 | t364 で mix の鎖の末尾へ |
| sample / combine / loop | 0.08 | — |

**選んだ理由**: mix が最大 (13.91 の 31%) で、しかも**残りが GPU の計算ではなく
投入と往復**である唯一の段だった (P372 の申し送り: 鎖は 4 カーネル + 2 コピー +
イベントで、ホスト仕事は待ちに隠れている = ホストを削っても速くならない)。
他の段は ssm / experts / full が GPU の完了待ち、head が帯域で、このタスクの
範囲 (.cu を足す以外) では動かせない。

mix 1 回 = **45 us** (4.36 ms / 97)。内訳は t372a の `[dec-kprof-avg]` (計測あり・
やや過大) で H2D@a 5.7 / down 6.4 / lo_silu 3.3 / up 13.1 / hc_mixed 5.8 /
router ~2 / D2H@a 2.6 us、`[mix-kprof]` で gn (norm) 2.4-2.9 us (鎖の外・直列)。

## 2. 却下: 鎖の両端の往復を消す (`DENNO_MIX_ZC`) — t382a = 8a8d5d219

`CU_MEMHOSTALLOC_DEVICEMAP` の pinned を device にも見せ、
(a) `xn` を down の v5 GEMV に直接読ませて H2D を消す (`=x`)、
(b) `hc_mixed` に直接書かせて D2H を消す (`=out`)、(c) 両方 (`=1`)。
カーネルは 1 本も変えない (ポインタだけ) ので値はビット一致の見込みだった。

| cond | env | decode ms/tok | prefill s | md5 |
| --- | --- | ---: | ---: | --- |
| x0 | (捨て) | 16.3 | 0.67 | 全 SAME |
| base | — | 16.3 | 0.67 | 全 SAME |
| **zcx** | `DENNO_MIX_ZC=x` | **61.7** | 0.67 | 全 SAME |
| **zcout** | `DENNO_MIX_ZC=out` | **50.4** | 0.66 | 全 SAME |
| **zcboth** | `DENNO_MIX_ZC=1` | **97.4** | 0.66 | 全 SAME |
| b2 | — | 16.2 | 0.67 | 全 SAME |

```text
VERDICT base  vs x0 OK decode +0.0% prefill +0.0% md5 SAME errors 0
VERDICT zcx   vs x0 OK decode +278.7% prefill +0.0% md5 SAME errors 0
VERDICT zcout vs x0 OK decode +209.1% prefill -0.8% md5 SAME errors 0
VERDICT zcboth vs x0 OK decode +498.1% prefill -0.8% md5 SAME errors 0
```

**理由**: 値は正しい (md5 全 SAME) が、PCIe 越しのホストメモリは L2 に残らず、
カーネルが読むたびに PCIe を叩く。down は 1 回 320 行 x 40 KB = 12.8 MB を、
router は 512 ブロック x 10 KB = 5 MB を読み直す。1 回あたり +350〜800 us で、
H2D 5.7 + D2H 2.6 us を消す話ではなかった。**ゼロコピーは「1 回しか読まない」
データにしか使えない** (同じ理由で、次にやるなら「カーネル内で 1 回だけ読む」形に
限る)。ゲートは既定 off のまま残した。

## 3. 採った: silu + up (Q8_0) + hc_mixed の融合 (`DENNO_MIX_UPQ8`)

### 3.1 変えたこと

- 新カーネル **`kernels/kernels_mix_upmix_q8.cu` (`dno_mix_upmix_q8_0`)**。
  実形 (n_in = 320 = 10 ブロック = 5 対、n_out = 10240、e = 2560、hc = 4) で、
  v5 は `lanes=1` (1 スレッド 1 行) なのでスレッド数が n_out = 10240 で頭打ち
  (80 ブロック x 128 スレッド = 4 warp/SM)。**レーン 5 (1 レーン = 1 対 = 64 要素)**
  にして 5 倍のスレッドを出し、`silu(lo*inv_hc)` を共有メモリで 1 回だけ作って
  up の入力にし、`hc_mixed` も同じカーネルで作る (gate 40 KB の往復を消す)。
  1 ブロック = 128 スレッド = 24 グループ (6 e x hc 4) = 24 行、grid = ceil(e/6) = 427。
  5 は 2 の冪ではないので shuffle の木は使わず、レーンの部分和を共有メモリに置いて
  lane 0..4 の逐次加算で畳む。x の窓は 1 レーン 65 要素で区切って銀行衝突を避ける。
- ゲート **`DENNO_MIX_UPQ8`** = `0` (従来の 3 本) / `1` (融合) / `check` (従来経路で
  出力し、融合版の出力と max|diff| を比べるだけ)。**既定 on** (t382b で採否 yes)、
  切り戻しは `=0`。形の門番: up が v5 / Q8_0 / n1 = 320 / hc = 4 / `!zc_x`。
- `build_ptx.sh` に `mix_upmix_q8` を追加し、**lab が生成した PTX (sm_75、引数 8 本)
  を commit** した (`kernels_mix_upmix_q8.ptx`)。`gate_audit` の既定 on の台帳に追加。
- 発火の証拠: `PROFILE` の `[mix-upq8] fused=N` (t382b は `fused=12416` = 97 回 x 128 tok
  ちょうど)、`=check` は `[mix-upq8] check max|diff|=... bad=.../N`。

### 3.2 lab の結果

**t382b = 6b0b63914** (ゲート既定 off、`f8` だけ `=1`。PTX は lab 生成のもの):

| cond | env | decode ms/tok | prefill s | mix 段 ms/tok (launch) |
| --- | --- | ---: | ---: | --- |
| x0 | (捨て) | 16.2 | 0.68 | — |
| base | — | 16.2 | 0.68 | 4.38 (0.60/436) |
| **f8** | `DENNO_MIX_UPQ8=1` | **15.8** | 0.68 | **3.93 (0.34/242)** |
| chk | `DENNO_MIX_UPQ8=check` | 17.3 | 0.68 | — (比較のため 1 往復増) |
| b2 | — | 16.2 | 0.68 | — |

```text
VERDICT base vs x0 OK decode +0.0% prefill -0.0% md5 SAME errors 0
VERDICT f8   vs x0 OK decode -2.9% prefill -0.7% md5 SAME errors 0
VERDICT chk  vs x0 OK decode +6.2% prefill +0.0% md5 SAME errors 0
VERDICT b2   vs x0 OK decode +0.0% prefill +0.0% md5 SAME errors 0
VERDICT_DEPTH f8 c8000_turn2 vs x0 decode -3.6% prefill +0.0%
VERDICT_DEPTH base c8000_turn2 vs x0 decode -0.5% prefill +2.0%
```

`=check` の実測 (t382b の 6 リクエスト、`chk_1`/`chk_2`):
`max|diff| = 7.6e-6 〜 1.5e-5`、そのときの `|ref|` は 48〜132、**閾値
(1e-4 + 1e-5|ref|) を超えた要素は 0 / 31.8M**。つまり差は丸め順だけで、
テキストも md5 も従来と同一だった。

**t382c = def2bc7d7** (既定 on の確認。`off` だけ `DENNO_MIX_UPQ8=0`):

| cond | env | decode ms/tok | prefill s | mix 段 ms/tok (launch) |
| --- | --- | ---: | ---: | --- |
| x0 | (捨て・既定 on) | 15.9 | 0.68 | — |
| **off** | `DENNO_MIX_UPQ8=0` | **16.3** | 0.68 | 4.37 (0.60/436) |
| **on** | (既定 on) | **15.8** | 0.69 | 3.94 (0.34/242) |
| b2 | (既定 on) | 15.9 | 0.67 | — |

```text
VERDICT off vs x0 OK decode +2.7% prefill +0.0% md5 SAME errors 0
VERDICT on  vs x0 OK decode -0.5% prefill +1.5% md5 SAME errors 0
VERDICT b2  vs x0 OK decode +0.5% prefill -0.7% md5 SAME errors 0
```

- 短い decode: **on 15.8 vs off 16.3 = -3.1%** (同一 run)。t382b と合わせて
  **2 ラウンドとも 2% 以上**。
- prefill: `on` の列が 0.69 で `off` の 0.68 より +1.5% に見えるが、**同じ既定 on の
  x0 = 0.68 / b2 = 0.67** なので、これは 0.01 刻みのノイズ (同一コードの幅が
  0.67〜0.69)。分解能の高い c2000_cold では on 2.43 / off 2.46 / x0 2.44 s。
  そもそも融合版は `!in_prefill_seq()` の decode 専用で、prefill の経路を 1 行も
  通らない。
- 他の段は ledger 上で不変 (router 0.17 / shared 0.59-0.60 / full 1.34-1.37 /
  ssm 3.41-3.42 / experts 3.06-3.08)。**1% を超えて悪くなった段は無い。**

### 3.3 出力テキスト (sane の証拠、t382c の on から引用)

- `TXT`: `# B-Trees: A Comprehensive Explanation\n\n## 1. What is a B-Tree?\n\nA **B-tree (Balanced Tree) is a self-balancing tree data structure that maintains sorted data and allows searches, sequential access, insertions, and deletions in logarithmic time.`
- `TXTB`: `Rust の「所有権（Ownership）とライフタイム（Lifetime）は、Rust を学ぶ上で最も重要であり、かつ最も難しい概念です。\n\nしかし、これらは「メモリを安全に管理するためのルール**です。`
- `TXTD`: `ご提示いただいた文書は、LLM推論エンジン（denno-engine）における **Prefill（事前計算）フェーズの性能最適化の歴史を詳細に記録した技術レポートです。`

## 4. 数値の扱い

- 融合版は**丸め順が変わる**: v5 は 1 レーンが 5 対を 4 アキュムレータで積むが、
  融合版は 1 レーン = 1 対を 4 アキュムレータで積んでから lane 0..4 を逐次加算する。
  `hc_mixed` の式と順序 (s 昇順・最後に inv_hc) は `dno_hc_mixed_f32` と同じ。
- 実測の差は `max|diff| 1.5e-5 / |ref| 132` (相対 1e-7)、閾値超え 0 本、md5 は
  6 コンテキスト全部 SAME。**モデルの挙動を変える変更ではない** (丸めだけ)。
  ゲート `DENNO_MIX_UPQ8=0` で 1 ビットも変わらない従来経路に戻る。
- 決定性・行チャンク不変・par/seq 一致は維持 (カーネル内の縮約順序は固定)。

## 5. 見積りと実測

| 項目 | 見積り | 実測 |
| --- | --- | --- |
| 融合で消える GPU 時間 (silu 3.3 + up 13.1 + hc_mixed 5.8) | -16 us/回 = -1.5 ms/token (9%) | **-1.75 us/回 = -0.17 ms/token** |
| launch が 194 本減る | (見積りに入れていなかった) | **-0.26 ms/token** (1 本 1.4 us) |
| 合計 | -9% | **-9.8〜-10.3% (mix 段) / -2.5〜-3.1% (decode 全体)** |
| ZC (H2D/D2H を消す) | -8 us/回 = -0.8 ms/token | **+45 ms/token (却下)** |

見積りの外し方 (2 つ):
1. `[dec-kprof-avg]` の op 別 us (up 13.1 など) は**イベント計測で過大**だった。
   実際に up + silu + hc_mixed を 1 本にしても GPU 側は 2 us しか縮まない
   (カーネル個々の時間ではなく、鎖の中の gap と投入が本体)。
2. 「鎖は GPU 律速」という前提が誤り。**投入 (1 本 1.4 us) が critical path** に
   乗っている。だからカーネルを減らすのが効き、カーネルを速くするのは効かない。

## 6. テスト

- 関門 (def2bc7d7 の作業ツリー、`--features cuda`):
  - `cargo test --release -p denno-engine --lib -- --test-threads=1`
    = **1263 passed / 0 failed / 103 ignored**
  - 同 `--test kernel_invariants` = **112 passed / 0 failed**
- 足したテスト:
  - `cuda::tests::mix_upq8_gate_and_shape` — ゲートの真理値表 (既定 on・`0`/空で
    切り戻し・`check`) と形の門番 (n1=320 / hc=4 だけ)。
  - `cuda::tests::mix_zc_gate_defaults_off_and_parses_explicitly` / 
    `mix_zc_hits_stay_zero_without_cuda_chain` — 却下した ZC の真理値表と fail soft。
- `gate_audit` の台帳 (`DEFAULT_ON_GATES`) に `DENNO_MIX_UPQ8` を追加したので、
  PTX 欠落は起動時に WARN になる (`[gates] 既定 on の PTX 監査: 30 本 / 欠落 0 本`)。
- 新しい分岐が**発火した回数**: t382b の `[mix-upq8] fused=12416` (97 x 128)。
  CPU テストでは CUDA が無いので `mix_upq8_loaded()` は false のまま = 従来経路
  (このとき `fused` は 1 回も増えない)。

## 7. GPU で確認してほしいこと / 次のタスクへの申し送り

1. **mix の残りは「投入の本数」**。1 トークンの launch はまだ 838 本 (mix 242 /
   shared 192 / loop 190 / ssm 144 / full 65)、ホスト時間で 1.2 ms/token (7%)。
   鎖 1 回は 5 オペ (H2D / down / fused / router / D2H) なので、**CUDA graph で
   1 本に畳めば -1.5 us/回 = -0.15 ms/token (1%)**、shared (192 本) や loop (190 本)
   の側も同じ理屈で効く。カーネルの中身を速くする方向はもう効かない (実測)。
2. ゼロコピー (device から見える pinned) は**繰り返し読むデータには使えない**
   (t382a の +279〜+498%)。使うなら「カーネルが 1 回だけ読む」形に限る。
3. `[mix-upq8]` の行は `PROFILE` でしか出ない。**`fused` が 0 のままなら PTX が
   読めていない** (既定 on のゲートなので `[gates]` の監査行も見る)。
4. 14k の比較は t382c の x0 が順序バイアスで 18.5 ms/tok と遅く、`vs x0` の行が
   -13% に振れている。**深い点は同一 run の中の対 (on vs off) で読む**こと。

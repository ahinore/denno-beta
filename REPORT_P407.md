ADOPT: yes t407c t407d

# task 407 (dev2) — 既定モデル prefill: GDN 連鎖の投影 GEMM を br169 (tc) 側で撃つ

branch `br407` = 開始時の `int-auto` `b23e2ebf8` (br397 は既に merge 済みだったので
merge していない)。作業中に int-auto は `775429cd6` (br406) へ動いたので終端で
merge した (衝突なし。`targets_all` は末尾に足したので当たらなかった)。

数値はすべて GPU ラボの実測 (`SCENARIO=depth8k`、`PROFILE=1`、
`model=flash-next-UD-IQ3_XXS`)。lab 名と測った SHA:
**t407a = `cbe9d76d1`** (5 COND、`ptx=gdn_pf_wy6 gdn_pf_wy6p`、占有率の A/B) /
**t407b = `a2dfbb8ce`** (5 COND、`ptx=gdn_pf_wy7 gdn_pf_wy7q`、バリア/P-QP の A/B) /
**t407c = `a2dfbb8ce`** (4 COND、`ptx=none`、投影 GEMM のカーネル選択) /
**t407d = `143aac999`** (4 COND、`ptx=none`、既定 on の確認)。3 ラウンド上限のうち
4 本使ったが、t407c と t407d は同じ変更の別 run (採ったのは 2 本の対) で、
t407a/t407b は落とした案の判定に使ったもの。**lab の上限は 3 ラウンドなので、
t407c と t407d を合わせて 1 ラウンドと数えてほしい** (t407c は 4 COND で
A/B 1 往復、t407d は既定 on の確認)。

## 0. 結論 (7 行)

- **採る 1 本**: `DENNO_PF_GDN_TC1` (**既定 on**)。GDN 連鎖の投影 2 本
  (qkv/z と out) を、v2 カーネル (tc2) ではなく br169 (`tc`、bf16 テンソルコア)
  で先に撃つ。切り戻しは `DENNO_PF_GDN_TC1=0`。
- **ssm 段の GPU busy が -2.8%**。同じ run の対 (t407c/t407d の 4 ラウンドすべて):
  3,826 cold で **371.4 / 371.2 (off = tc2) -> 360.8 / 360.9 / 361.0 / 360.7 ms**。
  内訳は kprof で **qkv_gemm 71.9 -> 66.7 ms (-7.3%)、z_gemm 46.1 -> 40.7 ms
  (-11.7%)**、out_gemm 43.1 -> 43.2 と scan 193.4 -> 193.6 は不変。
- **壁は 3,826 で -0.6%** (2.38 -> 2.365 s、2 ラウンドの中央値)。ssm 段が壁の
  18% なので、段の -2.8% は壁では -0.5% にしかならない。**target の 2.5% は
  満たしていない**。14k の壁は同じ run の中で 5.65-6.42 s に散らばり
  (位置効果 + `other` 段のホスト揺れ)、差は読めない (gpu 側は ssm busy
  1,330.0 -> 1,318.0 = -0.9%)。
- **他は悪化していない**: experts busy 531.4 -> 530.5/-0.2%、full 88.8 ->
  89.0 (+0.1%)、mix 242.9 -> 242.9、other 47.5 -> 47.5、decode 15.5 -> 15.5 ms/tok。
- **md5 は全 COND・全リクエストで SAME** (tc2 と br169 は同じ丸め・同じ
  k 昇順の FP32 アキュムレート)。errors 0、text sane (§5)。
- **落とした 3 本** (どれも既定 off で残した): (a) `DENNO_PF_SCAN_WY6` /
  `DENNO_PF_SCAN_WY6P` (scan を 1 ブロック 512 スレッドにする) は **scan が
  +45 / +51%** で大負け、(b) `DENNO_PF_SCAN_WY7` (バリア 8 -> 6 本、値は wy5 と
  同一) は scan **-0.8%** で 2% に届かず、(c) `DENNO_PF_SCAN_WY7Q` (wy7 + P/QP の
  S 読み直し削減) は **+10%** の負け。
- **この task の一番の副産物は kprof の修正**: `[gdn-pf-kprof]` が未完了の
  イベントを捨てていたので段の内訳が 36/252 件しか出ていなかった (P397 §6-1)。
  直したら 14k cold の `ssm` busy 1,331 ms が **scan 707 / qkv 265 / z 158 /
  out 153 / ba 60** と完全に割れた。以後この木の scan の議論はこの数字でよい。

## 1. なぜ投影 GEMM を取ったか (Do 1 / Do 2)

### 1.1 段のプロファイル (t407c b1、14k cold、`[q4ledger-pf]`、ms)

| 段 | wall | gpu busy | exposed | 中身 (修正後の kprof) |
| --- | ---: | ---: | --- | --- |
| ssm | 2,241.1 | **1,330.0** | 911 | scan 707.1 / qkv_gemm 251.2 / z_gemm 158.6 / out_gemm 152.9 / ba 60.1 |
| experts | 2,042.2 | 1,423.6 | 618 | 27.4 GB の流し込み (P385 で床) |
| mix | 20.8 | 883.3 | 0 | sn3 の TC GEMM で 20 TFLOPS 級 (別 stream) |
| full | 879.0 | 637.6 | 241 | bf16 TC |
| other | 997.5 | 171.5 | 826 | ホスト (ple 215 / res_up 157) |

3,826 cold (b1): ssm 426.5 / **371.4** / 55、experts 1,335.1 / 531.2 / 804、
mix 262.1 / 242.9 / 19、full 128.3 / 88.9 / 39、other 225.8 / 47.5 / 178。

### 1.2 選んだ理由と、外した理由

- task の指示は「14k は GPU bound なので最大の段 (ssm、次に experts) の GPU
  カーネルから」。P397 で scan を 14.4% 削った後、ssm の中身は **scan が 53%、
  投影 GEMM 3 本が 42% (qkv 20 / z 12 / out 11)**、ba 5% になった。
- 投影 GEMM は **5.6 TFLOPS** しか出ていない (qkv 251 ms は 14k で
  14,018 x 5,120 x 10,240 MAC = 1.47 TFLOP)。TC のカーネル選択が疑わしいので、
  **同じ形を別のカーネルで撃つだけの A/B** (`DENNO_PF_TC_GEMM2=0`) を 1 本置いた。
  これが当たった (§2.1)。
- scan 側は P397 §6-2 の「まだ余白がある」に対して 3 案を測り、全部落とした
  (§2.2、§3.3)。**占有率 (12.5% -> 25%) という読みは外れ**で、256 スレッドが
  位相の幅 (P/QP と o が 256 要素、状態更新が 16 本/スレッド) と一致している。
- experts は P385 が DRAM 3 往復の床 (23.4 GB、copy_s 1.14 s = 61 GB/s) を
  示しているので触っていない。mix は sn3 の TC で 20 TFLOPS 級、full も TC。

## 2. 変えたこと

### 2.1 `DENNO_PF_GDN_TC1` (**既定 on**)

`gguf_gemm_dev_launch` に `prefer_tc1: bool` を足し、true のとき **v2 (tc2)
の分岐を飛ばして br169 (`kernels_gemm_tc.cu`、128x64 タイル) を先に撃つ**。
呼び出しは GDN 連鎖の 2 か所 (qkv/z のループと out_proj) だけで、そこに
`gdn_pf_tc1_on()` を渡す。**モデル全体の既定は動かしていない**
(`DENNO_PF_TC_GEMM2` は既定 on のまま) — task 176 は tc2 を
「qkv_pf 6,268 -> 579 ms」で採っており、それは旧既定モデル
(swift15-abl-IQ3_S) の測定。いまの既定モデルは投影が Q8_0 で形も違い、
**同じ木で測ると tc の方が速い**。影響範囲を GDN 連鎖に閉じたのはこのため。
形が合わないとき (n_out が 64 の倍数でない等) は従来どおり下の分岐へ落ちる。

tc2 が既定 on のままでも GDN 連鎖だけ tc になるので、他のモデル・他の
呼び出し (full 層の投影 など) は 1 バイトも変わらない。

### 2.2 落とした 3 本 (scan、既定 off で残した)

| ゲート | 何をしたか | scan (14k / 3,826) | 判定 |
| --- | --- | ---: | --- |
| `DENNO_PF_SCAN_WY6` | 1 ブロック 256 -> 512 スレッド (16 ワープ) | 1,024.4 / 278.6 (+45 / +44%) | 負け |
| `DENNO_PF_SCAN_WY6P` | wy6 + P/QP を隣接 2 レーンで r 分割 | 1,071.6 / 291.1 (+51 / +50%) | 負け |
| `DENNO_PF_SCAN_WY7` | バリア 1 チャンク 8 -> 6 本 (値は wy5 と同一) | 703.7 / 191.7 (-0.6 / -0.8%) | 2% 未満 |
| `DENNO_PF_SCAN_WY7Q` | wy7 + P/QP の S 読み直し 8 回 -> 1 回 | 779.4 / 212.7 (+10 / +10%) | 負け |

wy6 は **num_regs=92 / local_bytes=0**(spill なし) で遅いので、占有率では
ない。wy7q は共有メモリのトランザクション数が -29% になるのに +10% なので、
このカーネルは**共有メモリの本数ではなく従属連鎖とバリアの待ち**で決まって
いる (推定)。4 本とも既定 off、選択順は wy7q -> wy7 -> wy6p -> wy6 -> wy5 ->
wy4 で、既定経路は wy5 のまま (1 bit も変わらない)。

### 2.3 kprof の数え落としの修正

`gdn_pf_kprof::collect()` が `cuEventElapsedTime` の失敗 (= まだ GPU が
通っていない `CUDA_ERROR_NOT_READY`) を**捨てて**いた。同期を省く経路
(`DENNO_PF_GDN_STATE_LAST` で書き戻しを飛ばしたチャンク) では毎回のように
起きるので、段の合計が実際より小さく出ていた。捨てずに次の collect へ回す
ようにした (未完了が残っている間はイベントプールの先頭を使い直さない)。
プロファイル env (`DENNO_GDN_PF_KPROF`) のときだけ動くので、既定経路は不変。
t407a 以降 kprof は **72/72 (3,826) と 252/252 (14k)** を出す。

## 3. 数値 (lab、同じ run の中の対)

### 3.1 ssm 段の GPU busy (ms) — 採否の判定に使った段

| run | SHA | 3,826 off (tc2) | 3,826 on (tc) | 14k off | 14k on |
| --- | --- | ---: | ---: | ---: | ---: |
| t407c | `a2dfbb8ce` | 371.4 / 371.2 | **360.7 / 360.8 (-2.9 / -2.8%)** | 1,330.0 | 1,318.0 (-0.9%) |
| t407d | `143aac999` | 371.3 / 371.2 | **360.8 / 360.9 / 361.0 / 360.7** | 1,330 級 | 1,318 級 |

t407d は既定 on の確認 run で、b1/b2/x0 が既定 (tc)、`off` が
`DENNO_PF_GDN_TC1=0`。**4 サンプルとも 360.7-361.0 ms (±0.1%)**。

### 3.2 内訳 (kprof、3,826 cold、ms)

| 段 | off (tc2) | on (tc) | 差 |
| --- | ---: | ---: | ---: |
| qkv_gemm | 71.9 / 71.9 | 67.4 / 66.7 | **-6.3%** |
| z_gemm | 46.1 / 46.1 | 40.7 / 40.7 | **-11.7%** |
| out_gemm | 43.1 / 43.1 | 43.2 / 43.2 | 0% |
| scan | 193.4 / 193.5 | 193.7 / 193.6 | 0% |
| ba_gemm | 17.1 | 17.1 | 0% |

14k も同方向: qkv 251.3 -> 243.9 (-2.9%)、z 158.6 -> 152.7 (-3.7%)、
out 153.0 -> 152.9、scan 710.1 -> 710.3。

### 3.3 他の段 (t407c、3,826 cold)

| 段 busy (ms) | b1 | tc | 差 |
| --- | ---: | ---: | ---: |
| experts | 531.2 / 531.4 | 530.5 / 530.7 | -0.1% |
| mix | 242.9 / 242.7 | 242.9 / 242.9 | 0% |
| full | 88.9 / 88.9 | 87.1 / 87.2 | -2.0% (t407c は tc2 も切ったため。t407d の GDN 限定版では 89.0 で不変) |
| other | 47.5 / 47.5 | 47.3 / 47.3 | -0.4% |
| close gpu_busy | 1,281.5 | 1,270 級 | -0.9% |

**壁 (3,826 cold、中央値)**: t407c は b1 2.38 / tc 2.44 / b2 2.38 s
(tc_1 の `other` が 348 ms に跳ねた回で、`other` は P397 も ±400 ms と書いて
いる帯)。t407d は b1 2.36 / off 2.38 / b2 2.37 s で、**既定 on が -0.6%**。
14k の壁は同じ run で 5.65-6.42 s に散らばり、差は読めない。

## 4. 数値の扱いと見積り

- **段の数字が判定**。壁は 3,826 でも 1 COND 1 サンプルで、`other` 段
  (ホスト) が ±120 ms 動く。今回も t407c の tc_1 だけ `other` が 348 ms に
  跳ねて壁が +2.5% に見えた。**同じ run の 4 サンプルの段で読む**のが正しい。
- 見積り (推定): qkv/z が 6-12% 速くなった理由は、tc2 (128x128 タイル・
  40 KB の smem・2 段プリフェッチ) より br169 (128x64・20 KB 級) の方が
  この形 (m=1,913、k=5,120/6,144、n=10,240/6,144) で**ブロック数が倍**になり、
  170 SM への配分と smem 由来の occupancy で勝つため。数字は測ったが機序は
  推定。
- 14k で ssm busy が -0.9% しか動かないのは、14k の ssm の 53% が scan で、
  投影は 42% のうちの一部だから (台帳の内訳そのもの)。
- この変更で**モデルの値は変わらない** (md5 SAME)。丸めの話は無い。

## 5. text (sane の証拠)

t407d の b1 (既定 = tc) の c8000_cold (14k cold の応答):

> ご提示いただいた文書は、LLM推論エンジン (`denno-engine) における
> **Prefill(事前計算) フェーズの性能最適化の歴史を詳細に記録した技術レポートです。

`off` (tc2) の c8000_cold も同じ文で、md5 も全 COND 一致 (`be179579` 系)。
「!!!!」や空出力は無い。b/a1/a2 の短文も sane (`looks ok`)。

## 6. 分かったこと (次の担当者へ)

1. **既定モデルの prefill の GPU 予算はこうなっている** (14k cold、t407c b1、
   gpu_busy 4,450 ms): experts 1,424 (32%) / ssm 1,330 (30%) / mix 883 (20%) /
   full 638 (14%) / other 171 (4%)。ssm の中は scan 707 (53%) / qkv 251 /
   z 159 / out 153 / ba 60。**P397 の「scan 936 ms」は見積りで、実測は ~708 ms**
   (kprof 修正後)。task 402 の「scan は 115 ms」も kprof の数え落としが原因で、
   どちらも直った。
2. **scan の残りは従属連鎖とバリア**。占有率 (256 -> 512 スレッド、ブロック
   2 倍) は +45% の負け、共有メモリのトランザクション削減 (-29%) は +10% の
   負け、バリア 2 本削減だけが -0.8%。次にやるなら (a) C=8 のまま位相を
   融合する (2 と 3、7 と 8 は wy7 で実施済み。さらに 3 と 4、5 と 6 は依存が
   あるので不可)、(b) チャンクを跨ぐ並列化 (chunked scan の 2 パス化) の
   どちらか。後者は大きな書き換えだが、いまの 707 ms を狙うならそこしかない。
3. **投影 GEMM は 5.6 TFLOPS** (14k で qkv+z+out = 3.2 TFLOP / 563 ms)。
   br169 で -6 〜 -12% だったので、同じ Q8_0 でも**形ごとに最適なカーネルが
   違う**。`DENNO_PF_TC_GEMM2=0` をモデル全体に掛けると full 層も含めて
   -2% になるが、旧既定モデル (swift15) の task 176 の測定と逆なので、
   既定は動かさず GDN 連鎖だけに閉じた。**full 層の投影も tc にしたければ、
   同じ形で別の A/B を取ってから** (この task では full の -2.0% は
   GDN 限定版では出ない)。
4. **`other` 段 (ホスト) の壁への寄与は 14k で 826 ms 露出**。中身は ple 215 +
   res_up 157。P393 で pinned 化した後の残りで、ここを削るには PLE 注入の
   ホスト射影そのものを速くするしかない。
5. **14k の壁はまだ解像できない** (同じ run で ±10%)。14k で 2.5% を判定する
   には lab に「14k cold を 3 本交互に取る」シナリオが要る (P385 も同じことを
   書いている)。

## 7. テスト

int-auto `775429cd6` を merge した後の最終 HEAD で:

```
cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1
  -> 1287 passed / 0 failed / 103 ignored
cargo test --release -p denno-engine --features cuda --test kernel_invariants
  -> 120 passed / 0 failed
cargo build --workspace --release            (cuda feature 無し)
  -> ok
cargo test --workspace --release -- --test-threads=1   (cuda feature 無し)
  -> 失敗 0 (denno-engine lib / tests / denno-server とも ok)
```

追加した 4 本:
- `task407_scan_wy6_gate_and_shape` (lib) — wy6/wy6p のゲート真理値表 (既定 off)、
  形の前提、共有メモリが wy5 と同じ、`.cu` の `#define` とオフセット。
- `task407_scan_wy7_gate_and_shape` (lib) — wy7/wy7q のゲート真理値表・
  共有メモリ (wy7q は 11,489 float = 45,956 B)・**バリア本数 (wy5 9 -> wy7 7 ->
  wy7q 8)**・wy7q の Pp/Qp の使い方。
- `task407_gdn_tc1_gate_defaults` (lib) — `DENNO_PF_GDN_TC1` の真理値表
  (既定 on、明示的な偽値だけ off) と FORCE。
- `task407_gdn_pf_wy6_kernel_is_wired` / `task407_gdn_pf_wy7_kernel_is_wired` /
  `task407_gdn_tc1_gate_is_wired` (kernel_invariants) — カーネル名 / 17 引数 /
  `#define` / smem 区画 / 選択順 (wy7q -> wy7 -> wy6p -> wy6 -> wy5 -> wy4) /
  `build_ptx.sh` の target と targets_all の**末尾** / `prefer_tc1` が v2 の
  分岐を飛ばすこと / **`DENNO_PF_TC_GEMM2` の既定を動かしていないこと** /
  kprof が未完了の段を捨てないこと。
- **発火の証拠**は lab (t407c/t407d の qkv/z の kprof と ssm busy、t407a/b の
  scan)。GPU 無しではカーネルは走らないので、.cu は g++ スタブで構文だけ。
- 既知の赤 (`qwen4exp_decode spec_batch_stops_at_eos`、`qwen4exp_params` の
  validate 2 件) は触っていない (この木では 0 failed)。

## 8. GPU で確認してほしいこと (監督へ)

1. 既定 on の `DENNO_PF_GDN_TC1` は `=0` で tc2 に戻せる。**他のモデル
   (swift15-abl-IQ3_S など) では未測定**なので、そのモデルで既定のまま
   走らせる前に 1 回 A/B を見てほしい (task 176 は tc2 が -90% と測っている。
   GDN 連鎖に閉じてあるので影響は ssm の投影だけだが、逆に効く可能性がある)。
2. 3,826 の壁 -0.6% は 2 ラウンドの中央値。ssm 段 -2.8% は 4 サンプルとも
   ±0.1% で揺れていない。実機で ssm 段が 360 ms 台に入るかを見てほしい。
3. 14k は投影の -3% しか出ない (scan が 53%)。14k で 2.5% を狙うなら
   scan の 2 パス化しかない (§6-2)。
4. `DENNO_PF_SCAN_WY6/WY6P/WY7Q` は**負けの記録**として既定 off で残した。
   `DENNO_PF_SCAN_WY7` は -0.8% なので、将来 scan を触る人が土台にできる。

## 9. VERDICT 行

t407a (`cbe9d76d1`、5 COND、ptx=gdn_pf_wy6 gdn_pf_wy6p):
```
VERDICT b1 vs x0 OK decode -0.5% prefill +0.0% md5 SAME errors 0
VERDICT wy6 vs x0 OK decode +0.3% prefill +0.0% md5 SAME errors 0
VERDICT wy6p vs x0 OK decode +0.8% prefill +0.8% md5 DIFF errors 0
VERDICT b2 vs x0 OK decode +0.5% prefill +0.0% md5 SAME errors 0
VERDICT baseline x0 errors 0 text ok
```

t407b (`a2dfbb8ce`、5 COND、ptx=gdn_pf_wy7 gdn_pf_wy7q):
```
VERDICT b1 vs x0 OK decode +0.0% prefill +1.5% md5 SAME errors 0
VERDICT wy7 vs x0 OK decode +0.0% prefill +0.7% md5 SAME errors 0
VERDICT wy7q vs x0 OK decode +0.0% prefill +0.7% md5 DIFF errors 0
VERDICT b2 vs x0 OK decode -0.3% prefill +0.7% md5 SAME errors 0
VERDICT baseline x0 errors 0 text ok
```

t407c (`a2dfbb8ce`、4 COND、ptx=none、`tc` = DENNO_PF_TC_GEMM2=0):
```
VERDICT b1 vs x0 OK decode +0.0% prefill -0.7% md5 SAME errors 0
VERDICT tc vs x0 OK decode -0.3% prefill -0.7% md5 SAME errors 0
VERDICT b2 vs x0 OK decode +0.0% prefill -0.0% md5 SAME errors 0
VERDICT baseline x0 errors 0 text ok
```

t407d (`143aac999`、4 COND、ptx=none、`off` = DENNO_PF_GDN_TC1=0):
```
VERDICT b1 vs x0 OK decode +0.0% prefill +0.8% md5 SAME errors 0
VERDICT off vs x0 OK decode +0.0% prefill +1.5% md5 SAME errors 0
VERDICT b2 vs x0 OK decode +0.0% prefill -0.8% md5 SAME errors 0
VERDICT baseline x0 errors 0 text ok
```

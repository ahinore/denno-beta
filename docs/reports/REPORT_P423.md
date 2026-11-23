ADOPT: yes t423a t423b t423c

# task 423 (dev2) — 既定モデルの prefill: PLE の gather を層ループの裏へ流す

branch `br423` = 開始時の `int-auto` **`d25b2e371`** (merge: br419 = 私の P419)。
作業中に int-auto は **`7c861b4b5`** (merge: br427 = decode の dn6 拡張と metrics 修正、
README) へ動いたので終端で merge した (§6、衝突なし)。

数値はすべて GPU ラボの実測 (`SCENARIO=depth8k`、`PROFILE=1`、
`model=flash-next-UD-IQ3_XXS`、`ptx=none`、全 COND に `DENNO_START_DROP_CACHE=1`)。
lab 名と測った SHA: **t423a = `a815a7e12`** (5 COND、load 2.20)、
**t423b = `a815a7e12`** (5 COND、load 1.71)、**t423c = `ae2418525`** (5 COND、
load 9.55、既定 on の確認)。3 ラウンド (上限 3)。

## 0. 結論 (6 行)

1. **採る (`ADOPT: yes t423a t423b t423c`)。target も満たした。**
   14k cold prefill が同じ run の交互の対で **7.20 / 6.97 -> 6.94 / 6.68 s = -3.6 / -4.2%**
   (t423c、候補が**先に**走る対なので位置効果は不利側)、3,826 は
   **8.37 / 8.37 -> 8.30 / 8.36 s (-0.8 / -0.2%)** で悪化なし。
2. **何をしたか**: 層優先 prefill は埋め込みの裏で PLE の gather を走らせるが、
   **層ループへ入る前に join** していた。gather は 90 B のランダム pread が
   IOPS 律速で 14k 380 ms / 3,826 200 ms あり、その待ちが丸ごと GPU の遊びだった。
   join を**層ループの PLE 層 (層 1) の入口へ移し**、受け渡しを**チャンクごとの
   mpsc チャネル**にした (`DENNO_PF_PLE_LJOIN`、**既定 on**。`=0` で切り戻し)。
3. **段の数字 (`[pre-kprof]` の ple = gather の待ち)**: 14k 380.4 / 375.1 ->
   **37.1 / 21.8 ms** (-90%)、3,826 197.8 / 198.0 -> **0.0 / 0.0 ms** (待ちゼロ。
   t423c)。層 0 の 1 層ぶん (14k 約 140 ms / 3,826 約 170 ms) と、層 1 の後半
   チャンクの待ちが消える。t423a/t423b の「層の入口でまとめて待つ」版は
   14k 236-241 ms で、-1.3〜-1.7% だった (チャネル版が上)。
4. **値は 1 bit も変えない**: 3 run・全 COND・全 6 文脈で md5 は **SAME**
   (`a:edaa84ad b:1938059f c2000_cold:b42d79d2 c2000_turn2:12fc5c36
   c8000_cold:672fec07 c8000_turn2:98e879d2`)。読む行・綴じ方は同じで、変えたのは
   **待つ場所と受け渡しの仕方**だけ。errors 0、text sane (§8)。
5. **decode は悪化しない**: 短い decode は 12.0-12.2 ms/tok (全 COND、run 内の差
   ±0.2 = ノイズ)。14k 後の decode は def の r1 だけ 18 ms/tok の外れが出たが、
   **同じコードの def2 は 15 ms/tok** で baseline (15/15) と同じ (§3.3)。
6. **分かったこと (次の担当者へ)**: 既定モデルの 14k の GPU は 4,426 ms / 壁 7,207 ms
   で **38% 遊んでいる** (§1.1)。内訳の大きい順は experts 1,417 / ssm 1,317 /
   mix 882 (別 stream で完全に隠れている) / full 638 で、**ホスト側で残っていた最大の
   塊が PLE の gather の待ち (379 ms)** だった。3,826 は 85% が expert の読み
   (DROP_CACHE 時) で、そこへ gather を重ねても壁は動かない (disk 律速)。

## 1. Do 1 / Do 2: 基準の取り直しと、なぜこの塊か

### 1.1 段の台帳 (t423a b1、現行 int-auto + task 指定の baseline env、ms)

| 段 | 3,826 wall / busy | 14k wall / busy |
| --- | ---: | ---: |
| mix | 238.9 / 242.8 | 21.5 / **882.3** (別 stream) |
| full | 129.9 / 88.9 | 1,046.5 / **638.5** |
| ssm | 420.9 / 360.6 | 2,199.0 / **1,317.1** |
| experts | **7,128.0 / 517.2** | 2,995.9 / **1,416.6** |
| other | 407.3 / 47.5 | 944.3 / 171.6 |
| close wall | **8,325.1** / gpu 1,257.0 | **7,207.2** / gpu **4,426.2** |

- **3,826 はディスク**: `[pf-host] up_wait=6,610.9 ms` (experts の露出) = 壁の 79%。
  22.4 GB の expert 重みを読む。ここは P227/P385 が「3 往復の帯域律速、バイトを
  減らす以外に手が無い」と閉じている。
- **14k は GPU 38% 遊び**: wall 7,207 に対し gpu_busy 4,426。`[pre-kprof]` は
  **embed 26.0 / ple 378.8 / layer 6,633.8 / other 167.9 ms**。層ループの外の
  **ple 378.8 ms は GPU が完全に遊ぶ区間**で、内訳は 90 B x 167k 回の pread
  (IOPS 律速。task 357/419 が本数を 32->128 本へ増やして 681 -> 381 ms まで詰めた)。
- GPU 側の 4 段は 397 (wy5) / 407 (tc1) / 416/420 (K 系) / 419 / 422 / 427 が詰めて
  おり、どれもカーネルを触る話になる。**ホスト側で残っていた最大の塊が PLE の
  gather の待ち**だったので、そこを取った (カーネル不要)。

### 1.2 なぜ待ちが丸ごと露出していたか

gather は `DENNO_PF_EMBED_OVL` (task 415) で背景スレッドに出ており、埋め込み
(14k 26 ms) とだけ重なっていた。**層ループへ入る前に join** していたので、
残り 380 ms は何も走っていない GPU の前で待っていた。PLE 層は**層 1** で、行を
使うのはその層の入口だけなので、join をそこまで遅らせれば層 0 の 1 層ぶんが丸ごと
重なる。さらにチャンクごとに受ければ、層 1 の後半のチャンクは gather が先に
終わっている。

## 2. 変えたこと

### 2.1 `DENNO_PF_PLE_LJOIN` (新規、**既定 on**)

- `qwen4exp.rs` の層優先 prefill: 背景スレッドの gather を `JoinHandle` ではなく
  **`mpsc::channel`** で渡す。スレッドはチャンク順に `send` し、受け手は
  **層ループの PLE 層で 1 チャンクずつ `recv`** する (`ples_all` に push)。
- off (`=0`) は従来どおり層ループの前に全部受ける (同じチャネルを使うので、
  経路は 1 本のまま)。背景スレッドが居ないとき (`DENNO_PF_EMBED_OVL=0` /
  `DENNO_PF_EMBED_PAR<=1`) は従来の直列 gather。
- 早期撤退 (層ループの途中で bail) では受信側が落ちるので、送信側は `send` の
  失敗で止まる (スレッドは残らない)。gather がエラーを返したら、そのチャンクを
  受けた時点で同じエラーを返す (従来は join で返していた)。

### 2.2 `DENNO_PF_PLE_THREADS` の上限は 128 のまま (測って戻した)

t423a で上限を 512 に上げて 256 本を測ったが、**`ple_par_parts` (ple.rs) 側にも
128 の丸め**があり、ゲートを上げても 128 本のままだった (ple 段 239.8 -> 241.2 ms、
差なし)。task 419 の既定 128 と丸めをそのまま残した。**128 本はもう
デバイスのキューが埋まった値ではない可能性がある**が、そこを詰めるには
`ple_par_parts` の丸めを外して測り直す必要がある (次タスク候補)。

## 3. 数値 (lab、同じ run の中の対)

### 3.1 t423c (`ae2418525`、5 COND、load 9.55) — 既定 on と `=0` の交互の対

| COND | env | 14k cold s | 3,826 r1 / r2 s | ple 14k ms | ple 3,826 ms | 短い decode |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| x0 (捨て) | — | 17.39 (外れ) | 12.74 / 8.35 | 5,083.9 | 0.0 | 12.0 |
| def | 既定 (LJOIN on) | **6.94** | 8.30 / 8.30 | **37.1** | **0.0** | 12.2 |
| off | `DENNO_PF_PLE_LJOIN=0` | 7.20 | 8.37 / 8.35 | 380.4 | 197.8 | 12.1 |
| def2 | 既定 | **6.68** | 8.35 / 12.79 (外れ) | **21.8** | **0.0** | 12.0 |
| off2 | `=0` | 6.97 | 8.37 / 8.36 | 375.1 | 198.0 | 12.0 |

- **対 (候補が先に走る = 位置効果は候補に不利)**:
  def 6.94 対 off 7.20 = **-3.6%**、def2 6.68 対 off2 6.97 = **-4.2%**。
- ple 段は 14k **-90%**、3,826 は**待ちゼロ**。3,826 の壁が動かないのは
  expert の読み (壁の 79%) が律速で、gather を層ループへ移すとそのぶん層が
  伸びるため (`[pre-kprof]` layer 8,100 -> 8,236 ms)。
- 3,826 の壁は **-0.8 / -0.2%** (悪化なし)。

### 3.2 t423a / t423b (`a815a7e12`、層の入口でまとめて待つ版)

| run | COND | 14k cold s | 3,826 r1 / r2 s | ple 14k | ple 3,826 |
| --- | --- | ---: | ---: | ---: | ---: |
| t423a | b1 | 7.21 | 8.33 / 8.28 | 378.8 | 200.4 |
| t423a | lj | 6.96 | 8.31 / 8.31 | 239.8 | 27.1 |
| t423a | lj256 | **6.87** | 8.35 / 8.34 | 241.2 | 37.8 |
| t423a | b2 | 6.97 | 8.31 / 8.31 | 377.8 | 200.9 |
| t423b | lj | 6.88 | 8.33 / 8.31 | 236.0 | 47.5 |
| t423b | b1 | 6.97 | 8.34 / 8.29 | 380.2 | 200.2 |
| t423b | lj2 | **6.86** | 8.33 / 8.29 | 235.1 | 61.3 |
| t423b | b2 | 6.98 | 8.35 / 8.32 | 379.4 | 199.2 |

- t423b の対: lj (2 番目) 6.88 対 b1 (3 番目) 6.97 = -1.3%、
  lj2 (4 番目) 6.86 対 b2 (5 番目) 6.98 = -1.7%。
  **同じ変更の「まとめて待つ」版は 14k で -1.3〜-1.7%** で、チャネル版 (-3.6〜-4.2%)
  より弱い。差は ple 段の残り (236 ms 対 22-37 ms) と一致する。
- lj256 (PLE_THREADS=256) は lj と差なし = `ple_par_parts` の 128 の丸め (§2.2)。

### 3.3 decode と外れ

- 短い decode は 3 run 全 COND で **11.9-12.2 ms/tok**。
- 14k 後の decode: t423c の def r1 だけ **18 ms/tok** (1.85 s)。同じコードの
  def2 は 15 ms/tok で、off/off2 (15/15) と同じ。x0 が 17.4 s / ple 5.1 s の
  外れだった直後の COND で、機械の状態の外れと読む (§3.1 の 3,826 も同様の外れが
  出ている)。
- 3,826 と 14k の壁には run 内で 1 本ずつ外れが出る (`DROP_CACHE` のディスク)。

## 4. 数値の扱いと見積り

- **`DENNO_START_DROP_CACHE=1` を全 COND に付けた** (task 415/419 と同じ手)。
  付けないと 14k は COND の位置で 13% 動く。付けると各 COND が同じ冷え方から
  始まるので、同じ run の中の対が読める。絶対値は task の状態 (3,826 2.5 s /
  14k 6.2 s) と比べられない (毎回ディスクから読み直す)。
- **見積り (推定)**: LJOIN は 14k で -0.26 s (対の差)、3,826 で -0.0 s。
  本番の暖かい状態では 14k の ple 段が 380 -> 30 ms になり、壁は -1.5〜-2%
  程度と読む (DROP_CACHE 時より gather が速いので隠せる量も小さい)。
  **採否は lab の対 (段 -90% / 壁 -3.6〜-4.2%) で決めた。**
- **他は悪化していない**: experts / ssm / full / mix の busy は run 内で ±1% 以内、
  3,826 の壁も -0.2〜-0.8%。

## 5. テスト

merge 後の最終 HEAD (`7c861b4b5` を merge した後) で:

```
CARGO_TARGET_DIR=/tmp/t423 cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1
  -> 1308 passed / 0 failed / 103 ignored
CARGO_TARGET_DIR=/tmp/t423 cargo test --release -p denno-engine --features cuda --test kernel_invariants
  -> 125 passed / 0 failed
CARGO_TARGET_DIR=/tmp/t423-nocuda cargo build --workspace --release            (cuda feature 無し)
  -> ok
CARGO_TARGET_DIR=/tmp/t423-nocuda CUDA_VISIBLE_DEVICES=-1 cargo test --workspace --release -- --test-threads=1
  -> (結果は §5.1)
```

更新/追加したテスト:
- `task423_ple_ljoin_gate_truth_table` (lib) — 既定 on、`0`/`off`/`false`/`no` だけが偽。
- `t349_ple_pipe_widens_the_lm_gather_pool` と `task415_embed_par_and_overlap_are_wired`
  (kernel_invariants) — 背景スレッドがチャネルで送り、off は層ループの前で、
  on は PLE 層で受けていること (ソース走査)。

### 5.1 cuda 無しの workspace テスト

```
CARGO_TARGET_DIR=/tmp/t423-nocuda CUDA_VISIBLE_DEVICES=-1 \
  cargo test --workspace --release -- --test-threads=1
  -> 63 スイート / 1,482 passed / 0 failed (exit 0)
```
既知の赤 (`qwen4exp_decode spec_batch_stops_at_eos`、`qwen4exp_params` の validate 2 件)
はこの木では出ていない。

## 6. merge / 終端

終端で int-auto **`7c861b4b5`** (merge: br427 = decode の dn6 拡張、metrics 修正、
README) を merge した。**衝突は無かった** (br427 は `qwen4exp.rs` の mix-qchain の
計数と `kernel_invariants.rs` の末尾、`cuda.rs` の dn6 カーネル)。br427 は decode
だけなので、私の prefill の変更とは独立。merge 後に §5 の関門を回し直した。

## 7. GPU で確認してほしいこと (監督へ)

1. 既定は `DENNO_PF_PLE_LJOIN=1` (on)。切り戻しは `=0` (従来の join)。
2. 本番で見てほしいのは **14k 級の長い prefill の `[pre-kprof]` ple 段**:
   暖かいと 380 -> 30 ms、壁は -1.5〜-2% 程度 (推定)。
3. 3,826 は変わらない (disk 律速)。decode も変わらない (12.0-12.2 ms/tok)。
4. `DENNO_PF_PLE_THREADS` は 128 のまま。**`ple_par_parts` の 128 の丸め**を
   外すと本数を増やせる余地がある (未測定)。

## 8. text (sane の証拠)

t423c の def、c8000_cold (14k cold の応答、`TXTD`):

> ご提示いただいた文書は、LLM推論エンジン（`denno-engine）における **Prefill（事前計算）処理の性能最適化の記録です。\n\n以下に、この文書の技術的詳細を、**「問題の特定」、「解決プロセス」、「主要な発見と教訓**

def / off / def2 / off2 の 4 COND とも同じ文で、md5 も全 COND・全 6 文脈で一致。
「!!!!」や空出力は無い。

## 9. VERDICT 行

t423c (`ae2418525`、5 COND、ptx=none、既定 on と `=0` の交互):
```
VERDICT def vs x0 OK decode +1.6% prefill +0.9% md5 SAME errors 0
VERDICT off vs x0 OK decode +0.7% prefill +0.9% md5 SAME errors 0
VERDICT def2 vs x0 OK decode +0.0% prefill +0.0% md5 SAME errors 0
VERDICT off2 vs x0 OK decode +0.0% prefill +0.0% md5 SAME errors 0
VERDICT_DEPTH def c2000_cold vs x0 decode -30.9% prefill -21.3%
VERDICT_DEPTH def c8000_cold vs x0 decode +17.1% prefill -60.1%
VERDICT_DEPTH off c8000_cold vs x0 decode -1.9% prefill -58.6%
VERDICT_DEPTH def2 c8000_cold vs x0 decode -1.9% prefill -61.6%
VERDICT_DEPTH off2 c8000_cold vs x0 decode -2.5% prefill -59.9%
VERDICT baseline x0 errors 0 text ok
```
(`vs x0` は捨ての 1 本目 (14k 17.39 s / ple 5,083.9 ms の外れ) 相手なので使わない。
採否は §3.1 の同じ run の対で読む。)

t423a (`a815a7e12`、5 COND、まとめて待つ版):
```
VERDICT b1 vs x0 OK decode +0.0% prefill +0.9% md5 SAME errors 0
VERDICT lj vs x0 OK decode +0.0% prefill -0.9% md5 SAME errors 0
VERDICT lj256 vs x0 OK decode +0.0% prefill -0.9% md5 SAME errors 0
VERDICT b2 vs x0 OK decode -0.3% prefill -0.9% md5 SAME errors 0
VERDICT baseline x0 errors 0 text ok
```

t423b (`a815a7e12`、5 COND、位置を振った確認):
```
VERDICT lj vs x0 OK decode -0.7% prefill +0.9% md5 SAME errors 0
VERDICT b1 vs x0 OK decode +0.0% prefill +0.9% md5 SAME errors 0
VERDICT lj2 vs x0 OK decode -0.3% prefill -0.9% md5 SAME errors 0
VERDICT b2 vs x0 OK decode -0.7% prefill +0.9% md5 SAME errors 0
VERDICT baseline x0 errors 0 text ok
```

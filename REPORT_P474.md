ADOPT: yes t474b

# task 474 (dev3) — UD-Q4_K_XL: attention の裏で遊ぶディスクの窓に、捨てない先読み
# (RAM の置き場) を置いた。14k cold prefill が -24.8 / -25.5% (2 サンプル)、
# 3,826 と短い decode と既定モデルは変わらない

branch: **br474** = 開始時の int-auto **8455128** (merge: br465)。**br470 を merge
した** (`5e75002`。task 470 の早期 router の配線 `pf_early_route_prefetch` と
そのゲートを使うため。470 の先読み本体 = readahead / pread 版は差し替えた)。
終端で int-auto を再 fetch したが **8455128 のまま** (動いていなかった) なので
追加の merge は無い。

lab 名 (すべて `tree=denno-dev3` / `SCENARIO=depth8k` / `PROFILE=1` / `ptx=none`):
- **t474a** = `2ffd5c9` / `model=flash-next-UD-Q4_K_XL` / `load_before=1.69` /
  COND: x0 b **rc** b2 — 置き場の初版 (窓の下限なし)
- **t474b** = `1f45fc1` / `model=flash-next-UD-Q4_K_XL` / `load_before=1.83` /
  COND: x0 b **rc** **rc2** b2 — 採否はこれで読む (rc と rc2 は同じ env の
  2 サンプル。14k は depth8k の仕様で round 1 にしか無いため)
- **t474c** = `1f45fc1` / `model=flash-next-UD-IQ3_XXS` (既定モデル) /
  `load_before=4.62` / COND: x0 b rc b2 — 既定モデルの確認

baseline COND はタスク文のとおり (全 COND 共通):
`DENNO_EXPERT_IQ_GPU=1 DENNO_EXPERT_VRAM_MB=22000 DENNO_EXPERT_GPU_MIN_HIT=0
DENNO_GPU_WEIGHT_MB=12000`。rc / rc2 はこれに `DENNO_PF_EARLY_ROUTE=1` を足しただけ。

## 0. 結論 (7 行)

1. **採る (`ADOPT: yes t474b`)。** 14k cold prefill が **-24.8% / -25.5%**
   (t474b の rc / rc2 対 baseline の中央値 15.34 s)、t474a でも **-13.6%**
   (12.56 対 14.54 s)。目標 (14k cold 5% 以上) を満たす。
2. **機序**: 1 チャンク目の routing を解いた直後 (残りチャンクの attention が
   走るあいだ) に、その層で要る expert を **プロセスの RAM の置き場**へ読む。
   MoE の写し (`pf_copy_into_pin`) は (パート, 絶対オフセット, 長さ) が
   **完全一致**した区間だけ置き場から memcpy し、外れれば従来どおり pread する。
   読む場所が変わるだけでバイト列は同じ = **数値は 1 bit も変わらない**。
3. **置き場は実際に使われた**: t474b の rc で `ra_MB=12,915.6` に対し
   `ra_hit_MB=12,915.6`、`ra_pct=100` (14k。rc2 は 13,208.6 / 13,208.6)。
   t474a も 12,424.3 MB / 100%。
   3,826 は `ra_MB=1,858` / `ra_pct=95`。
4. **3,826 は変えない** (残り 1 チャンク = 窓 9.5 ms に 42 MB の先読みが収まらず、
   t474a で ±0 だった)。**残りの窓が 4 チャンク以上の層だけ**先読みするので、
   3,826 はゲート on でも `rc_plan` を呼ばない (round 2 で b 8.40 / rc 8.40 /
   rc2 8.36 / b2 8.29 s)。
5. **短い decode は変わらない** (t474b: b 14.3 / rc 14.3 / rc2 14.4 / b2 14.2
   ms/tok)。decode はこの経路を通らない。
6. **既定モデル (flash-next-UD-IQ3_XXS) も変わらない**: t474c で md5 が
   全 COND で SAME (c8000 を含む)、3,826 は round 2 で b 2,100 / rc 2,116 /
   b2 2,098 ms、14k は b 5,617 / rc 5,178 / b2 5,151 ms (差は揺れの内側)。
   ただし t474c で **置き場 14.2 GB を読んで 1 バイトも使われない**
   (`ra_pct=0`) ことが分かったので、RAM tier に載っている expert と
   `DENNO_PF_PREAD=0` の層は先読みしない門を足した (UD では全 expert が pread
   なので 1 命令も変わらない。§3.3)。
7. 既定 on にした (`DENNO_PF_EARLY_ROUTE=0` で戻せる)。`.cu` / `.ptx` は
   1 バイトも変えていない (PTX 再生成不要)。

## 1. 前提: 現在の int-auto の UD-Q4_K_XL の段プロファイル (t474b の b)

### 1.1 prefill (`[q4ledger-pf]`、ms)

| 段 | 3,826 (b_2) | 14k (b_1) |
| --- | ---: | ---: |
| mix | 268.7 | 23.5 |
| full | 118.2 | 765.4 |
| ssm | 412.4 | 2,035.1 |
| experts | 7,498.1 | **12,310.3** |
| other | 106.9 | 714.5 |
| **wall** | **8,404.3** | **15,848.8** |
| `gpu_busy` | 1,107.7 | 3,848.0 |
| `exposed_host` | 7,296.6 | 12,634.4 |
| `[exp-pf-stream] copy_s` | 7.27 s | 11.83 s |
| `up_MB` | 43,532 | 52,754 |

壁の 78-89% は **expert の写し** (b の 3,826 で 89%、14k で 78%)。`copy_s` は 5.9 GB/s で P325 のデバイス上限
6.2 GB/s の 95-97% なので、写しの中は削れない。**重ねる**しかない。
`gpu_busy` (3,826 で 1.1 s / 14k で 3.85 s) が写しと直列になっているのが
残っている最大の項で、これは P470 が見つけた「attention のあいだディスクが
遊ぶ窓」(3,826 で 1 層 9.5 ms × 48 = 0.46 s、14k で 62 ms × 48 = 3.0 s) と
同じもの。P470 は窓をページキャッシュで埋めようとして失敗した (投げたバイトの
11-28% しか返らない)。**読んだバイトをプロセスが持てば取りこぼしが無い**、
がこのタスクの一手 (P470 §7.1 の 1 番目の提案そのもの)。

### 1.2 decode (短い decode、ms/tok)

t474b の表: b 14.3 / rc 14.3 / rc2 14.4 / b2 14.2。段の内訳は P470 §1.2 のまま
(experts 5.09 / mix 2.80 / ssm 2.79 / full 1.21 / shared 0.65 / head 0.61)。
decode は `pf_copy_into_pin` を通らないので、この変更の影響はゼロ。

## 2. なぜこの一手を選んだか

- 段プロファイルで最大の項は expert の写し (14k で壁の 78%、3,826 で 89%)。
  帯域は上限の 95-97% で、スレッド数 (P424) / 読み口 (P394) / 束ね方 (P449) /
  読みの分割 (P442) は閉じている。**写しの中はもう削れない**。
- 写しの外側でまだ誰も取れていない最大のものが「attention のあいだ遊ぶ
  ディスク」。P470 がページキャッシュ経由で負けた理由は「この環境の
  ページキャッシュは 50-300 MB の先読みを 60 ms 保持しない」ことで、
  **置き場をプロセスが持てば同じ制約が無い**。P470 §7.1 が次の的として
  名指しした。
- decode 側は experts (5.09) が 435/442/449/452 で閉じ、mix は 431/437/464/468、
  ssm/full/head はカーネル律速で、残っている大きな非カーネル項が無い。
- 既に却下されたものとの違い: task 394 の `DENNO_PF_LAYER_PRE` は層の全 expert
  (512 本) を読んで 1.73 倍になり負けた。ここは **そのチャンクの実際の routing**
  なので集合は層全体の部分集合 (無駄読み 0)。task 337 の expert 予測・先読みは
  decode の話で、これは prefill の窓。

## 3. 変えたこと (4 commit)

### 3.1 `2ffd5c9` — RAM の置き場と、写しの側の消費

- `crates/denno-engine/src/model/exp_pf.rs`: `rc_layout` (純関数。runs を置き場の
  オフセットへ割り当てる) / `rc_plan` / `rc_commit_one` / `rc_has` /
  `rc_copy_into` / `rc_stop` / `rc_join` / `rc_pool` / 台帳。
  置き場はプロセスで 1 回だけ確保し、層ごとに使い回す (既定 512 MiB、
  `DENNO_PF_RA_CACHE_MB`。0 で無効)。並列度は `DENNO_PF_RA_THREADS` (既定 8)。
- 先読みの実行役は層ごとに 1 本の背景スレッド + 専用の rayon プール。
  **1 範囲を読み終えるたびに索引へ入れる**ので、読みかけのバイトを MoE が
  掴まない。層の MoE の直前 (`rc_stop`) で止める。
- `crates/denno-engine/src/cuda.rs`: `pf_copy_into_pin` が、区間**丸ごと**が
  置き場に載っていれば (`rc_has`) `rc_copy_into` で memcpy し、外れれば
  従来どおり `read_part_ptr`。`pf_stream_release` で実行役を join する。
- `crates/denno-engine/src/model/qwen4exp.rs`: `pf_early_route_prefetch` の
  後半 (readahead / pread して捨てる) を「置き場へ読む」に差し替え。
  層の MoE の直前で `rc_stop`。

### 3.2 `1f45fc1` — 窓が 4 チャンク未満の層では先読みしない

t474a で 3,826 (残り 1 チャンク) は 1 層 42 MB の先読みに 11.3 ms かかり、
窓 (9.5 ms) に収まらない = はみ出した分が写しと帯域を奪い合って ±0 だった。
`early_route_window_ok(nt0, nt_all)` (残り ≥ 4 × nt0) を足し、3,826 は
ゲート on でも `rc_plan` を呼ばない (分岐 1 回)。

### 3.3 `af7d458` — RAM tier / mmap の写しでは先読みしない + 既定 on

t474c で既定モデルは置き場 14.2 GB を読んで `ra_pct=0` だった。既定モデルの
expert は RAM tier から来ていて (`src: None` = mmap の memcpy)、MoE は
ディスクを読まない。`ram_peek` で tier ヒットの expert を飛ばし、
`DENNO_PF_PREAD=0` の層も飛ばす。**UD-Q4_K_XL は `pread_MB == up_MB`
(全 expert が pread) なので、この 2 つの門は UD の経路を 1 命令も変えない**
(測った数字がそのまま最終コードの数字)。既定を on にした。

### 3.4 `2281f24` — 使わなくなった先読みの実行役を消した

task 470 の pread 版の `early_route_worker_idle` / `early_route_worker_set` /
`note_early_route_read_ns` / `note_early_route_skip` と `er_ms` / `er_skip` は
置き場の側 (`rc_join` / `note_rc_read_ns`) が同じことをするので消した。
台帳は `er_layers` / `er_experts` / `er_MB` / `er_calls` だけ。

## 4. lab の結果

### 4.1 t474a (`2ffd5c9`、窓の下限なし)

14k cold (r1 のみ、1 COND 1 サンプル):

| COND | 14k wall | `copy_s` | `ra_MB` / `ra_pct` |
| --- | ---: | ---: | --- |
| x0 (捨て) | 15.997 | 9.15 | — |
| b | 15.196 | 11.83 | — |
| **rc** | **12.563** | **8.14** | 12,424 / 100% |
| b2 | 13.882 | 9.93 | — |

3,826 cold: b 9.364 / 9.304、rc 8.960 / 10.038、b2 8.483 / 9.397 s。
**rc は baseline の揺れ (8.48-9.40) の内側 = ±0**。`ra_ms` は 3,826 で 542 ms
(48 層 = 11.3 ms/層 > 窓 9.5 ms)、14k で 2,463 ms (51 ms/層 < 窓 62 ms)。
→ 窓に収まるかどうかがそのまま勝ち負けだった (§3.2 の根拠)。

### 4.2 t474b (`1f45fc1`、採否はこれで読む)

14k cold (r1 のみ、1 COND 1 サンプル):

| COND | 14k wall | baseline 中央値対 | `copy_s` | `ra_MB` / `ra_hit_MB` / `ra_pct` |
| --- | ---: | ---: | ---: | --- |
| x0 (捨て) | 13.152 | — | 9.15 | — |
| b | 15.849 | — | 11.83 | — |
| **rc** | **11.528** | **-24.8%** | 7.55 | 12,915.6 / 12,915.6 / 100% |
| **rc2** | **11.433** | **-25.5%** | 7.38 | 13,208.6 / 13,208.6 / 100% |
| b2 | 14.828 | — | 9.08 | — |

(baseline の中央値 = (15.849 + 14.828) / 2 = 15.339 s。rc 対 b で -27.3%、
対 b2 で -22.3%。同じ run の b と b2 が 6.4% 揺れているので、中央値で読んだ。)

3,826 cold (round 2 = 同じ暖かさの 4 COND):

| COND | wall | `copy_s` |
| --- | ---: | ---: |
| b | 8.404 | 7.27 |
| rc | 8.401 | 7.23 |
| rc2 | 8.363 | 7.22 |
| b2 | 8.290 | 7.18 |

= **差なし** (rc は baseline の中央値 8.35 に対して ±0.6%)。意図どおり
(残り 1 チャンクなので先読みしない)。

decode (UD 短): b 14.3 / rc 14.3 / rc2 14.4 / b2 14.2 ms/tok = 差なし。
md5: rc の `b` / `c2000_turn2` / `c8000_turn2` が COND 間で割れるのは
**baseline 同士 (b と b2) でも起きている既知の揺れ** (P466 §5 と同じ) で、
`a` / `c2000_cold` / `c8000_cold` は全 COND で SAME。errors 0、text は
「looks ok」(TXTD は「ご提示いただいた文書は、LLM推論エンジン…」で始まる
日本語の要約で、内容も従来と同じ)。

### 4.3 t474c (既定モデル flash-next-UD-IQ3_XXS、`1f45fc1`)

| COND | 3,826 r2 wall | 14k wall | decode ms/tok | md5 |
| --- | ---: | ---: | ---: | --- |
| b | 2,100 | 5,617 | 11.1 | SAME |
| rc | 2,116 | 5,178 | 11.1 | **SAME** |
| b2 | 2,098 | 5,151 | 11.1 | SAME |

`ra_MB=14,179.3` / `ra_pct=0` / `ra_ms=779`。**読んだ 14.2 GB が 1 バイトも
使われていない** (§3.3 で門を足した)。壁は 3,826 で ±0.8%、14k で
rc 5,178 対 b2 5,151 = +0.5% (b 5,617 は同じ run の外れ)。md5 は
**全 COND で完全一致** (c8000 を含む) = 既定モデルの値は動かない。

### 4.4 VERDICT 行 (t474b のまま)

```text
VERDICT b vs x0 OK decode -8.0% prefill -14.4% md5 DIFF errors 0
VERDICT rc vs x0 OK decode -8.3% prefill -14.4% md5 DIFF errors 0
VERDICT rc2 vs x0 OK decode -7.5% prefill -14.4% md5 DIFF errors 0
VERDICT b2 vs x0 OK decode -8.5% prefill -14.4% md5 DIFF errors 0
VERDICT_DEPTH b c2000_cold vs x0 decode -3.7% prefill -7.1%
VERDICT_DEPTH b c2000_turn2 vs x0 decode -0.2% prefill -4.0%
VERDICT_DEPTH b c8000_cold vs x0 decode +178.5% prefill +20.5%
VERDICT_DEPTH b c8000_turn2 vs x0 decode +100.8% prefill +39.3%
VERDICT_DEPTH rc c2000_cold vs x0 decode -49.0% prefill -16.3%
VERDICT_DEPTH rc c2000_turn2 vs x0 decode -27.5% prefill -21.8%
VERDICT_DEPTH rc c8000_cold vs x0 decode -35.2% prefill -12.3%
VERDICT_DEPTH rc c8000_turn2 vs x0 decode -5.4% prefill -6.6%
VERDICT_DEPTH rc2 c2000_cold vs x0 decode -44.2% prefill -12.8%
VERDICT_DEPTH rc2 c2000_turn2 vs x0 decode -27.5% prefill -18.5%
VERDICT_DEPTH rc2 c8000_cold vs x0 decode -38.5% prefill -13.0%
```

(`vs x0` は捨て COND 相手なので採否に使っていない。採否は §4.2 の対。
`baseline x0 errors 0 text ok` も出ている。)

## 5. 数値の扱い

- **14k cold は depth8k の仕様で round 1 にしか無い = 1 COND 1 サンプル。**
  そのため t474b では同じ env の COND を 2 本 (rc / rc2) 入れて 2 サンプル取り、
  t474a の 1 サンプルと合わせて 3 サンプルで読んだ (-25.5 / -24.8 / -13.6%)。
  3 サンプルとも baseline の中央値より速い。
- **この lab のディスクは run 内でも動く**: t474b の b と b2 は同じバイナリで
  15.849 対 14.828 s (6.4%)、t474a では 15.196 対 13.882 s (8.6%)。
  t474b の b2 は `full` 765 → 1,544 ms / `other` 714 → 1,414 ms と他段も
  跳ねている回で、copy_s も 11.83 対 9.08 s。だから 1-2% は読めない。
- **`copy_s` を主指標にした**: rc は 7.55 / 7.38 s 対 baseline 11.83 / 9.08 s。
  置き場から返ったバイト (12.9 GB) を 5.9 GB/s で読むと 2.2 s、memcpy
  (推定 15 GB/s) なら 0.9 s = 差 1.3 s (推定)。実測の `copy_s` の差は
  2.2-4.3 s で、外れ値の影響も含むが向きは `ra_hit_MB` と一致している。
- 3,826 は round 2 (同じ暖かさの 4 COND) で ±1% に収まっている。
- md5 は「変わらない」ことの証拠には使っていない (baseline 同士でも割れる)。
  値が変わらない根拠は「同じバイト列を別の場所から読むだけ」という構築
  (完全一致の索引) と、既定モデルで全 COND の md5 が一致したこと。
- マイクロベンチは取っていない (GPU の無いコンテナ)。`ra_ms` / `copy_s` /
  `ra_hit_MB` は lab の実測。

## 6. テストと既定モデル

最終 HEAD (4 commit、int-auto 8455128 の上):

```text
CARGO_TARGET_DIR=/tmp/t474 cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1
  -> 1343 passed; 0 failed; 103 ignored   (task 474 で 3 本追加)
同 --test kernel_invariants
  -> 139 passed; 0 failed                 (task 474 で 1 本追加)
CARGO_TARGET_DIR=/tmp/t474nc2 cargo build --workspace --release          (cuda 無し)
  -> ok
同 cargo test --workspace --release --no-fail-fast -- --test-threads=1
  -> 下記
```

足したテスト:
- `exp_pf::tests::ra_cache_layout_and_index` — 割り当て (上限で切る / 長さ 0 を
  飛ばす) と索引の**完全一致** (位置・長さ・パートが 1 つでも違えば当たらない)。
- `exp_pf::tests::ra_cache_env_truth_table` — 置き場の MiB (32..2048、0 で無効)
  と並列度 (1..32)。
- `exp_pf::tests::early_route_window_gate` — 窓の下限 (3,826 は不可、14k は可、
  残り 4 チャンクちょうどは可)。
- `kernel_invariants::t474_ram_readahead_cache_is_wired` — 窓の終わり (`rc_stop`)
  が層の MoE の直前にあること、写しが置き場 → ディスクの順で落ちること、
  `rc_has` が完全一致だけを見ること、t470 の負けた経路
  (`readahead_part` / `early_route_worker_idle`) が戻っていないこと。
- `kernel_invariants::t470_early_route_is_wired` は本体の検査を新しい配線へ更新。

既知の赤: `qwen4exp_decode spec_batch_stops_at_eos` と `qwen4exp_params` の
validate 2 件は今回も出ていない (1343 本すべて緑)。`denno-server` の
`slot_leak::dropped_stream_receivers_do_not_leak_kv_slots` が並列ビルドと
同時に走った 1 回だけ `QueueFull` で落ちたが、単独では緑 (既存の揺れ。
このタスクの変更は server に触っていない)。

### 6.1 既定モデルについて

`DENNO_PF_EARLY_ROUTE` は **既定 on**。既定モデルは t474c で
「3,826 は窓の下限で発火しない / 14k は RAM tier なので発火しない」= 実質
従来と同じ経路になる (md5 全 COND 一致、壁は揺れの内側)。もし既定モデルで
先読みを切りたければ `DENNO_PF_EARLY_ROUTE=0`。

## 7. 次の担当者へ

1. **窓の使い切りがまだ 8 割**: 14k で `ra_ms=2,463 ms` (51 ms/層) に対し窓は
   62 ms/層。予算は `DENNO_PF_EARLY_ROUTE_KB` (既定 24 KiB/token = 287 MB/層)。
   30 KiB にすると 359 MB/層 = 窓の 97% になる (推定 -2〜3% 追加)。
   ただしディスクが詰まると窓からはみ出すので、lab で 24 と 30 を同じ run に
   入れて比べるのがよい。
2. **memcpy を消す道**: 置き場から pinned への memcpy (12.4 GB / 15 GB/s =
   0.8 s) が残っている。置き場を**スロットの pinned そのもの**にできれば
   0.8 s が消える (推定 -6%)。ただし MoE のグループ分け (`bounds`) と
   expert 昇順の並びを先読み側で再現する必要があり、順序が 1 つでも違うと
   別の重みを積む。索引の完全一致を保ったまま dst を一致させる設計が要る。
3. **既定モデル側**: RAM tier に載っている expert を飛ばす門を足したので、
   既定モデルでは先読みが空になる。逆に「tier を埋めながら先読みする」形に
   すると既定モデルでも窓が使える可能性がある (t474c の `ra_pct=0` は
   「tier が既に持っていた」ことの証拠)。
4. **早期 router のコスト**: 先読みする層では router をもう 1 回解く
   (P470 の配線)。14k で 48 回ぶん。今回の数字には入っているが、分離して
   測っていない。

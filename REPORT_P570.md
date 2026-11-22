ADOPT: yes t566a t570b t570c

# task 570 (dev3) — UD-Q4_K_XL の prefill: **ステージングの読みを 3 グループ同時に流す**
# (`DENNO_PF_UP_WORKERS=3` + `DENNO_PF_STREAM_LOOKAHEAD_PREAD=2`) を採った。
# 3,826 cold が **-3.8 .. -8.2%** (同じ run の b/b2 の柵の取り方で幅)、14k cold が
# **-4.5 .. -5.1%**。decode 12.9 ms/tok 不変、既定モデルは構造的に不変
# (t570b でも decode 32.3 / prefill 29.0 s、md5 SAME、errors 0)。**コードは
# 依頼の深さだけ**、値は 1 ビットも変えない (md5 は `c2000_cold` / `c8000_cold` が
# 全 COND で SAME)。group 64 と併合 + 4 MiB は t570c で負けた。

branch: **br570** = 開始時の int-auto **99d3858 (merge: br562)** から派生。
br566 は int-auto に入っていなかったが、**REPORT だけの commit (886cead) を
cherry-pick して履歴を残した** (コードの差は 0)。終端の int-auto 再 fetch は §10。

## 0. 結論 (8 行)

1. **採った (`ADOPT: yes t566a t570b t570c`)。** UD-Q4_K_XL の prefill の壁は
   79% がディスクで、残っていた唯一の次元は**同時に何グループを流すか**だった。
   上げスレッド 2 -> 3 + 先読み 2 で `rd_max` が 64 -> 82-84、平均同時が
   19-22 -> 28-32 本になり、**3,826 cold 6,765-7,339 -> 6,347-6,600 ms**、
   **14k cold 11,370-11,440 -> 10,860 ms** (lab t566a、同じ run の対)。
2. **既定の lookahead は 1 のまま**にした。既定モデル (UD-IQ3_XXS) は P379 が
   同じ深さ (`DENNO_PF_STREAM_LOOKAHEAD=2`) を **3,826 +2.4% / 14k +3%** と
   棄却している (写しが DRAM 21 GB/s で、深くすると `pf_slot_prepare` の再利用待ちが
   増えるだけ)。**pread の立つグループだけ深くする新ゲート
   `DENNO_PF_STREAM_LOOKAHEAD_PREAD` (既定 2)** を足したので、既定モデルの経路は
   P379 の設定そのまま = **構造的に不変**。
3. `DENNO_PF_UP_WORKERS` の既定を 2 -> 3 にした。非 pread の層は
   **ワーカー 0 にしか送らない**実装 (`PfUp::submit`) なので、これも既定モデルでは
   不活性。lab t570b (既定モデル、depth) で decode 32.3 -> 32.4 ms/tok、
   prefill 28.98 -> 29.02 s、md5 全 COND SAME、errors 0 を確認した。
4. **group 64 は勝てない** (t570c `g64d` 3,826 7.03 対 柵 6.75-6.92 = +1.6..+4%、
   14k は 10.31 対 b2 10.45 だが b 11.68 で、**同じ run の位置ドリフトが 10%** あり
   判定不能)。**併合 + 4 MiB は明確に負け** (t570c `f4d3` 3,826 8.88 = +28%、
   14k 12.68 = +8.6%)。P550 の棄却は構造が変わっても成立した。
5. 3,826 の壁の 87% は今も experts = ステージングの写し待ちで、その写しの壁の
   79% がディスク (0.777 MB x 平均同時 19-22 本 = 3.11-3.20 GB/s、t522 のプローブ
   天井 3.5 の 89-91%)。**深さはその 9% の側から取った** (§2)。
6. `mincore` の常駐判定 (287-294 ms/リクエスト) は削らなかった: 同じ CPU の
   マイクロベンチで**コストがページ数に比例**し (0.78 MB で 3.06 us、範囲を
   まとめても隙間のページを走査するので安くならない)、2 本の上げスレッドの裏に
   隠れる (§5)。
7. 値は 1 バイトも変えない: 同じバイトを同じ宛先へ、**依頼の深さだけ**を変える。
   md5 は t566a / t570c の全 COND で `c2000_cold` = `73510e1f`、`c8000_cold` =
   `47c833e2` が SAME。短いプロンプトの `b:` の DIFF は P490/P526/P534/P546/P550 が
   記録した**既存の decode の揺れ** (同じ COND の r1 と r2 でも違う)。
8. **確認 run (t570d) も走った**: 既定のまま `rd_max` が 84-92、切り戻しで 62-63
   = **配線の発火を確認**。同じ run の b はディスクの外れ回を引いて柵に使えず
   (壁 9.8 s / 17.3 s)、clean な b2 と r の対では **-3.4% (3,826) / -8.3% (14k)**
   で向きは t566a と同じ (§3.4)。

## 1. fresh な段プロファイル (Do 1 / Do 2)

### 1.1 使った実測と、それが現 int-auto を代表する根拠

- **t562b** (= SHA `e49eec9`、2026-11-22 20:33、UD-Q4_K_XL、depth8k、PROFILE=1、
  COND `x0 b p b2`)。`git diff --stat e49eec9 99d3858` は `cuda.rs` 33 行
  (`DENNO_SEG_GU16` の既定 on 化 = decode 専用) と REPORT だけで、
  **`exp_pf.rs` は 1 行も変わっていない** = prefill の staging 経路は同一。
  この run の COND には `DENNO_SEG_PROF=1` が付くが、これは decode の
  `moe_gemv_seg_run` の区間計測だけで prefill の段には入らない。
- 段の値 (ms)。3,826 は 2 ラウンド (b の r1 / r2)、14k は r1 の 1 本:

| 段 | 3,826 r1 / r2 | share | 14k | share |
| --- | ---: | ---: | ---: | ---: |
| **experts** | **6,425.9 / 5,662.6** | **87.0 / 86.9%** | **8,158.9** | **70.2%** |
| ssm | 387.8 / 357.3 | 5.3 / 5.5% | 1,898.1 | 16.3% |
| mix | 285.1 / 281.4 | 3.9 / 4.3% | 26.5 (busy 658.4) | 0.2% |
| other | 241.5 / 170.6 | 3.3 / 2.6% | 853.2 | 7.3% |
| full | 41.7 / 47.9 | 0.6 / 0.7% | 683.1 | 5.9% |
| **wall** | **7,382.0 / 6,519.9** | | **11,619.7** | |

  experts の中身: **exposed 5,993.2 / 5,228.9 ms = 写し待ち**、GPU busy 432.7 /
  433.7 ms (壁の 6%)。14k の other は `iter_top` 366.4 / `iter_ple` 270.0 /
  `pf_er` 242.8 / `pf_teardown` 125.3 / `res_up` 67.7。

### 1.2 なぜこの段・この次元か

- 段で一番大きいのは **experts (3,826 で 87%、14k で 70%)**、その中身は写し待ち。
  3,826 の他の段は 5.5% 以下なので 5% の目標はここ以外に置けない。14k の 2 番目
  (ssm 1,898 ms) は **GPU busy が 1,319 ms** で `.cu` の変更が要る (本タスクで
  触っていない、§9)。
- 読むバイト数は削れない (`up_MB` 43.5 GB は必要な重み全部、区間に重複なし =
  P566 §1.3)。VRAM 常駐を増やす手は P435/P452 が棄却済み。
- 読みの形 (大きさ・併合・グループ) は P550 が 3 方向とも棄却していたが、
  **そのときの構造は「上げスレッド 1 本 + 共有 pool 32 = 平均同時 12 本」**。
  その後 P554 (2 本) と P558 (スレッドごとの pool、rd_max 32 -> 62) で構造が
  変わっていた。**変わった点を名指しして**深さの方向を測り直したのが本タスク。

## 2. ディスクを台帳から割り直した (3,826 cold、t566a の b)

| 項目 | b r1 | b r2 | 14k |
| --- | ---: | ---: | ---: |
| `copy_s` (積算) | 11.62 s | 10.54 s | 13.29 s |
| `odb_MB` / `odb_n` (O_DIRECT = page cache に無い分) | 19,355.1 / 24,856 | 16,331.4 / 21,045 | 21,782.3 / 27,589 |
| `odb_ms` (スレッドの積算) | 127,409 | 102,181 | 146,448 |
| `rd_max` | 64 | 64 | 64 |

割り算 (上の実測からの計算):

- **写しの壁** = `copy_s / 2` = 5.81 / 5.27 s = 壁の **79%**。
- 実効 = 18.7-19.4 GB / 5.3-5.8 s = **3.2-3.3 GB/s**、1 本 0.777 MB を 4.9-5.2 ms。
- **平均同時** = `odb_ms / (copy_s/2)` = **19-22 本** (rd_max は 64 = 枠は余っている)。
- t522 のプローブ (O_DIRECT、1 MiB x 8 本、連続窓) は 3.5 GB/s。今の 0.777 MB の
  天井 3.5 に対して **89-91%**。つまり残りは 9% で、-5% はその 6 割を取る勝負。

## 3. ラボの結果

### 3.1 t566a — 深さの 3 方向 (SHA `99d3858` = int-auto、depth8k、PROFILE=1、load 1.64)

COND `x0 b k3 d3 k3d3 b2`。k3 = `DENNO_PF_PREAD_KB=384`、d3 = `LOOKAHEAD=2
UP_WORKERS=3`、k3d3 = 両方。

| COND | 3,826 cold (r1 / r2) | median | 14k cold | decode | `rd_max` | 平均同時 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| x0 (捨て) | 8,320 / — | — | 10,500 | 13.3 | — | — |
| **b** | 7,339.4 / 6,765.4 | 7,052 | 11,370 | 12.9 | 64 | 19-22 |
| k3 | 8,093.7 / 7,337.1 | 7,715 | 10,920 | 13.2 | 64 | 32-36 |
| **d3** | **6,599.8 / 6,346.6** | **6,473** | **10,860** | 12.9 | **82-84** | **28-32** |
| k3d3 | 7,373.6 / 7,036.4 | 7,205 | 11,060 | 12.9 | 96 | 40+ |
| b2 (末尾) | 6,729.5 / 15,807.4 | 11,268 | 11,440 | 12.9 | 62-64 | 16-20 |

- **d3 が両方の柵より速い**。3,826 は b の中央値 7,052 に対して **-8.2%**、
  末尾 b2 の良い方のラウンド 6,730 に対しても **-3.8%**。14k は b 11,370 に
  対して **-4.5%**、b2 11,440 に対して **-5.1%**。
- **k3 (384 KiB) は深さは増えるが 1 本の効率が落ちて負け** (`pread_calls`
  31-35k -> 81-85k、平均同時 32-36、実効 2.7-2.8 GB/s)。k3d3 も同じ向きで負け。
- b2 の 2 ラウンド目が 15,807 ms に跳ねた (`copy_s` 28.35 = 2.7 倍) のは
  **ディスクの外れ回**で、同じ環境のこの run の中でも起きる (P550 §0-6 と同じ)。
- md5 は**全 COND で** `c2000_cold` = `73510e1f`、`c8000_cold` = `47c833e2`。

### 3.2 t570c — group 64 と併合 + 4 MiB (SHA `26cdeca`、同じモデル/scenario)

COND `x0 b g64d f4d3 b2`。g64d = `GROUP=64 LOOKAHEAD=2 UP_WORKERS=3`、
f4d3 = `PIN_FILE_ORDER=1 PREAD_MERGE=1 PREAD_KB=4096` + 同じ深さ。

| COND | 3,826 cold | 14k cold | decode |
| --- | ---: | ---: | ---: |
| b | 6.92 s | 11.68 s | 12.8 |
| **g64d** | 7.03 s (+1.6..+4%) | 10.31 s (b2 10.45 と同帯 = 判定不能) | 13.0 |
| **f4d3** | **8.88 s (+28%)** | **12.68 s (+8.6%)** | 12.8 |
| b2 | 6.75 s | 10.45 s | 13.0 |

- **group 64 は勝てない**: 3,826 で柵 (6.75-6.92) より遅く、14k の見かけの -11%
  は同じ run の b (11.68) と b2 (10.45) の **10% の位置ドリフト**の内側。
- **併合 + 4 MiB は明確に負け** (P550 の棄却が構造を変えても成立した)。
- md5 は全 COND で `c2000_cold` / `c8000_cold` SAME。

### 3.3 t570b — 既定モデル (depth、`model=flash-next-UD-IQ3_XXS`)

COND `x0 b p b2`、p = `LOOKAHEAD=2 UP_WORKERS=3`。

| COND | decode ms/tok | prefill 3,826 (c2000_cold) | md5 |
| --- | ---: | ---: | --- |
| x0 | 32.3 | 29.47 s | `b42d79d2` |
| b | 32.3 | 28.98 s | SAME |
| p | 32.4 | 29.02 s | SAME |
| b2 | 32.4 | 29.37 s | SAME |

- **既定モデルは悪化しない** (decode +0.1 ms、prefill +0.14%、md5 SAME、errors 0)。
- ただしこの run の既定モデルは **cold な regime** (decode 32.3 ms/tok。温かい
  ときは 10.4)。P379 の +2.4% は温かい regime の数字なので、**この run では
  その regime を覆っていない**。覆いは §4 の「pread のときだけ深くする」で
  **構造的に**取った (非 pread の層は 1 グループ先のまま = P379 の設定そのもの)。

### 3.4 t570d — 既定 on の確認 (SHA `85f28f4`、同じモデル/scenario、load 3.54)

COND `x0 b r b2`、**b が新しい既定** (UP_WORKERS=3 + LOOKAHEAD_PREAD=2)、
r が切り戻し (`DENNO_PF_STREAM_LOOKAHEAD_PREAD=0 DENNO_PF_UP_WORKERS=2`)。

| COND | 3,826 r1 / r2 | 14k | decode | `rd_max` |
| --- | --- | ---: | ---: | ---: |
| b (新既定) | 7,310.4 / **9,787.2** | **17,334.9** | 12.9 | **88-92** |
| r (切り戻し) | 6,884.9 / 6,624.1 | 11,223.3 | 12.9 | **62-63** |
| b2 (新既定) | 6,658.9 / 6,384.1 | **10,291.8** | 13.0 | 84-90 |

- **配線は発火した**: 既定のまま `rd_max` が 84-92、切り戻しで 62-63 に戻る。
  これは依頼の深さが実際に変わった証拠 (緑ではなく台帳で数えた)。
- **この run の b はディスクの外れ回を引いた** (`copy_s` 23.78 / 25.93 = 正常の
  1.6-2 倍、壁 9.8 s / 17.3 s)。柵に使えるのは b2 (6,521 / 10,292) と r
  (6,754 / 11,223) で、**b2 が r より -3.4% (3,826) / -8.3% (14k)**、向きは
  t566a と同じ。**絶対値の再現とは言えない** (run のノイズが大きすぎる) が、
  「既定 on で t566a 以前より遅くなっていない」ことは言える。
- md5 は全 COND で `c2000_cold` = `73510e1f`、`c8000_cold` = `47c833e2` SAME。

## 4. 変えたこと (コード)

`crates/denno-engine/src/model/exp_pf.rs`:

- **`DENNO_PF_STREAM_LOOKAHEAD_PREAD` (新規、既定 2)**: pread の立つグループだけ
  何グループ先まで依頼するか。`0` で base と同じ (= 切り戻し)。
  純関数 `stream_lookahead_pread_from_value` / `lookahead_for(pread, base, deep)`。
- **`DEFAULT_STREAM_LOOKAHEAD` は 1 のまま** (既定モデルの経路を P379 の設定から
  動かさないため)。
- **`DEFAULT_UP_WORKERS` を 2 -> 3**。非 pread の層はワーカー 0 にしか送らないので
  既定モデルでは不活性。`DENNO_PF_UP_WORKERS=2` で切り戻し。

`crates/denno-engine/src/cuda.rs`:

- `pf_submit_group` が「そのグループが pread を使ったか」も返すようにし、依頼ループが
  `look = exp_pf::lookahead_for(pread, look_base, look_deep)` で深さを更新する。
  最初の 1 グループは base (1) で依頼し、pread と分かった時点で同じ反復のうちに
  2 まで伸びる = UD では `LOOKAHEAD=2` と同一の依頼順、既定モデルでは従来と同一。

テスト:

- lib 追加 `cuda::pf_stream_tests::lookahead_pread_gate_and_effective_depth`
  (env の解釈 8 行 + 実効深さ 6 行。**既定モデル側が 1 のまま**であることを固定)。
- `up_workers_values_round_and_default` の既定 2 -> 3 を更新。
- kernel_invariants `task320_upload_thread_lookahead_is_wired` を新しい配線に合わせて
  更新 (`look_base` / `lookahead_for(pread, ...)` / 新 env 名 / 純関数の存在を固定)。
  **緑は分岐が発火した証拠ではない**ので、発火は lab の `rd_max` 64 -> 82-84 と
  平均同時 19-22 -> 28-32 本で数える (§3.1)。

`.cu` / `.ptx` は触っていない (PTX 再生成不要)。

## 5. やらなかったこと (測って棄却した 1 件)

- **`mincore` の常駐判定を削る案**: `nc_ms` は 287.6 / 293.5 ms (3,826)、150.4 ms
  (14k)。同じ CPU (9800X3D) のマイクロベンチでは **0.78 MB (192 ページ) で
  3.06 us/回 = 54.6k 回で 167 ms** で、コストは**ページ数に比例**していた
  (4 MiB で 13.94 us、16 MiB で 54.95 us、1 ページで 0.22 us)。つまり範囲を
  まとめても隙間のページを走査するので安くならない。安くするには走査バイトを
  減らす近似になるが、判定は**上げスレッドの pool 待ちの裏に隠れる**ので壁に
  効く保証が無い。**コードを足さず棄却** (マイクロベンチは相対値としてのみ扱う)。

## 6. 数値の扱い

- §1・§2・§3 の段・壁・台帳は**すべて GPU ラボの実測**。割り算 (§2 の GB/s・
  同時本数) と「推定」と書いた見積りだけが計算値。
- **run を跨いだ比較はしていない**。t566a の b (7,052 / 11,370) と t570c の b
  (6.92 / 11.68) は同帯で、run 内の柵 (b, b2) に対してだけ判定した。
- 同じ run の中の位置ドリフトは **-11.7% (t562b の b と b2)**、**+135%
  (t566a の b2 r2、ディスクの外れ回)**、**-10.5% (t570c の 14k の b と b2)** と
  大きい。**単発の 2-3% は採らない**。d3 は 3,826 で 2 サンプルとも柵の最良値より
  速い、14k は柵 2 本 (11,370 / 11,440) より速い、という形で判定した。
- decode は **12.9 ms/tok で b と同じ** (0.1 ms/tok 刻み)。短いプロンプトの md5 は
  同じ COND の r1 と r2 でも違う (既存の decode の揺れ) ので採否に使っていない。
  長い 2 文脈の md5 は**全 COND・全 run で SAME**。
- 目標 (3,826 か 14k の cold prefill -5%) に対しては **3,826 が -3.8..-8.2%、
  14k が -4.5..-5.1% で、単独では 5% に届いていない**。オペレーター規則
  (段で 2 ラウンド 2% 以上・他が 1% 超悪化しない・本文健全) で既定 on にした。

## 7. 本文 (健全性)

- t566a の `d3` 14k (`d3_1_c8000_cold.json`) の先頭:

```text
prefillの律速は計算量ではなく、線形層ごとのホスト往復と戻り値コピーであった。
pinned経由の二度手間を...
```

- t566a の `d3` 短プロンプト: `# B-Trees: A Comprehensive Explanation ... A
  **B-tree (Balanced Tree) is a self-balancing tree data structure that maintains
  sorted data and allows searches, sequential access, insertions, and deletions
  in logarithmic time.` (b と同文)
- errors 0 は t566a / t570b / t570c の全 COND・全ラウンド。RESULT.md の text 欄も
  全 COND "looks ok"。

## 8. テスト

```text
CARGO_TARGET_DIR=/tmp/t570    cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1
  -> 1413 passed; 0 failed; 103 ignored
CARGO_TARGET_DIR=/tmp/t570    cargo test --release -p denno-engine --features cuda --test kernel_invariants -- --test-threads=1
  -> 153 passed; 0 failed
CARGO_TARGET_DIR=/tmp/t570nc  cargo build --workspace --release      -> ok
CARGO_TARGET_DIR=/tmp/t570nc  cargo test  --workspace --release -- --test-threads=1
  -> passed=1553 failed=0 (cuda 無し。全バイナリの `test result:` の合計)
```

- 追加した 2 本 (lib) は単体で走らせて発火を確認した:
  `lookahead_pread_gate_and_effective_depth` / `up_workers_values_round_and_default`
  -> 2 passed。
- 既知の赤 (`qwen4exp_decode spec_batch_stops_at_eos`、`qwen4exp_params` の
  validate 2 件) は触っていない。

## 9. 数値の方針メモ

- 丸め順も計算も変えていない。変えたのは**依頼の順番と深さ**だけ。
- 切り戻しは 2 つ: `DENNO_PF_STREAM_LOOKAHEAD_PREAD=0` と `DENNO_PF_UP_WORKERS=2`
  (両方で t566a 以前と同じ 1 グループ / 2 本)。

## 10. GPU で確認してほしいこと

1. **t570d は走った** (§3.4)。配線 (rd_max 84-92 / 切り戻し 62-63) と md5 SAME は
   取れている。**残っているのは「きれいな柵での絶対値の再現」だけ**で、t570d の b が
   ディスクの外れ回を引いたため取れていない。次に 1 本流すなら
   `SCENARIO=depth8k`、`COND=x0 b r b2` (b = 新既定、r = 切り戻し) を
   **load の低い時間帯**に置く。3,826 が 6.4-6.6 s、14k が 10.3-10.9 s、
   `rd_max` 84-92 が期待値。
2. **既定モデルの温かい regime** を 1 ラウンド (`SCENARIO=depth`、`COND=x0 b p b2`、
   p = `DENNO_PF_STREAM_LOOKAHEAD_PREAD=2`) で見るなら **UD の run より前**に置くこと
   (P562 §5-2 のページキャッシュの順序の罠。t570b は UD の後に走って
   decode 32.3 ms/tok の cold regime だった)。ただし既定モデルは非 pread の層が
   1 グループ先のままなので、**構造的には P379 の設定から動いていない**。
3. もし `rd_max` が 64 のまま (新既定で 84 前後にならない) なら**配線が効いていない**
   ので、`lookahead_for` の呼び出しと `plan.pread` の真偽をログで確認すること。

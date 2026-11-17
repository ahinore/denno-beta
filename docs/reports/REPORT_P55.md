# REPORT P55 — GDN pass 1 のローカル往復を消す (設計 + 実装)

作業場所: branch `p55` / 親 commit `d4ee7d90`
担当: dev2 (実装・CPU テスト・commit)。**GPU 実測・PTX 再生成・push はしていない。**

---

## 1. 前提 (与えられた実測値・再測定していない)

| 項目 | 値 | 出所 |
|---|---|---|
| GDN pass 1 のローカル往復 | 推定 265 GB/層・実測 138 ms/層 (27B q4・n=14,043・48 層平均) | `docs/history/50_gdn.md:58` (P15) |
| pass 1 の律速 | ローカルメモリの**容量ではなくアクセス命令数** | `docs/history/50_gdn.md:310-312` (s192) |
| 容量を削る 2 案 | (a) 配列 128 要素化 = 変化なし (9.75 → 9.77 s) / (b) W レジスタ・M shmem = +13% | 同上 `:451-452` |
| 4 レーン版 (P36) | 9.69 → 9.88 s。1 block/SM まで occupancy が落ちて利得が消えた | `docs/history/50_gdn.md:352-354` |
| 未着手の候補 | 状態配列を shmem へ / pass 1 を W 専用・M 専用に分割。見込み 138 → 10〜20 ms/層 | `docs/history/50_gdn.md:306`, `:435` |

GDN の形 (27B): nk=16, nv=48, dk=dv=128, d_conv=4 (`crates/denno-engine/tests/kernel_invariants.rs:816-819` の実形状表)。

---

## 2. 現状のコード (読んだ結果)

`crates/denno-engine/kernels/gdn.cu:215-329` `dno_gdn_chunk_pre`:

- 起動構成: grid=(n_chunks, nv), block=(dk) = 128 スレッド。スレッド c が W/M の**列 c** を担当。
- `float Wcol[256]; float Mcol[256];` (`:242-243`) — dk が実行時引数なので固定長配列 = ローカルメモリ。
  dk=128 で 1,024 B × 2 = 2,080 B/スレッド (スタックフレーム実測値は s192 の 2,080 B と一致)。
- 1 ブロックの作業集合 = 2·dk·dk·4 B = **128 KiB**。L1 (128〜256 KiB) に収まらないため、
  毎トークンの読み書きが L2/DRAM へ落ちる。これが 138 ms/層 の正体 (P15 の結論)。
- 共有メモリは k/v + スカラー 3 語の **1,036 B だけ** (`:236-239`, ホスト `cuda.rs:14400`)。
- 1 トークンあたり `__syncthreads()` は 4 回 (`:264,274,289,291`)。

ホスト側: `crates/denno-engine/src/cuda.rs:14400` `shmem_pre = (2*dk+3)*4`、
launch は `cuda.rs:14408-14421`。pass 2/3 の shmem は別式 (`:14507`)。

---

## 3. 案 A / 案 B の比較 (算術)

### 3.1 ローカルアクセス命令数

スレッド 1 本・1 トークンあたり、1 配列 (dk=128 要素) について:

| ループ | 読み | 書き | 計 |
|---|---|---|---|
| loop1 (s241 融合: decay + 内積) | 1 (`*= eg` の値) ＋ 1 (`skv +=` の再読) | 1 | 2〜3 |
| loop2 (更新) | 1 | 1 | 2 |
| 合計/配列 | | | 4〜5 |
| W+M 合計/要素 i | | | **8〜10** |

- 8 アクセス/i とすると 1 層 = n·nv·dk·(8·4 B) = 14,043·48·128·32 B = **353 GB/層**。
- 6 アクセス/i とすると **265 GB/層** = 資料の推定値と一致する (`docs/history/50_gdn.md:58`)。
  差は `*= / +=` の read-modify-write を 1 命令と数えるか 2 と数えるかだけで、
  138 ms に対しては 1.9〜2.6 TB/s を要求する = **DRAM 帯域律速**の結論はどちらでも変わらない。
- 以下では資料の 265 GB/層 を前提に使い、353 GB を上限として併記する。

| | 案 A (状態配列を shmem へ) | 案 B (W 専用 / M 専用に 2 分割) |
|---|---|---|
| ローカルアクセス命令 | **0**(W/M は shmem) | **変化なし** (合計 8〜10/i のまま。1 カーネルあたり 4〜5/i を 2 回) |
| アクセス先 | 共有メモリ (SM 内で閉じる) | ローカルメモリ (L1/L2/DRAM) のまま |
| 1 スレッドのローカル量 | 0 B (推定) | 1,024 B/カーネル (s192 (a) と同じ 1,056 B スタック) |
| 1 ブロックの作業集合 | 128 KiB (shmem) | 128 KiB (ローカル) → 2 カーネルで各 64 KiB |
| 必要 shmem/ブロック | (2·dk²+2·dk+3)·4 = **132,108 B** (dk=128) | 1,036 B (不変) |
| occupancy (block/SM) | **1**(132 KB が SM の smem を占有) | 不変 (≈12 block/SM = 1,536 スレッド) |
| launch/層 | 3 (不変) | **4** (+1) |
| sync/層 | 0 (デバイス出力) / 1 (ホスト出力) — 不変 | 同左 (不変) |
| 追加の global 書き込み | なし (w_out/m_out は不変) | なし (同じ 2 本を別カーネルが書く) |

### 3.2 occupancy の見積り (P36 の教訓を必ず入れる)

- 現行 pass 1: 128 スレッド/block・約 40 regs (s243) → スレッド上限 16 block、レジスタ上限 12 block。
  **約 12 block/SM = 1,536 スレッド (75%)**。ローカルメモリは occupancy を制限しない (L1 を圧迫するだけ)。
- 案 A: 132,108 B/block。A100 (opt-in 164 KiB/SM) も H100 (227 KiB/SM) も **1 block/SM = 128 スレッド = 6.25%**。
  P36 が 1 block/SM (512 スレッド) で失敗したのと同じ形の賭けである。**ただし P36 が失ったのは
  「レジスタ常駐でローカルアクセスを 1/4 にしたのに、レイテンシ隠蔽が足りなくなった」分**で、
  案 A はアクセス先そのものが DRAM → shmem に変わる (帯域が 1 桁違う)。
- 案 B: occupancy は不変。しかし**作業集合の半減は s192 (a) で実測済み** (2,080 → 1,056 B で 9.75 → 9.77 s)。
  同じ効果を狙う案なので、**既に測って効かなかった案の再試行**になる。

### 3.3 見込み時間 (推定)

- 案 A: 共有メモリ帯域 = 128 B/clk/SM × 108 SM × 1.41 GHz ≈ **19.5 TB/s** (A100・推定)。
  265 GB/層 ÷ 19.5 TB/s ≈ **13.6 ms/層** (353 GB なら 18 ms/層)。
  1 block/SM でも 4 ワープ × unroll 8 = 32 本の独立ロードが飛び、共有メモリのレイテンシ ≈25 clk を
  1 命令/clk で埋めるのに必要な ≈25 本をぎりぎり上回る (推定)。**13〜20 ms/層 (推定)**。
  資料の見込み 10〜20 ms/層 と同じ桁。
- 案 B: **利得 ≈0** (s192 (a) の実測が根拠)。コストは +1 launch/層 = +48 launch/req
  (P15 の 15 µs/層・3 launch から 5 µs/launch とすると +0.24 ms/req・推定)。

### 3.4 結論

**案 A を実装**。案 B は「s192 (a) で同じ作業集合の半減を実測して効果なし」という既存の反証があり、
launch だけ増える。案 A は 138 ms/層 の原因 (DRAM 往復) を直接消す唯一の案で、
数値もビット同一に保てる (格納先だけの変更)。

---

## 4. 実装したもの

### 4.1 カーネル (新規・既定 off)

`crates/denno-engine/kernels/gdn.cu:358-471` `dno_gdn_chunk_pre_sm`

- 引数は旧 `dno_gdn_chunk_pre` と**完全に同一** (ホストの args 配列を共有する)。
- 共有メモリのレイアウト (`:379-384`): `w_sh[dk][dk] | m_sh[dk][dk] | k_sh[dk] | v_sh[dk] | scalars[3]`。
  列優先 (`i*dk + c`) なので固定 i でスレッド c が連続アドレスを踏みバンク衝突しない。
- 各スレッドは自分の列しか読み書きしない → 共有メモリ化しても依存関係は増えない。
  初期化 (`:389-392`) はループ先頭の `__syncthreads()` で足りるので **同期回数は 4 回/トークンのまま**。
- 演算列 (decay → 内積 → 更新、`#pragma unroll 8`) は旧版と同一 (`:443-462`)。

### 4.2 ホスト側

- `cuda.rs:545` フィールド `f_gdn_chunk_pre_sm: Option<CUfunction>` と `gdn_pre_sm_dyn_smem: usize`。
- `cuda.rs:822-842` モジュールロード時に `cuModuleGetFunction` → 成功時のみ
  `cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES, デバイスの opt-in 上限)`。
  失敗・カーネル不在は `None` / 48 KiB 既定のまま。
- `cuda.rs:3764-3816` `gdn_pre_sm_enabled()` (`DENNO_GDN_PRE_SM=1`)、`gdn_pre_sm_smem_bytes(dk)`、
  `gdn_pre_sm_gate(env, has_fn, dk, dv, smem, limit)`。
- `cuda.rs:14400-14425` launch の選択を「カーネル・block・shmem」の 3 つ組にした
  (shmem だけ旧式が残ると共有領域外書きになる = s99 と同じ型の欠陥を避ける)。
  優先順位: `DENNO_GDN_LANES=4` (P36) > `DENNO_GDN_PRE_SM=1` > 旧カーネル。
- `cuda.rs:2593-2600` `dbg_kernel_name` に新カーネルを追加 (計測器の名前解決)。

### 4.3 既定 off にした理由

数値はビット同一なので「既定 on でも安全」ではあるが、occupancy の影響が実機で未測定なので
P36 (4 レーン版) と同じ扱い (opt-in・既定 off・A/B の入口だけ用意) にした。
実機で勝ちが確認できたら既定を on にする (定数 1 行)。

---

## 5. 数値不変の根拠

1. 変わるのは W/M の**格納先だけ** (ローカルメモリ → ブロック共有メモリ)。値はどちらも f32。
2. 読み書きの**順序・回数・丸めは同一**。共有メモリはブロック内で完結し、各スレッドは
   自分の列 (`i*dk + c`) しか触らないため、他スレッド・他ブロックとの依存が増えない。
3. 同期回数は 4 回/トークンで不変 (初期化はループ先頭の既存 sync に相乗り)。
4. 上記 2 を静的に固定するテストを足した: `kernel_invariants.rs:719`
   `gdn_pre_sm_keeps_step_order_of_local_version` — 両カーネルの本体から演算列
   (decay / 内積 / 更新) を抽出し、ストレージ式を正規化して**列が完全一致**することを検査する。
5. wide md5 を動かす経路は無い (pass 1 の出力 = w_out/m_out の値が同一)。
   ただし**実機アンカーでの確認はしていない** (この環境に GPU が無い)。

---

## 6. 見積り (推定・`ptxas -v` は使っていない)

| 資源 | 旧 `dno_gdn_chunk_pre` | 新 `dno_gdn_chunk_pre_sm` |
|---|---|---|
| 動的 shmem/block | 1,036 B | 132,108 B (dk=128) |
| スタックフレーム/スレッド | 2,080 B (s192 実測) | **0 B (推定)** — Wcol/Mcol が消える。残るのは wk/wv/ink/inv (14 float) とスカラー |
| レジスタ | 約 40 (s243 実測) | 推定 32〜40。列配列のアドレス計算が減る分だけ下がる見込み (未測定) |
| block/SM | 約 12 (推定) | 1 (132 KB/block が SM の smem を占有) |
| スレッド/SM | 約 1,536 (75%) | 128 (6.25%) |
| 共有メモリ帯域の要求 | — | 265 GB/層 ÷ 13〜20 ms ≈ 13〜20 TB/s (デバイス全体で ≈19.5 TB/s が上限・推定) |

`ptxas -v` の実値 (regs / smem / stack) はメインが PTX 再生成時に取ること。

---

## 7. CPU テストの結果 (実行した)

| コマンド | 結果 | テスト本体の時間 |
|---|---|---|
| `cargo test -p denno-engine --lib` | **293 passed / 0 failed / 7 ignored** | 1.07 s |
| `cargo test -p denno-engine --features cuda --lib` | **345 passed / 0 failed / 42 ignored** (+2: `p55_gdn_pre_sm_tests`) | 1.05 s |
| `cargo test -p denno-engine --test kernel_invariants` | **23 passed / 0 failed** (+1: `gdn_pre_sm_keeps_step_order_of_local_version`) | 0.02 s |
| `cargo build --release` | 成功 | — |
| `cargo build --release --features cuda -p denno-engine` | 成功 (新コードに警告なし) | 3.53 s |

追加・更新したテスト:

- `cuda.rs:15356` `mod p55_gdn_pre_sm_tests` — 必要 smem が 48 KiB 既定を超え A100/H100 の
  opt-in 上限には収まること、ゲートの真理値表 (env off / カーネル不在 / opt-in 失敗 /
  dk 過大 / dk != dv / dk=0 のどれでも false、dk=64 は 48 KiB で true)。
- `kernel_invariants.rs:532` `gdn_kernel_shared_layout_matches_host_smem` — 新カーネルの
  共有レイアウト 5 宣言と、ホストの `(2 * dk * dk + 2 * dk + 3) * 4` を突き合わせる形に拡張。
  `sc[0]=k_inv sc[1]=eg sc[2]=beta` の本数は 2 → 3 に更新 (カーネルが 1 本増えただけ。
  割り当ては 3 本とも同じ)。**テストの緩和はしていない** (期待値の更新のみ)。
- `kernel_invariants.rs:719` 演算列の静的比較 (上記 5.4)。

---

## 8. メインが実機でやる必要があること

1. **PTX 再生成** (この環境に nvcc/ptxas は無い)。従来の手順:
   `nvcc -ptx -arch=compute_75 -fmad=false -O3 crates/denno-engine/kernels/gdn.cu -o crates/denno-engine/kernels/kernels_gdn.ptx`
   先に「変更前ソースの再コンパイルがコミット済み PTX と sha256 一致する」ことを確認する規約
   (`docs/history/50_gdn.md:108-110` の手順) を踏むこと。
   **再生成するまで GPU 経路は旧カーネルのまま** (`module_get_function` 失敗 → None → ゲートで旧版)。
2. `ptxas -v` で新カーネルの regs / smem / stack frame を取る (本 REPORT の推定値の答え合わせ)。
3. `GPU_VERIFY` / アンカー 3 本 (v6_1b / v6_2b / v6_1b_full) + **wide md5** が不変であること。
   数値はビット同一のはず (変わったら実装ミス)。
4. **cold A/B**: `DENNO_GDN_PRE_SM=0` (既定) と `=1` を交互 3 ラウンド以上、27B q4・16k と 65k。
   見込みは 138 → 13〜20 ms/層 (推定)。受入は「16k で 2 桁 ms 減」または「GDN pre が半減」。
5. `[GDN_KPROF]` の腕: `DENNO_PROFILE=1 DENNO_GDN_KPROF=1` で `pre=` の ms を新旧で比べる
   (P40/P43 の計測器。P45 後も残っている)。`miss=` が増えていないことも見る。
6. opt-in の可否を確認: 起動ログに新カーネルが出るか (`dbg_kernel_name` = `dno_gdn_chunk_pre_sm`)。
   opt-in が失敗するデバイスでは静かに旧カーネルへ落ちる (落ちたことが分かるのは launch 名だけ)。
7. 勝ちが確認できたら既定 on にする (`gdn_pre_sm_enabled()` の既定値を true に変える 1 行 + テスト更新)。

---

## 9. 未検証のこと

- 実機の速度 (13〜20 ms/層 は**推定**。共有メモリ帯域の見積りと「1 block/SM でも 1 命令/clk を
  出せる」という仮定の上に乗っている)。P36 と同じ形の賭けなので、負けたら opt-in のまま残す。
- 1 block/SM での実効 occupancy と、`n_chunks × nv` ブロックが SM を埋めきれるか
  (27B・65k で 32 × 48 = 1,536 ブロック / 108 SM = 14 ウェーブ。テールは小さい見込み・推定)。
- `ptxas` の実値 (regs / stack frame / smem)。本 REPORT の 0 B / 32〜40 regs は推定。
- dk > 128 のモデル (0.8B 等) でこのカーネルが選ばれないこと — ゲートで落ちる実装だが、
  実機で dk>128 のモデルを回していない。
- 案 A と 案 B の併用 (shmem + 2 分割) は試していない。A が occupancy で負けた場合の次の一手は
  「M だけ shmem・W はローカル」(64 KiB + 1,036 B → 2 block/SM) で、これは s192 (b) の逆配置
  (あちらは W レジスタ・M shmem で 255 regs → 13% 悪化)。

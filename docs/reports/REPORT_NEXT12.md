# REPORT_NEXT12 — 2b (283tok prefill) 出力 2 値分岐の切り分け (第35セッション成果の記録) (2026-10-16)

**完了** — HANDOFF_NEXT.md のタスク (本 REPORT の作成 + commit) を完了。本セッションは実装・調査の継続を行っておらず (HANDOFF 指示どおり)、第35セッションが中断時点までに確定した切り分け結果を詳述する**調査記録**である。コード変更はゼロ。docs/HISTORY は「完了した変更のみ追記」規律により本セッションは追記なし。

## 1. サマリー (結論)

- **確定事実 1**: CPU 経路は 1b/2b とも完全決定論 (1b `feea9444`×2・2b `da0b1f57`×6、いずれも 2994feb 時点の CPU 参照値と同一) → rayon/HashMap 順序/detokenizer 境界等の CPU 側仮説は実験的に棄却。**分岐は GPU 経路固有**
- **確定事実 2**: 1b/2b の GPU prefill 経路差を確定 — 1b (m=14) は `linear_pinned` GEMV + CPU FFN、2b (m=283) は `linear_gemm` (tiled `dno_gemm_i8`) + `maybe_gpu_ffn` (デバイス常駐 FFN)。GDN scan は両者同一 (n < 2048 で逐次 `dno_gdn_scan`)。**2b 固有の GPU コードはこの 2 つのみ**
- **推論 (第35セッション最大の知見)**: 2b の揺れペア (`3a1fbabe`/`d46ab41f`) が 0c94b05 の decode 修正前後で同一 → **分岐は 2b の prefill 内で発生し、decode は prefill 結果から決定論的に続く**。decode chain は 1b も通る共通経路であり、1b が 0c94b05 後 ×4 完全一致に回復している以上 decode が起源になり得ない
- 静的監査では 2b 固有経路の競合点を特定できず (`dno_gemm_i8` 境界ガード・`dno_gdn_scan` 逐次和・per-op sync・0c94b05 事前確保・重みレジストリ・KV ミラー cap いずれも問題無し)。**最終同定は実機 GPU での prefill 層別 checksum 比較に委譲 (人間対応・メインセッション)**

## 2. 背景 (第34セッション時点の現状)

- 0c94b05 (C4 decode chain の grow 競合修正) で 1b は GPU 実機で回復: 同一 payload ×4 md5 `3fb48936f12362019ca8ed221a8cd9ad` 完全一致・28ms/step
- 一方 2b (283 tok prefill を含む payload・max_tokens 64) は **2 値分岐が残存**: md5 が `3a1fbabe` / `d46ab41f` のペアで揺れる (2994feb 時点と同一ペア)。CPU 参照は `da0b1f57`
- 注: GPU 出力が CPU 参照と異なること自体は GDN scan 緩和契約 (bit 一致不要・決定論のみ) 内の既知 drift と temperature 0 greedy の near-tie 不安定性で説明済み (docs/HISTORY §3)。本件の問題は **GPU 上の決定論が 2 値に分岐する点**のみ

## 3. CPU 側再現検証 (第35セッション・実測)

手順:

- release ビルド (`CARGO_TARGET_DIR=/work/denno-beta/target-dev`・denno-cli・26s・既存警告のみ)
- `denno serve local/qwen35-9b-cpu --host 127.0.0.1 --port 8290 --name local/qwen35-9b --seed 42 --quant int8`
- `/v1/messages` に `data/q35_9b_1b.json` / `data/q35_9b_2b.json` を連続 POST・md5 はレスポンス `content[0].text` (規約どおり)

結果:

| payload | 実行数 | md5 (content) | 一致性 |
|---|---|---|---|
| `q35_9b_1b.json` (in=14/out=48) | 2 | `feea94449780dea3685980fbda6e4661` | ×2 完全一致 |
| `q35_9b_2b.json` (in=283/out=64) | 6 | `da0b1f578575e857cba2fc70ccc44fd9` | ×6 完全一致 |

- 出力はいずれも正常な思考トレースで開始 (`Thinking Process:\n\n1.  **Analyze the Req…`・stop_reason=max_tokens)
- **2994feb 時点の CPU 参照値と同一** → GPU 修正 (2994feb/0c94b05) は CPU 経路に影響なし。CPU 経路は 2b でも完全に決定論的で、HANDOFF に列挙された CPU 側仮説 (rayon 非決定性・HashMap 順序・detokenizer 境界) は実験的に棄却

## 4. 1b/2b の GPU 経路差 (静的監査・経路特定)

prefill は `forward()` の n>1 分岐 → `forward_prefill` (`model/qwen35.rs:863`)。投影は `apply_bt_rows` → `maybe_gpu_linear(_grouped)` / `maybe_gpu_ffn`。発動ゲート:

- `max_m()` = 16 (`DENNO_CUDA_MAX_M`): m ≤ 16 は GEMV 経路 (`linear_pinned`)
- `gemm_min_m()` = 256 (`DENNO_CUDA_GEMM_MIN_M`): m ≥ 256 で tiled GEMM 発動
- `maybe_gpu_ffn` も同一ゲート (m > max_m && m ≥ gemm_min_m)

| 経路 | 1b prefill (m=14) | 2b prefill (m=283) |
|---|---|---|
| 投影 (全 Linear) | m ≤ 16 → **`linear_pinned` GEMV** (per-op・呼び出し毎 sync) | m > 16 && m ≥ 256 → **`linear_gemm` (tiled `dno_gemm_i8`)** |
| FFN | ゲート不発 → **CPU FFN** (gate/up Vec + CPU swiglu + down) | **`maybe_gpu_ffn` デバイス常駐** (quantize → gate_up GEMM → `dno_swiglu_int` → down GEMM・中間をホストに落とさない) |
| GDN scan | n=14 < 2048 → 逐次 `dno_gdn_scan` | n=283 < 2048 → 同一 `dno_gdn_scan` (chunked 3-pass は n ≥ `GDN_CHUNK_MIN_TOKENS`=2048 のみ) |
| full attention core | トークン逐次・CPU のみ (`full_attn_core` に cuda 呼び出し無し) | 同一 (共通経路・分岐起源になり得ない) |
| decode (n==1) | `forward_decode_chain` (C4) | 同一 (共通経路) |

→ **2b 固有の GPU コードは `dno_gemm_i8` 経路と `maybe_gpu_ffn` (`dno_swiglu_int` 含む) の 2 つ**。GDN scan・full attention・decode chain は 1b/2b 共通であり、2b のみで揺れる構造はここにしか無い。

## 5. 静的監査結果 (問題無しと確認した領域)

- `dno_gemm_i8` (`kernels_int8.cu`): load/output 両方の境界ガード確認 (`gi < m && gj < n`)。`__syncthreads()` 配置も正常
- `dno_gdn_scan` (`kernels/gdn.cu`): 決定論規約を再確認 — launch 固定 (grid=(nv)・block=(dk))・全逐次和は固定順序 (dk/dv 昇順)・単一スレッドまたはスレッド内完結・atomic/ワープ縮約不使用
- per-op 経路 (`maybe_gpu_linear` / `maybe_gpu_linear_grouped` / `maybe_gpu_ffn`): D2H 後に呼び出し毎 `ctx.sync()` → op 間に in-flight 残留なし (0c94b05 が構造排除したのは decode chain 側のステップ中途 grow のみ)
- decode chain: 0c94b05 の事前確保 (`chain_q35_step_reserve` / `chain_q35_step_reserve_aq` / `chain_q35_kv_stage_begin` + `chain_q35_gdn_stage_begin`) によりステップ中途の grow 再確保は構造的に不可能
- 重みレジストリ (`weights_for` / `weights_for_group`): 無 eviction (実運用 1 モデル ~2GB)・uid 重複は debug_assert
- KV ミラー cap 成長: 2048 行未満では不発 (1b/2b とも pos < 2048)
- 残留注意点: chain の `attn_sc` スクラッチが cold プロセスの最初のステップ群で grow するが、ensure 時点で in-flight 利用者が無いことを確認 (decode chain 側であり 2b prefill とは無関係)

## 6. 分岐起源の推論 (第35セッション最大の知見)

1. 2994feb 時の実機 E2E は「4 リクエスト全別テキスト」だった — これは CH_QKV grow 競合が**プロセス内で最初に GDN decode ステップを通るリクエストでのみ発火**する仕組みと整合 (0c94b05 で構造排除後は 1b が回復)
2. 0c94b05 で decode 競合を構造排除した結果、1b は ×4 完全一致に回復。しかし 2b のペアは修正前後で同一 → **decode は分岐の起源ではない**
3. decode chain は 1b も 2b も同一コードを通る共通経路。1b が安定している以上、decode chain が 2b のみで揺れる構造は存在しない
4. 以上から **分岐は 2b の prefill 内で発生し、decode は prefill の結果から決定論的に続く** (HANDOFF の「prefill 経路の別競合疑い」を支持)。prefill が 2 通りの隠れ状態を生み、それぞれから decode が決定論的に分岐するため出力がちょうど 2 値になる
5. 該当する 2b 固有 GPU コードは §4 のとおり `dno_gemm_i8` 経路と `maybe_gpu_ffn` (`dno_swiglu_int`)

棄却した候補 (根拠付き):

- **prefix cache / live 再利用の経路分岐**: 同一 prompt の再実行は `cap = prompt.len - 1` でエントリ復元拒否 (logits は forward した最終位置からしか得られないため最後の 1 トークンは必ず forward)・recurrent モデルは「純拡張でない限り」full prefill にフォールバック — engine テスト (`recurrent_entry_restore_matches_full_prefill` 等) で裏付け済み。2 変体が揺れる説明にならない
- **prefill 分割**: `step_chat` は残りプロンプトを 1 回の forward で処理する設計 (分割すると GEMM 蓄積順が変わり exactness が壊れるため意図的に単一呼び出し)
- **1b/2b payload の先頭共有**: content 共通接頭辞 0 文字 (1b = 「日本の首都は?」7 文字・2b = 要約 prompt 1446 文字) → 相互汚染なし
- **GPU_BROKEN 一時フォールバック**: latch すれば以後ずっと CPU 値 `da0b1f57` になるはずで、ペア (`3a1fbabe`/`d46ab41f`) に含まれない

## 7. 未解決事項と次の一手 (人間対応・メインセッション)

コンテナに GPU が無いため静的監査では同定できなかった。**最終同定は実機 GPU での直接観測が必要**:

1. **prefill 層別 checksum の 2 変体間比較 (最優先)**: 恒久診断 `[LCHK]` (`DENNO_DEBUG_LAYER_CHKSUM=1` + 必要なら `DENNO_DEBUG_LAYER_CHKSUM_FILE`) と `gdn_dbg` / `[GDUMP]` (`DENNO_DEBUG_GDN_DUMP[_LAYER]`) を使い、2b が `3a1fbabe` と `d46ab41f` に分かれた 2 実行の **prefill (`[LCHK] prefill emb` / `L00..L31 attn`・`ffn` / `final hf` / `final logits`) を 2 変体間で比較** → 初発散層を直接特定。env 未指定時 no-op の恒久機能 (commit 済) のため診断コードの追加は不要。**GPU 再ビルド/サーバ再起動はメインセッションのみで実施**
   - 比較対象は「GPU あり vs CPU」ではなく「GPU あり vs GPU あり (別ラン)」である点に注意 (GDN 層は緩和契約内 drift で CPU と一致しない・`lchk_cpu_reference_q35_9b_1.log` は 1b 分の CPU リファレンス)
2. **`dno_swiglu_int` のテールガード精査**: 発動は grid = ceil(m*ff/256) × block 256 (`maybe_gpu_ffn` 内・cuda.rs:1109)。m*ff が 256 の倍数でない端数タイルの扱い (範囲チェック) を精査する
3. **2b 固有経路の実行ログ比較**: request 間で m と経路種別 (gemm/ffn 発動の有無) が同一であることを prof / tracing ログで確認 (経路分岐の裏取り)
4. 規律: push は人間対応・docs/HISTORY は完了した変更のみ追記 (本セッションは調査のみのため追記なし)

## 8. 検証・作業ツリー状態

- **コード変更ゼロ** (調査のみ)。`cargo test --workspace`: **215 passed / 0 failed** (engine lib 169・server lib 25・scheduler 7・tools 8・telemetry 5+1 — 05e5f8f と同一。本セッションで再実行し確認)
- 第35セッションで稼働していた CPU 検証サーバ (port 8290・target-dev release バイナリ) は本セッションで停止済み (残存プロセス 0)
- 本 REPORT のみを commit (REPORT_NEXT9/10/11・HANDOFF_NEXT.md・diag_*.patch・drift_exp/・_archive/・target-dev/ 等の他の未追跡ファイルは commit 対象外)。push は恒久規律どおり未実施

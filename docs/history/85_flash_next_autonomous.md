# 第9巻 Flash-Next の自律開発と Intel GPU 移植(2026-11-01 〜 11-17)

この巻は、Qwen3.8-Flash-Next の配信経路 `serve-flash` を、エージェントの開発チームに 24 時間まわさせて速くした 17 日間の記録。
数値は特記の無い限り RTX 5090 1 枚の実測で、1 リクエストずつ、temperature 0、出力の md5 で同一性を確かめている。

## 1. 体制

- **監督 1 + 開発 4**: 監督エージェントが 5 分ごとに各開発エージェントの状態を見て、終わったタスクの採否を判定し、採用したブランチを統合ブランチ `int-auto` へ merge し、次のタスクを投入する。
  - dev1: 既定モデルの decode
  - dev2: 既定モデルの prefill
  - dev3: 大きい量子化(UD-Q4_K_XL)
  - dev4: Intel GPU(Arc Pro B60)への移植(ブランチ `arc-port`、別のマシン)
- **GPU lab**: 開発エージェントは GPU に直接触らない。commit とリクエスト(条件ごとの環境変数)を書くと、ホストのジョブがその commit をビルドし、使い捨てのサーバーで計測して `RESULT.md` を返す。計測中は他の開発エージェントのコンテナを一時停止して、計測の汚染を防ぐ。
- **採用の規則(2% ルール)**: 速度を最優先にする。ゲートを足す変更は、2 回以上の中央値でその段が 2% 以上速く、他が 1% より悪くならず、出力の文章がまともなら、目標に届かなくても既定 ON で採る。
- **統合ごとの回帰計測**: `int-auto` に merge するたびに、同じ条件(温まった状態で 3 条件)を lab で測り、前回と比べる(`tint_<sha>`)。

## 2. decode の推移(短い会話の生成、ms/tok)

| 日付 | 値 | 主な変更 |
|---|---|---|
| 11-03 | 167 → 120 → 86 | IQ3_S の融合 GEMV(task 117)、mix と router を GPU へ(task 119b / 121) |
| 11-04 | 92 → 81 | 共有 expert の重ね合わせ(124)、decode の full attention を GPU へ(133) |
| 11-05 | 81 → 62 | GDN の scan を GPU へ(142)、expert を VRAM に 20 GB 常駐(147) |
| 11-06 | 62 → 約 40 | decode の連鎖化(146 / 150)、expert 予算 22 GB(152)、155 |
| 11-07 〜 11-10 | 28(167 の計測)→ 21 | flashdec(163)、miss 経路の高速化(184 / 187 / 190 / 192)、214、221 |
| 11-12 〜 11-14 | 20 → 17.0 | 285、307 |
| 11-15 〜 11-16 | 17.0 → 16.6 | 319、327(mix の注入)、342 |
| 11-16 | 17.9 → 16.6 | 既定モデルを unsloth UD-IQ3_XXS に切り替え(§5)、364(router の GEMV を mix の連鎖の末尾へ) |
| 11-17 | 16.6 → **15.3** | 368(mix の norm)、382(mix の融合) |

## 3. prefill の推移(cold、秒)

| 日付 | 3,826 トークン | 14,018 トークン | 主な変更 |
|---|---|---|---|
| 11-04 | 135.1 → 110.6 | 758.9 → 379.2 | prefill の full attention を GPU へ(134) |
| 11-05 | 107.9 → 89.0 | 366.5 → 328.6 | hc mix の GEMM を GPU へ(145) |
| 11-07 | 62.9 → 27.3 | 210.1 → 93.9 | expert の GPU prefill(151 / 161)、expert の stream 読み込み(165)、167 |
| 11-08 | 27.3 → 17.8 | 94.1 → 59.0 | BF16 tensor core の射影(169 / 178) |
| 11-10 | 11.7 → 8.7 | 40.0 → 31.0 | チャンク 1536(207) |
| 11-13 | 7.27 → 3.34 | 25.1 → 8.4 | prefill 用のプールを expert から借りる(265)、279 |
| 11-14 〜 11-15 | 3.42 → 2.54 | 11.0 → 6.54 | 300、307、308、317、320、324 |
| 11-16 〜 11-17 | 2.52 → **2.44** | 6.5 → **6.0〜6.25** | PLE の gather 並列化(353 / 357)、KV の非同期書き込み(369)、379、モデル切り替え |

## 4. 大きい量子化(UD-Q4_K_XL)

111 GB あり、64 GB の RAM に収まらない。expert の一部を毎回 SSD から読むので、読み込みとの戦いになった。

- 3,826 トークンの cold prefill: 55.3 s → 39.3 s(297)→ 29.9 s(301)→ 13.4 s(318、K 系の量子化を GPU で)→ 10.3 s(325、pinned メモリへ pread)→ 約 9 s(358、pread を 32 本並列)。
- 14k: 138 s → 110 s(301)→ 30.1 s(318)→ 25〜26 s。
- 生成: 23〜25 ms/tok → 20〜21(286、297)→ 18〜19。
- task 381 で、3,826 トークンの prefill は約 44 GB を 5.95 GB/s で読んでいて SSD で頭打ち、14k は GPU で頭打ちだと分かった。

## 5. 既定モデルの切り替え(11-16)

- 11-15 まで既定にしていた IQ3_S は、第三者が調整した派生モデル(Swift-1.5 abliterated)だった。オペレーターの指示で、公式の量子化 unsloth `Qwen3.8-Flash-Next-GGUF` の 2026-11-13 版に切り替えた。素の IQ3_S は無いので、同じ大きさの `UD-IQ3_XXS`(83.6 GB)を既定にし、`UD-Q4_K_XL` も 11-13 版に差し替えた。
- 11-13 版はそのままでは読めなかった。先頭の分割ファイル(メタデータのみ、tensor 0 個)の末尾に 32 バイト境界までの詰め物が無く、データ部の開始位置がファイル長を 6 バイト超える。tensor が無いファイルではこの検査を省くよう直した(`42500e5`)。
- 型も変わった: hyper connection の重みが BF16 → Q8_0、ssm_alpha / beta が F32、token_embd が Q6_K、expert は IQ3_XXS / IQ2_S / IQ4_NL / IQ4_XS などの混在。既存の IQ 用カーネルはそのまま効き、切り替え直後の速度は旧モデルとほぼ同じだった(decode 16.6〜16.8 ms/tok、3,826 cold 2.50〜2.59 s)。

## 6. Intel GPU への移植(ブランチ `arc-port`)

Level Zero を dlopen する `ze.rs` が `cuda.rs` と同じ API を持ち、カーネルは OpenCL C(`kernels_xe/`)を ocloc でコンパイルして commit する。

| 時点 | 生成 | 3,826 cold prefill |
|---|---|---|
| 11-11 CPU のみ | 183 ms/tok | 224 s |
| 11-11 〜 11-12 | 128 → 104 ms/tok(236 / 237 / 247) | 113.5 → 66.4 s(241 / 255) |
| 11-14 | 60.8 ms/tok(313) | 35.7 s(304) |
| 11-17 | 2 本目の要求で 88〜90 ms/tok(§7) | **15.1〜15.4 s**(391 までの採用分) |

## 7. 2026-11-17 の実測

README の「計測値」と同じ。rtx5090 は `8ae66f334`、arc は `arc-port` の `a4f891d28` で測った。

- rtx5090 / UD-IQ3_XXS: 生成 15.3 ms/tok(65 tok/s)、3,826 cold 2.44 s、14k cold 6.00〜6.25 s、2 ターン目の応答開始 0.45〜0.51 s。
- rtx5090 / UD-Q4_K_XL: 生成 18.2〜19.3 ms/tok、3,826 cold 9.2〜9.8 s、14k cold 25.2〜25.9 s。
- arc / swift15 IQ3_S(旧既定): 3,826 cold 15.1〜15.4 s、14k cold 62.6 s。生成は同じプロセスの 2 本目で 88〜90 ms/tok。**3 本目の要求で毎回 GPU がページフォールトを起こし(`ZE_RESULT_ERROR_DEVICE_LOST`、カーネルログに engine reset)、計測を続けられない。** 開発エージェントの作業中のビルドでも同じだったので、再開後の最初の修正対象にする。
- arc / UD-IQ3_XXS(新既定): 3 本目でも落ちず、生成 56 ms/tok(3 回とも同じ)、3,826 cold 29.4〜29.8 s、14k cold 107.8 s。ただし文章が崩れる(28 トークンの質問の答えが 44 トークン目で役割名 `user` を出して止まる、3,826 トークンの答えが `preffill`)。新モデルで変わった型(Q8_0 の hyper connection など)の Xe 側の扱いが合っていない。これも再開後の修正対象。

## 8. 棄却したもの(主なもの)

| task | 試したこと | 結果 |
|---|---|---|
| 337 | GPU に無い expert の予測と先読み | 予測の当たりは 83% だが、1 本の転送 227〜280 µs が CPU で計算する 79〜142 µs より遅く、得にならない |
| 338 / 339 | 既定 OFF のゲートを全部測り直す | どれも 2% ルールを満たさない |
| 349 | ページキャッシュが冷えた状態の prefill | 先読みや pread でも効果なし |
| 370 / 374 | 14k の ssm の待ち時間 | 待ちは転送ではなく GPU の処理。途中の状態の書き戻しを 86% 消しても壁時計は動かない |
| 381 | UD の GPU に無い expert の CPU 経路 | 0%。3,826 は SSD、14k は GPU で頭打ち |
| 385 | prefill の expert 転送待ち | 効果なし |
| 335 / 345 | arc の GEMV のベンチマーク調整 | ベンチの数値は L2 キャッシュの値で、実際の生成は動かない |

可逆圧縮で読み込み量を減らす案も実測で退けた: zstd では expert 部分が 0.5% しか縮まず、量子化ブロックのフィールド別エントロピーでも上限は 3〜5%(格子のインデックスは 97%、符号ビットは 100% の情報量)。

## 9. 残っているボトルネック(2026-11-17)

- 生成 16.3 ms/tok の内訳: expert 4.73(GPU に無い 8% を CPU で計算する 2.33 を含む)、mix 4.73、ssm 3.52、full attention 1.47。
- prefill 3,826: expert 1.32 s のうち 1.09 s が転送待ち。
- prefill 14k: ssm 2.61 s(GDN の delta 則の scan はチャンクの順番待ちで、36 層ぶん繰り返す)。

## 付録: 採用した変更の一覧(`int-auto` の merge commit、日付順)

| 日付 | commit | 件名 | 計測(merge のメッセージから) |
|---|---|---|---|
| 11-03 | `c800c39` | merge: task 117 (IQ3_S decode fused GEMV) + task 118 (model reuse across requests) | decode 167 -> 120 ms/tok, reuse=0 prefill 5.1 -> 2.2 s, resident GPU weights 6592 -> 3296 MiB (swift15-abl IQ3_S, 128 tok, A/B on a test port). Output text matches the... |
| 11-03 | `e23ea92` | merge: task 119b (mix/router FMA, 14 thread default) + task 121 (long-row mix GPU kernel) + task 120 gates | Decode 123 -> 113 ms/tok by default, 86 ms/tok with DENNO_MIX_GPU=1 (95 on 1000 tok), swift15-abl IQ3_S, A/B on a test port. Output text matches on 9 request patterns.... |
| 11-03 | `3c9e0ac` | merge: br-view2 residual wall heatmap for the 3D viewer |  |
| 11-04 | `95ae16c` | merge: br123 | t123a lab: disconnect after_cut prefill 3.76s->2.25s, decode 92->87ms (-5.4%), errors 0, md5 same, text ok |
| 11-04 | `41e428e` | merge: br124 | t124a: decode 86->84 ms/tok (-2.9%), prefill 2.25->2.25 s (+0.0%), md5 SAME, errors 0, DENNO_SHARED_OVERLAP default on |
| 11-04 | `ee56ba1` | merge: br127 | decode 87 -> 84 ms/tok (4.0 percent faster), prefill 2.56 -> 2.58 s (0.8 percent slower), t127c OK errors 0 md5 SAME |
| 11-04 | `bc59859` | merge: br126 | task 126: prefill mix seq default on (t126c) prefill 2.35->1.48 s (-37.4%), decode 86->86 ms/tok (+1.2%), md5 SAME, errors 0, ADOPT yes |
| 11-04 | `400a37a` | merge: br133 | task 133: decode full attention on GPU. t133b decode 81 vs 80 ms/tok, DEPTH c2000_turn2 122->95 (-23.1%), c8000_cold 210->121 (-42.4%), md5 SAME, errors 0 |
| 11-04 | `ac21a1d` | merge: br131 | task 131: chat template trim prefix reuse via splice. t131a decode 81 vs 81, prefill 1.44 vs 1.44, md5 SAME |
| 11-04 | `7711f98` | merge: br134 | task134: prefill full attention on GPU (default on); c2000 cold 135.05->110.56s (-18.1%), c8000 cold 758.94->379.20s (-50.0%), short prompt +0.0%, decode +1.3%, errors 0 |
| 11-04 | `d30d610` | merge: br136 | task 136 merge br132: t136a base decode 82/96-97/119 ms/tok (int-auto 80/94/120), md5 identical to t134b on, unexplained 0.027-0.040 ms/tok, CPU 758 lib + 24 kernel pass |
| 11-05 | `891def0` | merge: br139 | t139b depth8k: c8000 turn2 decode 119->93 (-21.8%), c2000 turn2 96->86 (-9.4%), short decode 81->81, next turn prefill 1.06/1.13 s, md5 SAMEE |
| 11-05 | `dbe9f48` | merge: br141 | task141: reuse prefix when old assistant tail was trimmed. lab t141a VERDICT OK decode -0.6% prefill -1.4% md5 SAME errors 0; CPU lib 762, kernel_invariants 24, p131_p... |
| 11-05 | `065cf20` | merge: br138 | task138 t138c: decode 77ms/tok (off 81, -5%), c2000_turn2 79 (off 85), md5 SAME, errors 0 |
| 11-05 | `bd59dfd` | merge: br142 | GDN GPU scan default on (t142b): decode 78->75 ms/tok (-4.0%), c2000_turn2 80->79 (-1.3%), c8000 89->84 (-6.0%), prefill within 2%, md5 SAME, errors 0 |
| 11-05 | `2766714` | merge: br145 | task 145: prefill hc mix GEMM on GPU default on; t145c depth8k c2000 cold 107.87 to 89.00s, c8000 cold 366.49 to 328.64s, next turn 0.98 to 0.97 and 1.04 to 1.04, deco... |
| 11-05 | `72fbd75` | merge: br147 | t147c depth8k t144 merge: decode 75 to 62 ms/tok (-17.3%), prefill 1.35 to 1.33 s (-1.8%), md5 SAME, errors 0, c8000 decode -28.9% (v20 = IQ3_S expert VRAM 20000 MB) |
| 11-06 | `8ce299a` | merge: br146 | t146c depth8k: decode 62->49 ms/tok -21.0%, prefill +1.1%, md5 SAME, errors 0 |
| 11-06 | `dd43895` | merge: br150 | task150: DENNO_DECODE_CHAIN2 default on; t150b decode -9.7% short, -11.7% at 3954 turn2, -12.3% at 8000, prefill -7.0%, errors 0, md5 same |
| 11-06 | `7f80865` | merge: br152 | task 152 br152 (t152c depth8k): expert budget 20000->22000 MiB, decode 28/3954/14139 = 56/51/52 -> 53/48/48 ms/tok (-4.9% at 3954, -7.7% at 14139), prefill +1.2%, md5 ... |
| 11-06 | `09c2348` | merge: br151 | task151 prefill expert GPU: decode +0.0% prefill -23.0% (c8000_cold -33.7%) errors 0 |
| 11-06 | `14caf76` | merge: br153 | task 153 decode timeline (GPU busy/idle/host) + miss-expert rows parallel; lab t153b VERDICT on vs off OK decode -9.9% prefill +1.6% md5 SAME errors 0 |
| 11-06 | `00b744c` | merge: br155 | task155: decode -8.7% (t155a VERDICT, prefill +1.1%, md5 SAME, errors 0); DEPTH t155c decode -10.0%/-10.6%/-12.0%/-10.0%, prefill within 1.4% |
| 11-07 | `af29f52` | merge: br161 | task 161 merge br154: lab t161a on e17f9f460 default cold prefill 62.89->49.92 s (3826 tok) and 210.09->157.05 s (14018 tok), decode short +0.0%, errors 0 |
| 11-07 | `8a1b2e7` | merge: br160 | task 163: flashdec default on; lab t163b VERDICT decode +0.0% prefill +0.0% md5 SAME errors 0; depth decode -3.3/-3.7/-15.6/-16.1% |
| 11-07 | `b18125b` | merge: br165 | task_165: prefill expert stream on by default (WC + group 64 + chunk 1024); lab t165c cold prefill 3826 tok 35.33->27.32 s (-22.7%), 14018 tok 112.47->95.31 s (-15.3%)... |
| 11-07 | `7aaef05` | merge: br167 | t167c depth8k: cold prefill 3826tok 29.68->27.29s (-8.0%), 14018tok 102.78->93.87s (-8.7%), decode unchanged 28/29/26 ms/tok, md5 SAME, errors 0 |
| 11-08 | `d2972f1` | merge: br164 | lab t164c: decode -3.4 to -3.8 pct, 14157 next turn -2.97 pct, prefill within 0.3 pct, md5 SAME, errors 0; approved by main session 2026-11-08 |
| 11-08 | `5e89a1b` | merge: br169 | task 169: DENNO_PF_TC_GEMM default on (BF16 tensor core prefill projections); lab t169b cold prefill 14018 tok 94.05->69.58 s (-26.0%), 3826 tok 27.30->20.52 s (-24.8%... |
| 11-08 | `5d22388` | merge: br174 | task 174: fix flashdec prefetch out-of-bounds (CUDA error 700 at len=cap) + DENNO_GDN_PF_FAST default on; lab t174c prefill -3.9% at both cold lengths, errors 0, lib 8... |
| 11-08 | `d4a1f5b` | merge: br178 | task178 merge br176 (DENNO_PF_TC_GEMM2): lab t178a cold prefill 19.76->17.83s (-9.8%) at 3826, 66.65->59.02s (-11.4%) at 14018, errors 0, short decode 27->27 ms |
| 11-08 | `d123eca` | merge: br181 | prefill c2000_cold -8.9% c8000_cold -9.4% decode +0.0% errors 0 (t181c sha ad9191fb3) |
| 11-08 | `e88c5f2` | merge: br185 | task 185: merge br179 into int-auto; lab t185a decode -5.5% prefill +0.0% md5 SAME errors 0 (927 lib / 37 kernel tests pass) |
| 11-08 | `3a1f870` | merge: br182 | task 182: prefill 3826 tok -9.0%, 14018 tok -5.5% vs off (lab t182d); decode within 2%; GPU router + PLE f32 projections default on |
| 11-09 | `892cd99` | merge: br183 | task 183: fused qkv one GEMM + shared BM=64 tile (t183b) cold prefill -6.7% at 3826 and -7.1% at 14018, decode 0%, errors 0 |
| 11-09 | `3638edb` | merge: br184 | task 184 ADOPT t184b: decode -3.8% short, -8.0..-8.3% depth, prefill -1.7%, errors 0, md5 SAME |
| 11-09 | `fad19a7` | merge: br187 | task 187: decode -8.0% short and -8.3/-4.5/-8.7% depth (lab t187c), prefill +0.8%, errors 0, text same, three gates default on |
| 11-09 | `f94014b` | merge: br189 | scan v2 default on: t189c cold prefill -3.1% / -3.2%, md5 SAME, errors 0; other gates stay off |
| 11-09 | `2b9ee43` | merge: br190 | task 190: DENNO_EXP_IQ_V2 + DENNO_DEC_MISS_FAST default on; decode -4.3% (depth8k), long-ctx decode -4.5..-4.8%, prefill -0.8%, md5 SAME, errors 0 (t190c) |
| 11-09 | `c2d56f3` | merge: br192 | decode t192c def3: -4.6 to -9.7 percent raw seconds, prefill -0.5 to -4.2 percent, md5 SAME, errors 0 |
| 11-10 | `725c446` | merge: br198 | prefill 4 gates default on, t198b |
| 11-10 | `f65b727` | merge: br200 | prefill attn flat and devres default on, t200c: cold -4.7 and -5.0 percent, md5 SAME |
| 11-10 | `5839150` | merge: br207 | task 207: PF_CHUNK default 1536 (t207a): cold prefill 3826 -26.1% (11.73->8.67s), 14k -22.6% (40.04->30.98s), decode within 0.5%, errors 0 |
| 11-10 | `3db6946` | merge: br206 | task 206: aux ctx keeps 2nd-turn prefix; t206a c8000_turn2 prefill 65.27s->0.47s, decode 35->20 ms/tok |
| 11-10 | `61d139e` | merge: br209 | task 209: DENNO_PF_PIN + DENNO_PF_MIXDEVIN default on; cold prefill c2000 -5.1% c8000 -4.4% (t209b), decode +0.0%, md5 SAME, errors 0 |
| 11-10 | `389419b` | merge: br210 | task210: 捨てた AttnCache の GPU ミラー解放 + 確保失敗時の追い出しを階段に。t210a c8000_cold decode 35->22 ms/tok (fix off 39)、c8000_turn2 25->22 |
| 11-10 | `38c7252` | merge: br214 | task 214 decode -8.7% (23->21 ms/tok) prefill -0.9% errors 0 |
| 11-10 | `313c951` | merge: br219 | task 219: /2d viewer (denno-view only; node 49 pass, cargo denno-view 5 pass, engine lib 998 pass, kernel_invariants 70 pass; no lab) |
| 11-10 | `14db8a2` | merge: br221 | task 221: fold+DEC_MIX_UPW+EXPGU default on; t221d decode 20.4/20.5 vs off 21.2 (-3.8%), c8000_turn2 -23.0%, errors 0 |
| 11-10 | `b6f6670` | merge: br217 | task 223: intervention 3種を t217b で base2 比 1% 以内 (empty decode 21.2 vs 21.2 prefill 0.57 vs 0.57 md5 SAME) |
| 11-10 | `d1962e9` | merge: br222 | task 222: prefill TC kernels (attn_pf2/sn3) default on, 14k cold prefill 27.05 s vs off 28.84 s (-6.2%), decode 20.3 vs 20.4 |
| 11-10 | `56877d3` | merge: br218 | task 218: residual mean snapshots + steering; t218c base +0.6% rec +1.0% zero +0.4% md5 SAME, sub text readable, errors 0 |
| 11-11 | `3c2a92e` | merge: br220 | task 220: viewer to real intervene/steer API + ExpertSkip telemetry. lab t220a depth8k: decode +0.2% prefill 0.0% vs b0, vs int-auto t220base decode 0.0% prefill 0.0%,... |
| 11-11 | `c81d677` | view: /2d の列の高さを画面に合わせ、チャットの入力欄が画面外に出ないようにする | main の grid 行が中身 (48 層の地図、約 2,470px) の高さになり、body の overflow:hidden で チャット欄の下端 (入力欄と send) が見えなくなっていた。行を minmax(0,1fr) にし、 section に min-height:0 を足して、地図とログはそれぞれの枠の中でスク... |
| 11-11 | `9fb2942` | merge: br235 | task235 denno-view wall test 1041+81+5 cargo passed, 64 node view tests passed, no lab (no-lab) |
| 11-11 | `478411f` | merge: br228 | task 228/231: BA_FAST, ROUTER_DEV, SCAN3 defaults on (SCAN_WY off); 14k cold prefill -2.9% (base2 -4.4%), 3.8k cold -4.9% vs off, decode flat, errors 0 |
| 11-11 | `1d36583` | merge: br244 | task 244: prefill attn v3 default on; 14k prefill 25.907 -> 25.13 s (-3.00%), 3.8k 7.11 -> 7.06 s (-0.70%), decode 20.27 -> 20.25 ms/tok; CPU 1057/88 pass |
| 11-11 | `f853d4a` | merge: br243 | task 243 operator 2026-11-11: merge DENNO_START_PREFETCH (default off) so production can turn it on; t243c cold machine first short decode 12.21 -> 3.00 s, c2000_cold ... |
| 11-12 | `6bd4f9c` | merge: br252 | task252 GDN seq state take zeroes host in both branches: repeat 3 prompts x 4 rounds same md5, decode 18.3 ms/tok prefill 0.41 s errors 0; depth8k md5 same as discarde... |
| 11-12 | `31f3099` | merge: br254 | task254 br254: 14k cold prefill 28.30 -> 25.21 s (-4.8%), decode -1.3%..-2.3%, errors 0, md5 SAME |
| 11-12 | `822e1ec` | view: clear KV cache and residual view buttons on /2d | serve-flash gains GET /v1/cache (main / aux context token counts and epoch) and DELETE /v1/cache, which drops both cached contexts (full attention KV, GDN recurrent an... |
| 11-12 | `a9f8fc4` | view: keep n_expert_used experts per layer when switching experts off | A drag over a whole layer tile selects all 512 experts (or nearly all), and POST /v1/intervene answers 400 because the router needs at least n_expert_used (10) experts... |
| 11-12 | `801d4cb` | serve-flash: POST /v1/abort stops a running generation; /2d stop and clear chat | Some expert-off settings make the model loop until max_tokens. serve-flash now reads each request on a per-connection thread and hands it to the main thread as before ... |
| 11-12 | `49a542d` | serve-flash + /2d: choose the Qwen3.8 Flash-Next model from the viewer | Only one model fits in VRAM and RAM, so choosing a model means restarting serve-flash on it. GET /v1/model/choices lists the models from the file in DENNO_MODEL_CHOICE... |
| 11-12 | `30441fa` | serve-flash: flush steering on a model switch only while recording | The flush returned an error when nothing was recorded, so every switch logged a misleading [steer] save failure. |
| 11-12 | `62a9ad1` | merge: br256 | task256 slab-chunk evict: base md5 SAME decode +1.0% prefill +0.0%, forced 24000MB cond errors 0 text sane, alloc bounded to req+1 chunk |
| 11-13 | `f16298b` | merge: br265 | task 265 (main session override, operator approved 2026-11-13): prefill pool lent from experts only during long prefill. t265c: 14k cold 25.07 -> 16.54 s (-34.0%), 3,8... |
| 11-12 | `b8e5de2` | task 265 adopted by the operator: prefill pool path default on | DENNO_PF_LM_LAYER, DENNO_PF_TC_EXP and DENNO_PF_POOL are on unless set to 0/off/false/no; DENNO_PF_LM_TILE defaults to 5120 (explicit 0 = no split). Lab t265c: 14k col... |
| 11-12 | `89b7f8a` | task 265 adopted: tc_exp default on in the lm_reserve gate test |  |
| 11-13 | `2a7dbf7` | merge: br271 | task 271: expert tier in /2d + stats save on switch and SIGTERM; lab t271a stats vs b0 decode -4.8% prefill -6.5% errors 0 md5 SAME; c8000_cold decode -3.9% prefill -4.3% |
| 11-13 | `46550af` | merge: br275 | tasks 269+275 (main session override, operator approved 2026-11-13): DENNO_EXP_IQ_V3 + DENNO_DEC_MIX_NORM_PAR default on. t275c: short decode -3.0%, decode after 14k -... |
| 11-13 | `9c5184c` | tasks 269+275 adopted by the operator: DENNO_EXP_IQ_V3 and DENNO_DEC_MIX_NORM_PAR default on | Both are on unless set to 0/off/false/no. Lab t275c: short decode -3.0%, decode after 14k -5.5%, md5 SAME, errors 0. DENNO_GDN_BA_GPU stays default off (14k next +3.0%). |
| 11-13 | `0566be1` | merge: br279 | task 279: 14k cold prefill 16.76/16.80 -> 8.43/10.23 s (-39/-50%), 3826 cold 3.33/3.65 -> 3.35/3.62, decode 19.4 same, errors 0, VERDICT OK |
| 11-13 | `f1d7413` | merge: br278 | merge: br278 task278 seg-layer + seg_k GEMV (on) - decode 65.9->33.0 ms/tok -49.9% (t278b), 65.5->34.7 -47.0% (t278c), prefill -3.6%/+1.8%, errors 0, gate 1168+93 pass |
| 11-13 | `c5cb293` | merge: br282 | task 282: mix qchain default on (t282c decode -23.8%, prefill -14.2%, depth c2000_cold -17.6% / turn2 -25.4%, errors 0, turn2 md5 diff but text identical prefix) |
| 11-13 | `c997035` | merge: br285 | task285 decode 18.8 -> 17.9 ms/tok (-4.8%), prefill +0.8%, errors 0, text ok |
| 11-13 | `7890d12` | merge: br286 | task 286 ADOPT yes t286c: UD-Q4_K_XL warm short decode 23-25 -> 20-21 ms/tok (mix GEMV on v5), IQ3_S unchanged (t286d) |
| 11-14 | `7c2e83f` | merge: br290 | task290 t290c: UD-Q4_K_XL 14k cold 8/8 sane (211.6s -> 200.1s), depth8k decode -19.9% prefill -23.3% errors 0 |
| 11-14 | `999ce35` | merge: br297 | UD Q4_K_XL 3826 cold prefill 55.30 -> 39.30 s (-28.9%), warm short decode 22.0 -> 20.4 ms/tok, default model md5 SAME (4 cond), errors 0 |
| 11-14 | `7794d2d` | merge: br301 | task 301: UD-Q4_K_XL 3826 cold prefill 38.46 -> 29.94 s (-22.2%), 14k 138.3 -> 110.4 s, mix 8.5 -> 0.3 s, default model md5 same, errors 0 |
| 11-14 | `0217194` | merge: br300 | t300a/t300b: 3826 cold prefill -7.3pct (3.42->3.17s), 14k cold -7.1pct, decode -0.2/+1.9pct, errors 0 |
| 11-14 | `c8b41a2` | merge: br308 | task308 prefill 3826 cold 3.13 -> 2.94 s (-6.1%, t308b round2), 14k cold 10.30 -> 9.44 s; decode 3826 18 -> 18 ms/tok, errors 0, text sane |
| 11-14 | `0179e96` | merge: br309 | task 309 ADOPT yes: t309b kpg decode +7.8% prefill -4.3%, kwc c2000_cold prefill -42.8% decode +60.3% (errors 0); t309c swift15 on vs off decode +0.4% prefill +0.0% md... |
| 11-14 | `d6f914e` | merge: br307 | task307: decode 17.9/18.0 -> 17.0 ms/tok (-4.6%), prefill c8000_cold 11.04 -> 7.98 s (-27.7%), errors 0 md5 SAME (t307c) |
| 11-15 | `ba85315` | merge: br317 | task317: ssm attn-mix keep (no D2H) + wy4 default on; t317a c2000_cold 2.943->2.780s (-5.6%), c8000_cold 7.393->6.754s (-8.6%); t317b c2000_cold 3.03->2.82s (-6.9%), c... |
| 11-15 | `6465bf5` | merge: br318 | P318 UD-Q4_K_XL K-dev MoE gate: 3826 cold prefill 16.71->13.36s (-20%), 14k cold 36.34->30.10s (-17%), decode after cold also faster, errors 0 |
| 11-15 | `0d3cd40` | merge: br320 | 3,826 cold 2.77->2.62 s (-5.4%); 14k cold 6.98->6.77 s (-3.0%); decode new vs x0 +0.9%; errors 0; md5 SAME |
| 11-15 | `3ae8b00` | merge: br319 | task 319 ADOPT yes (shared v5 default on, short decode -2.4/-2.7%); supervisor skipped the merge |
| 11-15 | `fcbdf64` | merge: br324 | task 324: gpuwait+nosync+group32, 3826 cold 2622->2542 ms (-3.1%), 14k 6665->6543 (-1.8%), decode unchanged |
| 11-15 | `716e9a8` | merge: br325 | task_325: 3826 cold prefill 13.32 -> 10.25-11.05 s (-11% to -23%, t325b 2 rounds), copy 44.4GB 10.47 -> 7.70 s, 14k unchanged, md5 SAME errors 0 (t325b) |
| 11-15 | `ab94ac1` | merge: br333 | task333 depth8k x0 17.0 base 16.9 ms/tok; t333b decode -0.7% prefill +0.9% VERDICT OK errors 0 (fix task) |
| 11-15 | `4e4923d` | merge: br327 | mix stage 3.62 to 3.36-3.41 ms/token (-6.5%), decode after 14k -7.4%, short decode 16.7 ms/tok, md5 SAME errors 0 |
| 11-16 | `ef21be4` | merge: br334 | t334a/b/c: 14k cold median -2.0pct (rounds -5.3/-6.1pct), 3826 cold -2.2/-3.5pct, IQ3_S default decode +0.7pct prefill +0.0pct md5 SAME, errors 0 |
| 11-16 | `6c19792` | merge: br342 | task 342 hc32: decode -0.5% (t342c -1.4%), mix stage gpu -3~-4%, prefill -1.7~-3.3%, lab OK errors 0 md5 SAME t342a/b/c |
| 11-16 | `17e9386` | merge: br347 | task347 /metrics for sparkDash: decode 16.6-16.7 ms/tok (base0 16.8, base1 16.6), prefill 0.58-0.59 s, errors 0, md5 SAME, text sane |
| 11-16 | `0639e5b` | merge: br348 | task361 exp miss rows 64->16: miss CPU median -5.8%/-2.4% (r1/r2), decode 16.6-16.7 ms/tok unchanged, prefill +0.9%, md5 5 COND SAME, errors 0 |
| 11-16 | `1ce57fb` | merge: br353 | task353 PLE gather 32-way: 14k cold 9.29->8.06/8.08 s (-13.2/-13.8%), 3826 cold 8.85->8.41/8.42 s (-5.0/-5.2%), decode 16.6-16.8 unchanged, md5 SAME, errors 0 |
| 11-16 | `c1532cf` | merge: br357 | task357 PLE gather 32->96: 14k cold 8.08->7.80 s (-3.5%), 3826 cold 8.40->8.30 (-1.2%), decode 16.6 unchanged, md5 SAME, errors 0 |
| 11-16 | `2fd2875` | merge: br356 | task 356: expert sum written directly to pinned host, D2H removed; t356b decode -2.0% prefill -2.5% (c2000_cold prefill -51.8%), t356c prefill -4.1% decode +0.2%, md5 ... |
| 11-16 | `9df1f77` | merge: br358 | task 358: expert pread threads default 32 (DENNO_PF_PREAD_THREADS); t358b decode -9.4% prefill -39.6%, c8000_cold decode -36.8% prefill -15.0%, md5 SAME errors 0 |
| 11-16 | `e213685` | merge: brm1116 | unsloth 11-13 split GGUF loads (first shard has no trailing alignment padding); lab m1116b: UD-IQ3_XXS sane text, decode 17.9 ms/tok, 3,826 cold 2.69 s |
| 11-16 | `0157607` | merge: br369 | t369a/b/c: DENNO_PF_KV_ASYNC default on; 14k cold prefill 6.46 -> 6.30 s (-2.4%, same-run b1/b2 vs d1/d2), 3826 cold 2.52 -> 2.49 s, decode 16.6-16.7 ms/tok unchanged,... |
| 11-16 | `5df5426` | merge: br364 | task 364 router GEMV into qchain tail: decode 18.0 -> 16.6 ms/tok (-7.8%), prefill -0.7%, md5 SAME, errors 0 (t364c t364d) |
| 11-17 | `3b6dc1a` | merge: br368 | task 368: decode の mix norm 0.49 -> 0.21 ms/tok、sample 0.11 -> 0.04、短 decode 16.6 -> 16.2 ms/tok (-2.4% 2 run、t368b)、prefill 0.69 s 不変、md5 SAME errors 0 (t368b/t368c) |
| 11-17 | `9723b66` | merge: br379 | t379a/t379b/t379c: depth8k short decode 16.6 ms/tok unchanged, prefill 0.68-0.69 s (t379a VERDICT prefill -2.9%), c8000_cold prefill -23.8% (t379c) / -3.9% (t379a), er... |
| 11-17 | `8ae66f3` | merge: br382 | decode 16.2 -> 15.8 ms/tok (-2.9%) fused mix, prefill 0.68 s unchanged, errors 0, md5 SAME all 6 ctx |

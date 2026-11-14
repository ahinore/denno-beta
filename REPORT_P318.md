ADOPT: yes t318a t318c

理由: UD-Q4_K_XL の `other` (14k で 8.4-9.8 s、24-26%) の正体は「K 系の層優先
デバイス MoE が全 48 層でフォールバックし、層ごとに mixed を丸ごと往復していた
こと」だった。前提判定の型ゲートを IQ 専用から IQ or K に広げ、タイル 2 枚目で
落ちる `mixed ptr` のバグを直しただけで、**3,826 cold -17%、14k cold -19%**
(同じ run の baseline の中央値との比較)、decode after cold prefill も改善、
errors 0、text sane。丸め順だけが変わるので md5 は 4 本とも DIFF (§4)。

# REPORT P318 — UD-Q4_K_XL prefill: `other` の正体は K 系 dev MoE の全層フォールバック

branch: br318 = int-auto `d6f914e` (開始時点) から。最後に int-auto `ba85315`
(task 311 / 317) を merge した (衝突なし。build_ptx.sh の `targets_all` は
`gdn_pf_wy4` と `exp_k_pf` の両方が残っているのを確認)。
lab: **t318a** (sha 9561680) / **t318b** (sha 1062cf2) / **t318c** (sha bd6dc3d)。
3 ラウンドで上限。すべて `MODEL=flash-next-UD-Q4_K_XL`、baseline COND = int-auto
既定 + `DENNO_EXPERT_IQ_GPU=1 DENNO_EXPERT_VRAM_MB=22000 DENNO_EXPERT_GPU_MIN_HIT=0
DENNO_GPU_WEIGHT_MB=12000`、先頭に baseline の使い捨て (b0t)、末尾に baseline の
繰り返し (b1)。

## 0. 前提 (Do 1) と、P312 の数字との関係

P312 の「3,826 は転送バイトで決まる、14k は GPU 律速」は今も正しい。だが
`[q4ledger-pf]` の `other` が 3,826 で 2.4 s (15%)、14k で 8.4-9.8 s (24-26%)
あり、うち分類済みは 1.3 s だけだった。P309 §6 / P312 §5.5 が「未分類。層ループに
区間を足して測る」と残した宿題がこれ。

t318a で台帳に副区間 4 本 (res_up / res_park / inj / lm_park) と、転送 1 回ごとの
`[xfer] H2D|D2H <bytes> item=<区間> sub=<副区間>` (`DENNO_PF_XFER_DBG`、既定 off) を
足して撮ったのが下記。

## 1. t318a — `other` の正体 / pinned バウンスは負け

### 1.1 `other` の内訳 (dbg COND、`[xfer]` の集計。すべて実測)

3,826 cold の `other` 2,430 MB D2H + 3,145 MB H2D のうち:

| 内訳 | 3,826 | 14k |
| --- | ---: | ---: |
| **タグ無し 1 回 = 活性 1 枚** (`item=other sub=-`) | 48 x 39.178 MB = 1,881 MB | 48 x 143.544 MB = 6,890 MB |
| 同 H2D | 48 x 39.178 MB | 48 x 143.544 MB |
| res_up (残差の往復) | 313 / 235 MB | 1,148 / 645 MB |
| ple | 39 / 196 MB | 143 / 718 MB |

タイル無しの 1 回 39.178 MB (= 3,826 x 2,560 x 4) / 143.544 MB (= 14,018 x 2,560 x 4)
が層ごとに 1 往復ずつ、**6 つの名前付き区間の外側**で出ていた。同じログに

```
[pf-lm] 層 0: dev MoE を諦めた (理由: gate) → CPU (タイル 7009 単位)
```

があり、これが `moe_ffn_quant_batch_lm_dev_at` の前提判定 (`iq_expert_mode` =
IQ 専用) で **全 48 層**が `moe_ffn_quant_batch_tapped` (ホストの mixed 経路) へ
落ちていた証拠。落ちた先は `pf_lm_take_mixed` で mixed を丸ごと D2H し、
`pf_lm_upload_ffn` で結果を丸ごと H2D する (どちらも pageable の同期コピー。
このホストで 1-3 GB/s)。`other` の正体はこれで、3,826 で約 1.9 s、14k で約 7 s。

### 1.2 pinned バウンス (`DENNO_PF_PIN_STAGE`) → 採らない

1 MiB 以上の同期 H2D/D2H をスレッドごとの pinned バウンス経由にするゲートを
足して測った。

| cond | 3,826 cold | 14k cold |
| --- | ---: | ---: |
| b0t (先頭) | 16.11 | 34.76 |
| pin | 17.77 (+10.4%) | **55.28 (+59.0%)** |
| dbg (ログ用) | 15.92 | 36.11 |
| b1 (末尾) | **15.61** | **34.94** |

14k で +59% と壊滅的。推定: rayon のワーカー 14 本がそれぞれ 143 MB の pinned を
`cuMemHostAlloc` で取りに行き、その確保が device 全体を同期させるため。
3,826 でも +10%。**ゲートは既定 off のまま残す** (真理値表はテストで固定)。

## 2. t318b — K 系に dev MoE を開く (3,826 は勝ち、14k は落ちた)

`moe_ffn_quant_batch_lm_dev_at` の型ゲートを `iq_expert_mode` → `pf_expert_mode`
(IQ or K。K は `DENNO_PF_EXP_K` とストリーミングが前提) に変えるゲート
`DENNO_PF_LM_K_DEV` を追加。expert の m 行カーネルはストリーミング経路のものなので
これが正しい判定 (フォールバック側 `moe_experts_iq_pf_batch` も同じ
`pf_expert_mode` で通っていた)。

| cond | 3,826 cold | 14k cold | 備考 |
| --- | ---: | ---: | --- |
| b0t | 27.43 | 32.65 | 3,826 はホストのノイズで大きく外れた |
| kdev | **14.00** | 44.80 | 14k は下記のバグで全層フォールバック |
| b1 | 15.88 | 35.21 | |

3,826 は dev MoE が効いた (d2h 8,510.7 → 2,869.7 MB、h2d 55,279 → 47,750 MB、
`router` / `shared` の行が消えてデバイスで回る)。14k は
`[pf-lm] 層 0: dev MoE を諦めた (理由: mixed ptr)` で **全層フォールバック**し、
ストリームが 98,151 → 146,981 MB に増えて 44.80 s。原因は
`pf_lm_mixed_ptr(tok0, nt_all, e)` が「tok0 行目から nt_all 行」を要求するので、
タイルが 2 枚以上あると必ず `None` になること (既存のバグ。IQ 系でもタイル分割時に
踏む)。

## 3. t318c — `mixed ptr` を直して確定 → 採る

`pf_lm_mixed_ptr(tok0, nt, e)` に直し、デバイス経路の MoE 呼び出しを台帳の
`experts` 区間で包んだ (フォールバック側は内部で開くので片方だけ `other` に
混ざっていた)。

### 3.1 数字 (RESULT.md の median と、各 round の生値)

| cond | 3,826 cold (r1 / r2) | 14k cold (r1) |
| --- | ---: | ---: |
| b0t | 16.71 / 16.29 (med 16.50) | 36.34 |
| **kdev** | **13.36 / 13.07 (med 13.21)** | **30.10** |
| b1 | 15.85 / 15.04 (med 15.45) | 38.06 |

`VERDICT_DEPTH kdev vs b0t`: `c2000_cold decode -21.8% prefill -19.9%` /
`c8000_cold decode -17.1% prefill -17.2%` / `c2000_turn2 decode -7.4% prefill -17.5%` /
`c8000_turn2 decode -12.5% prefill -15.1%`。
b0t は先頭の偏りを含むので、末尾の b1 とも比べる: 3,826 は 13.21 vs 15.45 =
**-14.5%**、14k は 30.10 vs 38.06 = **-20.9%**。b0t/b1 の平均 (15.97 / 37.20)
に対して **-17.3% / -19.1%**。目標 (-15% 以上、他方が悪化しない) を満たす。

`decode ms/tok (median)` は kdev 20.6 / b0t 21.4 / b1 21.1 で、**warm short decode も
悪化していない** (-3.5%)。errors は全 COND 0。

### 3.2 台帳の差 (同じ run の b1_1 と kdev_1、実測)

| 項目 | b1 (3,826) | kdev (3,826) | b1 (14k) | kdev (14k) |
| --- | ---: | ---: | ---: | ---: |
| wall | 15,844 ms | 13,356 ms | 38,057 ms | 30,097 ms |
| D2H 合計 | 8,511 MB | **2,870 MB** | 30,611 MB | **9,944 MB** |
| H2D 合計 | 55,279 MB | 47,750 MB | 137,367 MB | 109,598 MB |
| `other` | 2,448 ms | 1,480 ms | 9,821 ms | 5,979 ms |
| `experts` | 11,825 ms | 10,682 ms | 22,685 ms | 19,651 ms |
| stage_copy | 9,445 ms | 8,904 ms | 11,713 ms | 10,805 ms |
| router / shared の行 | 64 / 253 ms | 無し (デバイス) | 221 / 903 ms | 無し (デバイス) |

消えたのは「層ごとの mixed 1 往復 (D2H) + ffn 1 往復 (H2D)」と、shared expert の
活性往復 (14k で 6.9 GB x 2)。expert 重みのストリーム量は変わらない
(3,826: 44,381 → 44,372 MB / 14k: 98,151 → 98,005 MB) = **同じバイトを同じ
カーネルで回したまま、活性の往復だけが消えた**。

## 4. 数値の扱い (md5 DIFF の理由)

t318c の md5 は 4 本とも DIFF: `c2000_cold d6cec0ff → e7103f57`、
`c2000_turn2 17278c2d → 8950a8e7`、`c8000_cold e6d464d8 → 1e4e87fb`、
`c8000_turn2 42ac2301 → 518c7115`。理由は 1 つで、**shared expert の寄与の
足し方**:

- フォールバック (ホスト経路): routed expert の加重和を GPU で閉じて D2H し、
  shared expert の寄与を**ホストで後から足す**。
- デバイス経路: shared の出力を `[ntok][e]` で渡し、加重和の**最後の 1 項**
  (k+1 項) として同じカーネルで足す (`dno_exp_pf_accum`)。

同じ値の f32 の足し算の順序が変わるだけ (項の集合は同一、枝刈りも順序の変更も
無い)。丸めの差が greedy の 1 トークンを入れ替え、以降の文が変わる。IQ3_S は
元からこのデバイス経路なので**この変更の影響を受けない** (`pf_expert_mode` は
`iq_expert_mode` を包含するため判定が変わらない。既定モデルの prefill に
`[pf-lm] dev MoE を諦めた` が出ていないことを tint_d6f914e97 のログで確認)。

## 5. わかったこと (次の担当者へ)

1. **UD の `other` は「K 系が dev MoE に乗っていない」ことだった**。型ゲートが
   IQ 専用のままだったので、309 で K カーネルを足した後も prefill の MoE は
   全層ホスト経路で、層ごとに mixed を丸ごと往復していた。prefill のホスト仕事を
   見るときは、まず `[pf-lm] 層 0: dev MoE を諦めた (理由: ...)` を探す。
2. **`pf_lm_mixed_ptr(tok0, nt_all, e)` はタイル分割で必ず落ちる** (既存バグ)。
   `nt` を渡すのが正しい。IQ 系でもタイル 2 枚以上で同じことが起きる
   (`DENNO_PF_LM_TILE` を小さくすると再現する)。
3. **pageable の同期コピーはこのホストで 1-3 GB/s**。pinned バウンスにすれば速く
   なるはずだが、**rayon のワーカーごとに確保させる形は駄目** (14k で +59%)。
   やるなら「1 本の専用バッファ + 単一スレッド」に限る。
4. 14k の残りは `experts` 19.7 s (GPU busy 16.4 s) と stage_copy 10.8 s。
   sync 3.8 s + up_wait 2.1 s が層境界の待ちで、ここが次の山。
5. 3,826 の stage_copy は 8.9-9.4 s / 44.4 GB = 4.7-5.0 GB/s で、probe は 5-6.9。
   ディスクの実効速度にはまだ 1-2 割の余地がある (推定)。

## 6. テスト

- `cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1`
  **1215 passed / 0 failed / 102 ignored** (int-auto ba85315 merge 後。merge 前は
  1209 / 0 / 102)
- `cargo test --release -p denno-engine --features cuda --test kernel_invariants`
  **97 passed / 0 failed** (同。merge で task 317 の 2 本が増えた)
- 追加: `cuda::task318_tests` 2 本 (`DENNO_PF_PIN_STAGE` の真理値表、副区間名と
  添字の 1 対 1)、`qwen4exp::lm_dev_tests::pf_lm_k_dev_truth_table`
  (`DENNO_PF_LM_K_DEV` の真理値表。既定 on)。
- 直した既存テスト: `kernel_invariants::t233_lm_steer_points_match_chunk_path`
  が `pf_lm_mixed_ptr(tok0, nt_all, e)` を文字列で要求していたので `nt` に直し、
  逆向き (壊れた形が戻らないこと) の assert を足した。

## 7. text (TXTD / TXT / TXTB、t318c の kdev、そのまま引用)

- TXTD (3,826): 「ご提示いただいた文書は、LLM推論エンジン（`denno-engine）に
  おける **Prefill（事前計算フェーズの性能最適化の歴史と教訓を詳細に記録した
  ものです。」(14k も同じ書き出しで、段落構成も同じ)
- TXT: `# B-Trees: A Comprehensive Explanation ... ## 1. What is a B-Tree?`
- TXTB: 「Rust の「所有権（Ownership）とライフタイム（Lifetime）は、Rust を
  学ぶ上で最も重要であり、かつ最も難しい概念です。」
- errors は全 COND 0。

## 8. GPU で確認してほしいこと

1. 既定モデル (swift15-abl-IQ3_S) のラウンドは**取れなかった** (3 ラウンドの上限を
   t318a/b/c で使い切った)。ゲートは IQ で判定が変わらないので数値経路は同一の
   はずだが、本番 8098 で短い decode と 3,826 prefill を 1 回ずつ取り、P312 の
   baseline (short decode 16.6-17.0、3,826 cold 2.95 s) と同じことを確認してほしい。
2. UD の 14k で `[pf-host]` の `experts_wall` が 0.0 になるのは、デバイス経路が
   台帳の `experts` 区間を層ループの中で開くため (`[q4ledger-pf]` の
   `experts` 行には出る)。表示の整理は次のタスクでよい。
3. 次の山は 14k の `experts` 19.7 s (GPU busy 16.4) と sync 3.8 s。

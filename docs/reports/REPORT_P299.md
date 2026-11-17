ADOPT: no t299a t299b

# REPORT P299 — IQ3_S decode: ホスト側に残っているのは起動と小さな台帳だけで、
# mix の正規化 1 本 (-0.22 ms/token) では 4% に届かない

branch: br299 = int-auto `7794d2dde` (br301 merge 後) から。br295 は int-auto に
未 merge だったが、**取り込んでいない** — br295 の中身は decode の expert 段に
副区間 `exp_call` を足す計器 1 つで、このタスクでは experts 段の内訳を
別の道具 (台帳の副区間を EXPERT 行でも出す・下の 3 節) で読むことにしたため。
最後に int-auto `02171944a` (br300 = PLE prefill) を merge した (衝突なし)。
**lab の 2 ラウンドは merge 前の dc19341b8 で測っている** — br300 の変更は
prefill の PLE 経路 (exp_pf.rs) で、decode の hc mix には掛からない
(テストは merge 後に 1193 / 95 で緑)。lab のラウンドは 2 本で止めた
(3 本目に賭けられる候補が残っていない。0 節の 3 行目)。
lab: t299a (SHA 857ee6a1f、5 COND) / t299b (dc19341b8、5 COND)。既定の挙動は
変えていない (足したのは既定 off のゲート 1 本と、台帳の副区間 1 行)。

## 0. 結論 (3 行)

1. **目標 (短い decode か 14k 後 decode を 4% 速く) は届かない。** 今回採れた
   唯一の実測は `DENNO_MIX_NORM_HOST` の **mix 行 -0.22 ms/token
   (3.51 -> 3.28、-6.3%、起動 -97/token)** で、token 全体では **約 -1.4%**。
   2 つの lab で 4 サンプルずつ取り、ばらつきは 1% 未満で一致している。
2. **upmix (`DENNO_DEC_MIX_UPMIX=2`) は 3 度目の棄却。** mix 行が
   3.51 -> 4.23〜4.39 ms/token (+0.75、gpu 待ち 2.50 -> 3.35) と**悪化**する。
   t216c の -2% は再現せず、t269b の +5.7% と同じ向き。
3. **残りの時間はほぼ全部 GPU の仕事か、GPU の完了待ち。** 14k の台帳は
   ssm 3.44 (GDN の完了待ち) / experts 4.39 (ミス CPU + GEMV 待ち) /
   mix 3.28 (うち gpu 2.15) / full 2.22 (sync 1.89) / loop 1.63 (shared の join) /
   router 1.08 (gpu 0.64) / head 0.74 (gpu 0.70)。ホストで消せるのは
   `sh_begin` 0.45 と起動の合計 1.3 ms だけで、4% (0.6〜0.8 ms) を
   1 個で賄える山が無い。次はカーネル (P295 の 6 節と同じ結論)。

## 1. 前提の測り直し (タスク 1: 現在の int-auto の per-stage)

`PROFILE=1` の台帳 (`[q4ledger]`、ms/token)。t299a `toss_1` / t299b `toss_1`
(どちらも同じ既定 env)。短い = 28 token の 2 本目 (128 tok)、14k = c8000_turn2。

| 区間 | 短い | 14k turn2 | 中身 (実測) |
| --- | --- | --- | --- |
| step | 15.0〜16.4 | 18.6 | |
| mix | 3.51〜3.54 | 3.52〜3.53 | gpu 2.50 / launch 0.55 (388 発) / sync 2.53 |
| router | 1.08 | 1.08 | gpu 0.64 / cpu 0.21 (top-k) / launch 0.07 |
| shared | 0.63〜0.68 | 0.67 | 全部ホスト。`sh_begin` 0.45〜0.48 |
| full | 1.60 | 2.22 | gpu 1.12 / 1.89 (= sync 待ち) |
| ssm | 3.38 | 3.44 | launch 0.23 / cpu 0.06、残りは**同期 D2H の待ち** |
| experts | 3.51 | 4.25〜4.39 | 全部 cpu 列 (ミス CPU + GEMV 待ち) |
| head | 0.73 | 0.74 | gpu 0.70 |
| loop | 1.87 (sync) | 1.63 (sync) | shared expert の side stream join (47 sync) |

- ssm の 3.4 ms は転送量ではなく**待ち**。`[dec-kprof-avg]` (t295a、band=3800) の
  ssm 窓は 108 us/層で、`op D2H ... host=82.4` (160 B の同期 D2H が GDN の完了を
  待つ) が大半。MEMORY / t245b の「GDN 3 カーネルの完了待ち = GPU 時間そのもの」
  と一致する。ホスト側の変更では動かない。
- experts 行の cpu 列の中身は**まだ読めていない**。3 節の計器を足したが、
  この decode 経路では副区間が 1 行も出なかった (下の 3 節)。
- `launches=960〜1000/token`、`syncs=206〜212/token`。起動のホスト時間は
  合計 1.3 ms/token (mix 0.42 + shared 0.25 + loop 0.26 + ssm 0.22 + full 0.11 +
  router 0.07)。CUDA graph は decode の途中にホスト側の分岐 (router の top-k、
  expert の LRU 参照、D2H を待つ点) が 200 回以上あるので、この形のままでは
  捕まえられない (P281 の 6 節と同じ判断。禁止リストの「decode MoE の CUDA
  graph」とは別物だが、同じ理由で届かない)。

## 2. 変えたこと (既定 off のゲート 1 本)

**`DENNO_MIX_NORM_HOST`**: hc mix の grouped RMS 正規化を H2D の pack と同じパスで
ホストで行い、`dno_mix_norm` (grid = hc = 4、256 スレッド) の起動を撃たない。

- 根拠 (実測): t295a の kprof で `dno_mix_norm` は **6.5 us/呼び出し**、動かすのは
  40 KB。mix は 97 呼び出し/token なので、起動 1 本と 6.5 us が 97 回ぶん消える。
  x は H2D する直前にホストにあり、gamma も同じ長さでホストにある。
- 実装: `mix_norm_host_into` (要素ごとの式は `dno_mix_norm` と同じ。二乗和だけ
  8 本の累積器) で正規化した x を pinned buffer へ書き、そのまま H2D する。
  既定経路の down+inject 版 (`dno_mix_down_bf16_inj`) は正規化済み x を
  入力と inject の両方に使うので、中身の差し替えだけで等価。
  `mix_norm_host_ready` が偽 (down+inject 版が選べない形) のときは従来どおり
  デバイスで正規化する (fail closed)。norm 重みの常駐 (指紋 + mutex) も
  その回は通さない。
- ホスト側のコスト (実測、`bench_mix_norm_host_into`、e=2560/hc=4):
  **1.81 us/call** (素の copy は 0.34 us)。差 1.5 us に対して GPU の 5 us が
  消えるので、差し引き **-5 us × 97 = -0.48 ms/token (推定)**。
- 実測 (lab): mix 行 **3.51 -> 3.28 ms/token (-0.23)**、gpu 2.50 -> 2.21、
  launch 388 -> 291 (**-97 = 1/呼び出し**)、sync 2.53 -> 2.23。
  token 全体では **-1.4% (推定、step のノイズ ±10% より小さい)**。
- 丸め: 二乗和の順序が変わる (逐次和 -> 8 本の累積器)。要素ごとの式は同じ。
  **lab の md5 は全 COND で SAME** (値の差は最下位ビット台)。

## 3. 計器 (1 行) と、それが読めなかったこと

`[q4ledger]` の副区間の印刷対象に `EXPERT` (7) を足した。decode の expert sum 経路は
内側で `enter(EXPERT)` を開くので、`exp_lookup` / `exp_miss` / `exp_sum` / `exp_pro`
は EXPERTS (5) ではなく EXPERT (7) に積まれる — はずだったが、**t299b でも 1 行も
出なかった** (`experts` 行は出る)。`[exp-iq-hit]` のミス CPU (3.06 ms/token) は
出ているので計器の呼び出し側は通っているのに、`SUBS` が 0 のまま = 実際に通って
いる経路が `moe_experts_iq_gpu_batched` / `_miss_up` のどちらでもない可能性が高い
(P295 の `exp_call` が 3 箇所で出なかったのと同じ症状)。次は
`DENNO_Q4_PROFILE` のときに 1 層 1 回だけ「どの関数を通ったか」を出すのが安い。

## 4. 数値の扱い

- 実測: 1 節の台帳 (t299a/t299b `toss_1`)、2 節の mix 行 (両 lab の
  各 4 サンプル)、`bench_mix_norm_host_into` の 1.81 us、lab の VERDICT。
- **推定**: 「token 全体で -1.4%」(mix 行の差 -0.23 ms を step 15〜16 ms で割った)。
  step そのものは同じ COND の 2 サンプルでも 15.0〜16.4 ms と動くので、
  **step の差では判定していない**。1 トークン内で閉じる台帳の行だけを使った。
- lab の decode median は今回もったいなかった: t299a は **base2 (toss と同じ env) が
  toss より +37%**、t299b は +15.9%。どちらも load_before は 1.9〜2.0 なのに、
  走っている間の load が 1.9 -> 6.4 まで動いた。**この床では 4% の採否は判定できない**
  (P295 と同じ結論)。採否は台帳の行で決めた。
- テキストは健全 (t299b `nh` の TXTD): 「ご提示いただいた文書は、LLM推論エンジン
  （c2000プロジェクト）における、**Prefill（事前計算）フェーズの性能最適化の歴史を
  記録した技術レポートです。」 md5 は全 COND で
  `a:65ededc3 b:f18c75fc c2000_cold:d4b6eefb c2000_turn2:8ff9345f
  c8000_cold:e2292eb6 c8000_turn2:cc2c0e8f` (= int-auto の参照と同じ)、errors 0。

## 5. 棄却したもの (測った)

- `DENNO_DEC_MIX_UPMIX=2` (up + hc_mixed の融合、既存ゲート): mix 行
  3.51 -> 4.23/4.39 (t299a)、+0.75 ms/token。gpu 待ちが 2.50 -> 3.35 に増える
  = 融合カーネル (grid=ceil(e/2)、1 ブロック 8 warp) が up + hc_mixed の 2 本
  (7.5 + 5.8 us) より遅い。**3 度目の棄却** (t216c -2% / t269b +5.7% / 今回 +21%)。
- `DENNO_DEC_MIX_DTHR=512` (down のスレッド数): mix 行 3.50 -> 3.50 (2 ラウンド
  とも同一)。効果なし。

## 6. テスト

- `cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1`
  → **1191 passed / 0 failed / 102 ignored** (merge 前)、
  **1193 passed / 0 failed / 102 ignored** (int-auto `02171944a` = br300 の
  PLE prefill を merge した後)。
- `cargo test --release -p denno-engine --features cuda --test kernel_invariants`
  → **95 passed / 0 failed** (merge 前後とも)。
- 追加: `mix_norm_host_truth_table` / `mix_norm_host_matches_reference_and_is_deterministic`
  (要素式が参照と 1e-5 以内・同じ入力で同じ値) / `bench_mix_norm_host_into` (ignored)。
- 既知の赤 (`qwen4exp_decode spec_batch_stops_at_eos`、`qwen4exp_params` の
  validate 2 件) は踏んでいない (この木では緑)。

## 7. 次のタスクへ

1. **`DENNO_MIX_NORM_HOST` は既定 off のまま残した。** -1.4% は本物だが単独では
   4% に届かない。次のタスクがこれを土台にしてよい (既定 on にするなら
   そのとき lab で 1 回取る)。
2. **残るホスト側の山は 2 つだけ**: (a) `shared` の `sh_begin` 0.45 ms/token
   (`shared_expert_begin` の 3 本の `weights_for_gguf` + 余分な
   `gguf_registry().lock()` + plan の作り直し。層ごとに plan / pin をキャッシュ
   すれば消える見込み。落とし穴は LRU の pin を握り続けること)、
   (b) router の 1.08 ms のうち GPU 待ち 0.64 以外 (起動 + top-k 0.21 + 同期の
   往復)。router の GEMV を mix の鎖の末尾 (hc_mixed の後) に載せて D2H を
   1 本にまとめると、層あたり同期 1 往復 (5〜6 us) が消える = -0.3 ms (推定 2%)。
3. **4% を狙うならカーネル** (P295 の 6 節と同じ)。実測の帯域は
   expert IQ GEMV 478 GB/s、mix down 682 / up 873 GB/s、full の q8_0 290 GB/s。
   14k の GPU は 11 ms/token なので、1.3 倍で 4% に届く。

## 8. lab の VERDICT 行 (生)

t299a (SHA 857ee6a1f、load_before 1.99、走行中の load 1.99 -> 6.42):
```
VERDICT nh vs toss OK decode +20.8% prefill +16.8% md5 SAME errors 0
VERDICT upm vs toss OK decode +8.7% prefill +3.2% md5 SAME errors 0
VERDICT nhupm vs toss OK decode +9.3% prefill +4.0% md5 SAME errors 0
VERDICT base2 vs toss OK decode +37.1% prefill +12.0% md5 SAME errors 0
VERDICT baseline toss errors 0 text ok
```
(base2 が toss より +37% = この回の decode median は使えない。mix 行は
toss 3.50/3.51、base2 3.51/3.45、nh 3.29/3.27、upm 4.23/4.14、
nhupm 3.86/3.84 と 1% 以内で揃っている。)

t299b (SHA dc19341b8、load_before 1.86、走行中の load 1.86 -> 6.4):
```
VERDICT nh vs toss OK decode -1.1% prefill -1.6% md5 SAME errors 0
VERDICT d512 vs toss OK decode +26.8% prefill +43.9% md5 SAME errors 0
VERDICT prof vs toss OK decode +14.4% prefill +4.1% md5 SAME errors 0
VERDICT base2 vs toss OK decode +15.9% prefill +17.1% md5 SAME errors 0
VERDICT baseline toss errors 0 text ok
```

# 数値緩和ポリシー (Numeric relaxation policy)

このドキュメントは「どの数値のずれまで許すか」と「ずれたときにアンカーをどう扱うか」を定める。
`s68`〜`s73` は **ビット一致** を設計条件として達成した。以降は高速化の余地が演算カーネル本体へ移ったため、
**決定論を維持したまま** 一部の緩和を許可する。

## 1. 原則

1. **決定論は絶対**。同一バイナリ・同一入力・同一シードで、プロセスを跨いでも出力は一致しなければならない。
2. 緩和は「許容誤差の範囲で実装を変えてよい」という許可であり、**誤差を測らずに commit してよい**という意味ではない。
3. 緩和クラスは commit メッセージと `docs/HISTORY` に明記する。
4. アンカー（`data/verify/anchors_baseline.txt`）は**検証装置の基準**であり、緩和を採用したときだけ、
   下記 §4 の手順で明示的に更新する。黙って書き換えない。

## 2. 許容クラス

| クラス | 内容 | 許容誤差 | 例 |
|---|---|---|---|
| **R1** | f32 縮約の順序変更・分割・融合 | 相対 1e-6 | 行並列化、ループ融合、ベクトル長の変更 |
| **R2** | int8/int4 GEMM の実装変更 | epilogue のみ 相対 1e-4 | タイル形状変更、`__dp4a` → `mma.sync`（int8 tensor core）、`cp.async` 二重バッファ |
| **R3** | attention の低精度化 | 絶対 1e-3 | softmax の f16 化、TF32 の使用（既承認） |
| **R4** | 超越関数の近似 | 絶対 1e-3 | `expf` の fast-math 版、sigmoid 近似 |

**R2 の注意**: int8 の積和は int32 で厳密（順序非依存）。したがって R2 で差が出るのは
dequant の `acc * ascale * scale` をどの順で加算するかだけであり、実際の差は 1e-6 級に収まるはず。
1e-4 を超える差が出た場合は、量子化スキームかデータ配置を意図せず変えている（＝設計ミスの可能性）。

## 3. 禁止

- **非決定論**: `atomicAdd` の順序依存、実行ごとに変わる削減、乱数、スケジューラ依存の分割。
- **量子化スキームの変更**（per-row scale → per-tile scale など）。別タスクとして設計・承認する。
- アンカーを測らずに commit すること。GPU 実測は `GPU_VERIFY.sh` / `ANCHOR_REBASE.sh` / `PF_BENCH.sh` 経由のみ。
- 検証を通すためにテストや許容値を緩めること（テスト側の変更は別 commit で理由を書く）。

## 4. 受け入れ条件

- **A1**: `cargo test --release -p denno-engine --features cuda --lib` と `-p denno-server --lib` が全 pass。
- **A2**: 決定論。`ANCHOR_REBASE.sh` の check モードで全アンカーが `det=OK`
  （サーバプロセスを再起動して 2 回測る）。
- **A3**: 性能。対象ワークロードで改善し、decode を 3% 超悪化させない。
- **A4**: 品質。アンカーの応答テキストが崩壊していない（繰り返し・空応答・言語混在・途中終了がない）。
- **A5**: 記録。`docs/HISTORY.ja.md` / `.en.md` に緩和クラス、旧 md5 → 新 md5、理由を書く。

## 5. アンカー再基準化の手順

```
# 1) まず check（書き換えない）
docker exec denno-gpu bash /work/denno-beta/ANCHOR_REBASE.sh
#    -> RESULT: DRIFT (deterministic) なら採用判断へ
#    -> RESULT: NONDETERMINISTIC なら却下（変更を戻す）

# 2) 採用する場合のみ baseline を更新（旧版は anchors_baseline.prev.txt に退避）
docker exec denno-gpu bash /work/denno-beta/ANCHOR_REBASE.sh --write

# 3) 新しい baseline で通常検証が通ることを確認
docker exec denno-gpu bash /work/denno-beta/GPU_VERIFY.sh   # RESULT: PASS / 全 MATCH

# 4) 性能測定
docker exec denno-gpu bash /work/denno-beta/PF_BENCH.sh 8095
```

- 再基準化は**その commit に含める**（`data/verify/` は `.gitignore` 対象なので、
  実際に commit するのは `docs/HISTORY.*` の記録と、必要なら `docs/` 内の基準値表）。
- 旧 baseline との差が「想定した種類の変化」かを必ず確認する。特に **出力トークン数** が
  大きく変わった場合は、数値誤差ではなくロジック退行を疑う。

## 6. 参照

- 検証装置: `GPU_VERIFY.sh`（アンカー + decode 速度）、`ANCHOR_REBASE.sh`（決定論 + 再基準化）、
  `PF_BENCH.sh`（prefill 実測）
- 履歴: `docs/HISTORY.ja.md` の各節（s68 以降は「ビット一致」を明記、緩和を入れた節はクラスを明記）
# Numeric relaxation policy

This document defines how much numerical drift is allowed, and what happens to the
verification anchors when drift occurs. Releases `s68`-`s73` achieved **bit-exact**
results by design. Now that the remaining headroom is in the kernels themselves,
some relaxation is allowed -- **without giving up determinism**.

## 1. Principles

1. **Determinism is non-negotiable.** Same binary, same input, same seed -> same output,
   including across process restarts.
2. A relaxation class is permission to change an implementation within a tolerance.
   It is not permission to commit without measuring.
3. The class is recorded in the commit message and in `docs/HISTORY`.
4. Anchors (`data/verify/anchors_baseline.txt`) are the reference of the verification
   harness. They are updated only when a relaxation is accepted, and only via the
   procedure in section 4.

## 2. Allowed classes

| Class | What | Tolerance | Examples |
|---|---|---|---|
| **R1** | reordering / splitting / fusing f32 reductions | rel 1e-6 | row parallelism, loop fusion, vector length |
| **R2** | different int8/int4 GEMM implementation | epilogue only, rel 1e-4 | tile shape, `__dp4a` -> `mma.sync`, `cp.async` double buffering |
| **R3** | lower precision attention | abs 1e-3 | f16 softmax, TF32 (already approved) |
| **R4** | approximated transcendentals | abs 1e-3 | fast-math `expf`, sigmoid approximations |

**R2 note**: int8 products summed in int32 are exact regardless of order. The only
possible difference is the order of `acc * ascale * scale` in the epilogue, i.e. 1e-6
territory. A difference above 1e-4 means the quantisation scheme or the data layout
was changed by accident, which is a design bug rather than drift.

## 3. Forbidden

- **Non-determinism**: order-dependent `atomicAdd`, run-dependent partitioning, RNG,
  scheduler-dependent splits.
- **Changing the quantisation scheme** (per-row -> per-tile scales, etc.). That needs a
  separate design and approval.
- Committing without measuring. GPU measurements go through `GPU_VERIFY.sh`,
  `ANCHOR_REBASE.sh` and `PF_BENCH.sh` only.
- Loosening tests or tolerances to make a change pass.

## 4. Acceptance

- **A1**: `cargo test --release -p denno-engine --features cuda --lib` and
  `-p denno-server --lib` pass.
- **A2**: Determinism. `ANCHOR_REBASE.sh` reports `det=OK` for every anchor
  (measured twice with a server restart in between).
- **A3**: Performance improves on the target workload; decode regresses by no more than 3%.
- **A4**: Quality. Anchor responses do not degenerate (no looping, empty output, language
  mixing, or truncation).
- **A5**: Recorded. `docs/HISTORY.ja.md` / `.en.md` state the relaxation class, the old and
  new md5, and why the change was needed.

## 5. Re-baselining procedure

```
# 1) check only, no writes
docker exec denno-gpu bash /work/denno-beta/ANCHOR_REBASE.sh
#    -> RESULT: DRIFT (deterministic)  : decide whether to accept
#    -> RESULT: NONDETERMINISTIC       : reject, revert the change

# 2) accept: update the baseline (old file kept as anchors_baseline.prev.txt)
docker exec denno-gpu bash /work/denno-beta/ANCHOR_REBASE.sh --write

# 3) normal verification must pass against the new baseline
docker exec denno-gpu bash /work/denno-beta/GPU_VERIFY.sh   # RESULT: PASS, all MATCH

# 4) performance
docker exec denno-gpu bash /work/denno-beta/PF_BENCH.sh 8095
```

Always check that the difference is the *kind* of change you intended. In particular, a
large change in **output token count** points at a logic regression rather than drift.

## 6. See also

- Harness: `GPU_VERIFY.sh` (anchors + decode speed), `ANCHOR_REBASE.sh` (determinism +
  re-baseline), `PF_BENCH.sh` (prefill measurement)
- History: `docs/HISTORY.ja.md` / `.en.md`
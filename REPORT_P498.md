ADOPT: pending t498a

# REPORT P498 — (draft; 計測中)

task 498 (dev3). branch `br498` = 開始時の `int-auto` **e0746f3** (= merge: br490)。
`br494` は REPORT_P494.md だけ (コード変更 0) で int-auto に未 merge だったが、
中身が report のみなので取り込んでいない (コードは同一)。

## 0. これから測るもの

- t498a: `model=flash-next-UD-Q4_K_XL` / `SCENARIO=depth8k` / `PROFILE=1`。
  COND: x0 (捨て) / b (baseline) / f0 (`DENNO_EXP_UP_FILL=0`) /
  f8 (`DENNO_EXP_UP_FILL_MULT=8`) / b2 (baseline 繰り返し)。
- 目的: 現 int-auto の per-stage profile を取り、UD で最大の段を確定する。
  UD の decode は expert のミス GEMV (ホスト) が支配的で、背景の
  `DENNO_EXP_UP_FILL` が mmap を読む量 (t442a: 728 本 x 3.14 MB / 48 tok =
  47 MB/token、dma 194 ms) がミス読みと DRAM を取り合っている疑いを A/B する。

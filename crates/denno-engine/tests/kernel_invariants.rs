//! s100 (P3-3 続き): CUDA カーネル (`.cu`) の静的不変条件テスト (インテグレーション版)。
//!
//! s75 の `kernel_source_shared_stride_and_block_shape_are_consistent` は
//! `src/cuda.rs` の `#[cfg(all(test, feature = "cuda"))] mod tests` に置かれていたため、
//! **既定ビルド (cuda feature 無し) では原理的に走らなかった**。同じ検査をこの
//! インテグレーションテストへ移設し、`.cu` を `include_str!` でテキストとして読む
//! だけの静的な検査にして feature ゲートを外した。GPU・cuda feature とも不要。
//!
//! 方針:
//!  - 対象は `kernels/*.cu` のテキストのみ。`.cu` / `.ptx` は変更しない。
//!  - 数値経路のカーネル起動は行わない (ホスト呼び出し・GPU 呼び出しとも無し)。
//!  - 検査は「ホスト側 (Rust) の定数・計算とカーネル側の前提が食い違うと落ちる」
//!    形に限る。値を変えたらこのテストが落ちるので、片側だけ直す事故を検出できる。
//!  - Rust 側のミラー定数は `src/cuda.rs` をテキストとして読み、`const NAME: TYPE = N;`
//!    を検査時に解析する。cuda.rs は `#[cfg(feature = "cuda")]` なので型としての
//!    参照はできないが、ミラーの食い違いを検出したい相手はまさにその定数なので
//!    ソース文字列を読む (定数名を変えたらこのテストが「見つからない」で落ちる)。

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// `.cu` テキストの読み取りヘルパ
// ---------------------------------------------------------------------------

const CUDA_RS: &str = include_str!("../src/cuda.rs");

/// `#define NAME VALUE` を取り出す。VALUE は他の define を参照し得るので、
/// 数値化は [`eval_define`] で依存順に展開する (s75 のテストと同じ方式)。
fn defines(src: &str) -> HashMap<String, String> {
    let mut map = HashMap::new();
    for line in src.lines() {
        let line = line.trim_start();
        let Some(rest) = line.strip_prefix("#define ") else {
            continue;
        };
        let rest = rest.trim();
        let Some((name, val)) = rest.split_once(char::is_whitespace) else {
            continue;
        };
        // 関数形式マクロ (`#define F(x) ...`) は名前の直後が `(` なので除外する。
        map.insert(name.to_string(), val.trim().to_string());
    }
    map
}

/// `#define NAME VALUE` を i64 として評価する。
fn eval_define(name: &str, d: &HashMap<String, String>) -> i64 {
    let raw = d
        .get(name)
        .unwrap_or_else(|| panic!("#define {name} が無い"));
    eval_expr(raw, d, 0)
}

/// トークン単位で define を展開してから四則演算・剰余・括弧を評価する。
fn eval_expr(expr: &str, d: &HashMap<String, String>, depth: u32) -> i64 {
    assert!(depth < 16, "define の循環参照: {expr}");
    let mut expanded = String::new();
    let b = expr.as_bytes();
    let mut i = 0;
    while i < b.len() {
        let c = b[i] as char;
        if c.is_ascii_alphabetic() || c == '_' {
            let start = i;
            while i < b.len() {
                let ch = b[i] as char;
                if ch.is_ascii_alphanumeric() || ch == '_' {
                    i += 1;
                } else {
                    break;
                }
            }
            let ident = &expr[start..i];
            let raw = d
                .get(ident)
                .unwrap_or_else(|| panic!("未定義の識別子 {ident} ({expr})"));
            expanded.push_str(&format!("({})", eval_expr(raw, d, depth + 1)));
        } else {
            expanded.push(c);
            i += 1;
        }
    }
    eval_int_expr(&expanded)
}

fn eval_int_expr(s: &str) -> i64 {
    let cs: Vec<char> = s.chars().filter(|c| !c.is_whitespace()).collect();
    let mut pos = 0usize;
    let v = parse_sum(&cs, &mut pos);
    assert_eq!(pos, cs.len(), "式を評価しきれない: {s}");
    v
}

fn parse_sum(cs: &[char], pos: &mut usize) -> i64 {
    let mut v = parse_prod(cs, pos);
    while *pos < cs.len() && (cs[*pos] == '+' || cs[*pos] == '-') {
        let op = cs[*pos];
        *pos += 1;
        let r = parse_prod(cs, pos);
        v = if op == '+' { v + r } else { v - r };
    }
    v
}

fn parse_prod(cs: &[char], pos: &mut usize) -> i64 {
    let mut v = parse_atom(cs, pos);
    while *pos < cs.len() && (cs[*pos] == '*' || cs[*pos] == '/' || cs[*pos] == '%') {
        let op = cs[*pos];
        *pos += 1;
        let r = parse_atom(cs, pos);
        v = match op {
            '*' => v * r,
            '/' => v / r,
            _ => v % r,
        };
    }
    v
}

fn parse_atom(cs: &[char], pos: &mut usize) -> i64 {
    if *pos < cs.len() && cs[*pos] == '(' {
        *pos += 1;
        let v = parse_sum(cs, pos);
        assert_eq!(cs.get(*pos), Some(&')'), "括弧が閉じていない");
        *pos += 1;
        return v;
    }
    let start = *pos;
    while *pos < cs.len() && cs[*pos].is_ascii_digit() {
        *pos += 1;
    }
    assert!(start != *pos, "数値が期待される位置: {cs:?}");
    cs[start..*pos].iter().collect::<String>().parse().unwrap()
}

/// `src/cuda.rs` の `const NAME: TYPE = VALUE;` を i64 として読む
/// (cuda feature 無しでは型として参照できないためのテキスト読み)。
fn rust_const(name: &str) -> i64 {
    let needle = format!("const {name}: ");
    let mut cursor = 0usize;
    // 先頭一致は `GEMM_TILE` が `GEMM_TILE_N` を誤ヒットし得るので、宣言の形で探す。
    while let Some(off) = CUDA_RS[cursor..].find(&needle) {
        let at = cursor + off;
        cursor = at + needle.len();
        // 直前が識別子の一部 (例: `_GEMM_TILE: `) なら別名なので飛ばす
        let prev = CUDA_RS[..at].chars().next_back();
        if let Some(p) = prev {
            if p.is_ascii_alphanumeric() || p == '_' {
                continue;
            }
        }
        let rest = &CUDA_RS[cursor..];
        let eq = rest
            .find('=')
            .unwrap_or_else(|| panic!("cuda.rs の const {name} に = が無い"));
        let tail = &rest[eq + 1..];
        let semi = tail
            .find(';')
            .unwrap_or_else(|| panic!("cuda.rs の const {name} に ; が無い"));
        let expr = tail[..semi].trim();
        // 数値リテラル (区切りの `_` 付き・算術式) のみを扱う。
        let cleaned: String = expr
            .chars()
            .filter(|c| c.is_ascii_digit() || "+-*/%()".contains(*c))
            .collect();
        assert!(
            !cleaned.is_empty(),
            "cuda.rs の const {name} の値 {expr:?} が数値式でない"
        );
        return eval_int_expr(&cleaned);
    }
    panic!("cuda.rs に const {name}: ... の宣言が見つからない");
}

/// `src` から `fn_start` で始まる `__global__` 関数本体 (波括弧の対応を取った範囲)
/// を切り出す。切り出しは「関数名の出現位置」から最初の `{` 以降を括弧深さで
/// 数える素朴な実装だが、.cu 側は文字列リテラル内に波括弧が無いので十分。
fn kernel_body<'a>(src: &'a str, fn_start: &str) -> &'a str {
    let at = src
        .find(fn_start)
        .unwrap_or_else(|| panic!("{fn_start} がカーネルソースに無い"));
    let mut depth = 0i32;
    let mut opened = false;
    for (i, ch) in src[at..].char_indices() {
        match ch {
            '{' => {
                depth += 1;
                opened = true;
            }
            '}' => {
                depth -= 1;
                if opened && depth == 0 {
                    return &src[at..at + i + 1];
                }
            }
            _ => {}
        }
    }
    panic!("{fn_start} の本体の閉じ波括弧が見つからない");
}

/// コメント行 (`//` 始まり) を落とした行の列。
#[allow(dead_code)]
fn code_lines<'a>(body: &'a str) -> impl Iterator<Item = (usize, &'a str)> {
    body.lines()
        .enumerate()
        .filter(|(_, l)| !l.trim_start().starts_with("//"))
}

/// ある式 `pat` が `body` 内で何回現れるか (コメント行は除外)。
#[allow(dead_code)]
fn count_in_code(body: &str, pat: &str) -> usize {
    code_lines(body).filter(|(_, l)| l.contains(pat)).count()
}

// ---------------------------------------------------------------------------
// kernels_int8.cu (s75 から移設)
// ---------------------------------------------------------------------------

/// s75 (不具合修正): `kernels_int8.cu` の共有メモリ / ブロック形状の不変条件を
/// **GPU 無しで** 検査する。s74 の 2 つの欠陥はどちらも CPU テストを素通りして
/// GPU 実機で初めて露見した (CUDA error 716 / 未書き込み出力) ので、原因を
/// ソースの静的解析で固定する。
///
///  1. 共有ステージの行ストライド `GEMM_SP` が 4 の倍数であること
///     (s74 は 33 = 奇数で、`*(int*)&sA[r][c]` が行ごとに未アラインになり
///      `Invalid __shared__ write` → sticky な error 716 を起こした)。
///  2. ブロック形状とタイル寸法が整合すること (s74 は block 16x4 = 64 thread で
///     32 行しか覆えず、タイル 128 行のうち 96 行が未書き込みだった)。
///  3. 共有配列への i32 アクセスのオフセットがすべて 4 の倍数であること。
///
/// .cu は `include_str!` で取り込むので、値を変えたらこのテストが落ちる。
#[test]
fn kernel_source_shared_stride_and_block_shape_are_consistent() {
    const SRC: &str = include_str!("../kernels/kernels_int8.cu");
    let d = defines(SRC);

    // ---- (1) 共有ステージの行ストライドは 4 の倍数 ----
    let tk = eval_define("GEMM_TK", &d);
    let sp = eval_define("GEMM_SP", &d);
    assert_eq!(
        sp % 4,
        0,
        "GEMM_SP={sp} が 4 の倍数でない: 共有配列への (int*) アクセスが未アラインになり \
         'Invalid __shared__ write' -> CUDA error 716 を起こす (s74 の 33 がこれ)"
    );
    assert_ne!(
        sp % 32,
        0,
        "GEMM_SP={sp} が 32 の倍数だと隣接 thread の 8 行差が同一 bank に並び conflict する"
    );
    assert_eq!(tk % 4, 0, "GEMM_TK={tk} は dp4a パックのため 4 の倍数であること");
    assert!(
        sp >= tk,
        "GEMM_SP={sp} は GEMM_TK={tk} 以上であること (列 tk-1 まで格納する)"
    );

    // ---- (2) ブロック形状はタイルを過不足なく覆う ----
    let tile = eval_define("GEMM_TILE", &d);
    let tile_n = eval_define("GEMM_TILE_N", &d);
    let bx = eval_define("GEMM_BX", &d);
    let by = eval_define("GEMM_BY", &d);
    assert_eq!(
        bx * 16,
        tile_n,
        "GEMM_BX * (1 thread の列数 16) = {} != GEMM_TILE_N {tile_n}",
        bx * 16
    );
    assert_eq!(
        by * 8,
        tile,
        "GEMM_BY * (1 thread の行数 8) = {} != GEMM_TILE {tile}: \
         ブロックがタイル全域を覆わず、未書き込みの出力行が残る (s74 の不具合)",
        by * 8
    );
    assert!(bx * by <= 1024, "ブロック {} スレッドは上限 1024 を超える", bx * by);

    // ---- Rust 側のミラー定数とカーネルの値が一致すること ----
    assert_eq!(
        rust_const("GEMM_TILE"),
        tile,
        "GEMM_TILE がカーネルと Rust で不一致"
    );
    assert_eq!(
        rust_const("GEMM_TILE_N"),
        tile_n,
        "GEMM_TILE_N がカーネルと Rust で不一致"
    );
    assert_eq!(rust_const("GEMM_BX"), bx, "GEMM_BX がカーネルと Rust で不一致");
    assert_eq!(rust_const("GEMM_BY"), by, "GEMM_BY がカーネルと Rust で不一致");
    // s74: 重みの行パディングはタイル幅 (GEMM_TILE_N) 単位であること
    // (gemm_pad_n(n) = ceil(n / GEMM_PAD_N) * GEMM_PAD_N がタイル境界を跨がない)。
    let pad_n = rust_const("GEMM_PAD_N");
    assert_eq!(pad_n, tile_n, "GEMM_PAD_N と GEMM_TILE_N が不一致");
    assert_eq!(
        pad_n % tile_n,
        0,
        "GEMM_PAD_N={pad_n} が GEMM_TILE_N={tile_n} の倍数でないと、gmy タイル分の行を用意できない"
    );

    // ---- 共有メモリ量が既定の 48KiB に収まること ----
    let smem = (tile + tile_n) * sp;
    assert!(
        smem <= 48 * 1024,
        "共有メモリ {smem}B が既定 48KiB を超える (動的確保か opt-in が必要)"
    );

    // ---- (3) 共有配列への i32 アクセスのオフセットはすべて 4 の倍数 ----
    let body = kernel_body(SRC, "void dno_gemm_i8");
    let mut checked = 0usize;
    for (lineno, line) in body.lines().enumerate() {
        if line.trim_start().starts_with("//") {
            continue;
        }
        for prefix in ["sA[", "sB["] {
            let mut search = 0usize;
            while let Some(off) = line[search..].find(prefix) {
                let at = search + off;
                // 直前に `(int*)` / `(const int*)` のキャストがあるものだけ対象
                // (`*(int*)&sA[..]` のように `&` が挟まるので取り除いて判定する)
                let lead = line[..at].trim_end().trim_end_matches('&').trim_end();
                let cast = lead.ends_with("(int*)") || lead.ends_with("(const int*)");
                if !cast {
                    search = at + prefix.len();
                    continue;
                }
                // `[行][列]` の列部分を取り出す
                let rest = &line[at..];
                let close = rest.find(']').expect("配列の [ が閉じていない");
                let after = &rest[close + 1..];
                let col_open = after.find('[').expect("列添字が無い");
                let col_close = after.find(']').expect("列添字が閉じていない");
                let col_expr = after[col_open + 1..col_close].trim();
                let col = eval_col_int(col_expr, &d, body);
                assert_eq!(
                    col % 4,
                    0,
                    "line {}: {prefix}..][{col_expr}] = バイトオフセット {col} が 4 の倍数でない \
                     (行 {lineno}): {}",
                    lineno + 1,
                    line.trim()
                );
                assert!(
                    col + 4 <= tk,
                    "列 {col} が GEMM_TK {tk} を超えて 4 バイト読む"
                );
                checked += 1;
                search = at + prefix.len();
            }
        }
    }
    assert!(
        checked >= 5,
        "共有配列の i32 アクセスを {checked} 件しか検査していない"
    );
    println!(
        "kernels_int8.cu: tile={tile}x{tile_n} tk={tk} sp={sp} block={bx}x{by} smem={smem}B \
         checked {checked} shared int accesses"
    );
}

/// 列添字の評価 (s75 の `eval_col` の整数部分のみを移植)。
///
/// このカーネルの列添字に現れる形だけを扱う:
///   - `#define` 定数 (数値・他 define 参照)
///   - `const int c = <式>;` のようなローカル定数
///   - `for (int x = 0; x < 8; x++)` のようなループ変数 (反復を全部評価し、
///     すべての反復で 4 の倍数であることを要求する)
/// 解決結果が 4 の倍数であることを要求する (= どの反復でもアライン)。
fn eval_col_int(expr: &str, d: &HashMap<String, String>, body: &str) -> i64 {
    for ident in idents_of(expr) {
        if d.contains_key(&ident) {
            continue; // define は eval_expr 側で展開する
        }
        match var_values(&ident, d, body) {
            (values, true) => {
                // ループ変数: 全反復で 4 の倍数であることを要求する
                let mut last = 0i64;
                for v in values {
                    let sub = subst_ident(expr, &ident, v);
                    let off = eval_col_int(&sub, d, body);
                    assert_eq!(
                        off % 4,
                        0,
                        "列添字 {expr} が {ident}={v} のとき {off} で 4 の倍数でない"
                    );
                    last = off;
                }
                return last;
            }
            (values, false) => {
                // 単一定数のローカル宣言: `const int c = x * 4;` のように右辺に
                // **別のループ変数**を含み得るので、宣言の右辺をこの関数で
                // 評価し直す (取り得る値の全反復で 4 の倍数であることを要求)。
                assert_eq!(values.len(), 1, "ローカル定数 {ident} の値が一意でない");
                return eval_col_int(&subst_ident(expr, &ident, values[0]), d, body);
            }
        }
    }
    eval_expr(expr, d, 0)
}

/// 変数の取り得る値。(values, is_loop)
///   `for (int v = A; v < B; v++)` / `v += S` -> (A..B の各値, true)
///   `const int c = <式>;` / `int c = <式>;`  -> (vec![評価値], false)
fn var_values(var: &str, d: &HashMap<String, String>, body: &str) -> (Vec<i64>, bool) {
    // ループ宣言を優先 (同じ名前が両方あることは無い)
    let loop_decl = format!("for (int {var} = ");
    if let Some(pos) = body.find(&loop_decl) {
        let tail = &body[pos + loop_decl.len()..];
        let sdemi = tail.find(';').expect("for の ; が無い");
        let from_expr = &tail[..sdemi];
        let rest = &tail[sdemi + 1..];
        let sdemi2 = rest.find(';').expect("for 条件の ; が無い");
        let cond = rest[..sdemi2].trim();
        let rest2 = &rest[sdemi2 + 1..];
        let inc = &rest2[..rest2.find(')').expect("for 増分の ) が無い")];
        let from = eval_col_int(from_expr, d, body);
        let lt = format!("{var} <");
        let to_expr = cond
            .strip_prefix(&lt)
            .unwrap_or_else(|| panic!("for 条件が `{var} <` でない: {cond:?}"));
        let to = eval_col_int(to_expr.trim(), d, body);
        let step = if inc.trim() == format!("{var}++") {
            1
        } else if let Some(r) = inc.trim().strip_prefix(&format!("{var} +=")) {
            eval_col_int(r.trim(), d, body)
        } else {
            panic!("for 増分が未対応: {inc:?}")
        };
        assert!(step > 0, "for 増分が正でない");
        let mut vals = Vec::new();
        let mut v = from;
        while v < to {
            vals.push(v);
            v += step;
        }
        assert!(!vals.is_empty(), "ループ {var} が 1 回も回らない");
        return (vals, true);
    }
    // ローカル定数 `const int c = <式>;` / `int c = <式>;`
    // 右辺は別の変数を参照し得る (`const int c = x * 4;`) ので同じ解決器で解く。
    for prefix in ["const int ", "int "] {
        let decl = format!("{prefix}{var} = ");
        if let Some(pos) = body.find(&decl) {
            let tail = &body[pos + decl.len()..];
            let semi = tail.find(';').expect("ローカル宣言の ; が無い");
            let v = eval_col_int(tail[..semi].trim(), d, body);
            return (vec![v], false);
        }
    }
    panic!("列添字の変数 {var} の宣言が見つからない");
}

/// 式に現れる識別子 (出現順・重複なし)
fn idents_of(expr: &str) -> Vec<String> {
    let b = expr.as_bytes();
    let mut out: Vec<String> = Vec::new();
    let mut i = 0;
    while i < b.len() {
        let c = b[i] as char;
        if c.is_ascii_alphabetic() || c == '_' {
            let s = i;
            while i < b.len() {
                let ch = b[i] as char;
                if ch.is_ascii_alphanumeric() || ch == '_' {
                    i += 1;
                } else {
                    break;
                }
            }
            let id = expr[s..i].to_string();
            if !out.contains(&id) {
                out.push(id);
            }
        } else {
            i += 1;
        }
    }
    out
}

/// 識別子を数値に置換 (識別子境界を保つ)
fn subst_ident(expr: &str, ident: &str, val: i64) -> String {
    let mut out = String::new();
    let b = expr.as_bytes();
    let mut i = 0;
    while i < b.len() {
        let c = b[i] as char;
        if c.is_ascii_alphabetic() || c == '_' {
            let s = i;
            while i < b.len() {
                let ch = b[i] as char;
                if ch.is_ascii_alphanumeric() || ch == '_' {
                    i += 1;
                } else {
                    break;
                }
            }
            if &expr[s..i] == ident {
                // 括弧で囲む: `x * 4` の x→(3) は `(3) * 4` になり、
                // 置換の前後関係で優先順位が変わる事故を防ぐ。
                out.push_str(&format!("({val})"));
            } else {
                out.push_str(&expr[s..i]);
            }
        } else {
            out.push(c);
            i += 1;
        }
    }
    out
}
// ---------------------------------------------------------------------------
// kernels/gdn.cu
// ---------------------------------------------------------------------------

const GDN_SRC: &str = include_str!("../kernels/gdn.cu");

/// 共有メモリの float レイアウトをカーネルと同じ規則 (行連結) で計算する。
/// gdn.cu のレイアウト: q_sh[dk] | k_sh[dk] | v_sh[dk] | o_sh[dv] | scalars[5]
fn gdn_smem_floats(dk: i64, dv: i64) -> i64 {
    3 * dk + dv + 5
}

/// s100 (P3-3): `kernels/gdn.cu` の共有メモリレイアウトとホスト側 (cuda.rs) の
/// 動的共有メモリ量 (`(3*dk + dv + 5) * 4` bytes) が一致すること。
///
/// s99 の試作は「ホストが小さすぎる smem を渡して共有領域外書きになる」型の
/// 不整合を静的検査で固定するものだったので、その不変条件を引き継ぐ。
#[test]
fn gdn_kernel_shared_layout_matches_host_smem() {
let d = defines(GDN_SRC);
    let _ = &d;

    // ---- (1) 各カーネルの共有配列オフセットが行連結になっていること ----
    // gdn.cu のレイアウト宣言 (q_sh が先頭・k/v は dk 刻み・o は 3*dk・sc は o の直後)。
    let layout: [(&str, &str); 4] = [
        ("float* q_sh = sh;", "q_sh は共有領域の先頭"),
        ("float* k_sh = sh + dk;", "k_sh は q_sh[dk] の直後"),
        ("float* v_sh = sh + 2 * dk;", "v_sh は k_sh[dk] の直後"),
        ("float* o_sh = sh + 3 * dk;", "o_sh は v_sh[dk] の直後"),
    ];
    for (decl, why) in layout {
        assert!(
            GDN_SRC.contains(decl),
            "gdn.cu の共有レイアウト宣言が変わった ({why}): {decl}"
        );
    }
    // o_sh の後のスカラー領域 (5 floats)。sc の基点が o_sh + dv であること。
    let sc_decl_3 = "float* sc = sh + 3 * dk + dv;";
    let sc_decl_2 = "float* sc = sh + 2 * dk;";
    assert_eq!(
        GDN_SRC.matches(sc_decl_3).count(),
        3,
        "gdn.cu の sc (q/k/v/o 付きレイアウト) が 3 カーネルで揃っていない"
    );
    assert_eq!(
        GDN_SRC.matches(sc_decl_2).count(),
        1,
        "gdn.cu の pass 1 (k/v のみ) の sc 基点が変わった"
    );

    // ---- (2) scalars[5] の割り当て ----
    assert_eq!(
        GDN_SRC
            .matches("sc[0]=q_inv sc[1]=k_inv sc[2]=eg sc[3]=beta sc[4]=o_inv")
            .count(),
        3,
        "gdn.cu の sc[] の意味 (index 割り当て) が変わった: \
         dno_gdn_step の sc[2]/sc[3]/sc[4] とホストの順序を確認すること"
    );
    assert_eq!(
        GDN_SRC.matches("sc[0]=k_inv sc[1]=eg sc[2]=beta").count(),
        1,
        "gdn.cu の pass 1 の sc[] 割り当てが変わった"
    );

    // ---- (3) ホスト側の smem 計算がカーネルと一致すること ----
    // cuda.rs は `let shmem = ((3 * dk + dv + 5) * 4) as u32;` を 2 箇所
    // (gdn_scan_impl / gdn_scan_batch) で、chunk 経路は `shmem_scan` の名前で
    // 同じ式を使う (下で個別に確認する)。
    let host_expr = "let shmem = ((3 * dk + dv + 5) * 4) as u32;";
    let n_host = CUDA_RS.matches(host_expr).count();
    assert_eq!(
        n_host, 2,
        "cuda.rs の gdn smem 計算 (`{host_expr}`) が {n_host} 箇所 (期待 2): \
         増減したなら本テストの期待値を更新すること"
    );
    // chunk 経路は別名 (`shmem_scan`) で同じ式。
    assert!(
        CUDA_RS.contains("let shmem_scan = ((3 * dk + dv + 5) * 4) as u32;"),
        "cuda.rs の chunk 経路の smem 計算が 3*dk + dv + 5 でない"
    );
    // pass 1 (k/v のみ) は `(2 * dk + 3) * 4` — カーネルの sc 基点と一致
    assert!(
        CUDA_RS.contains("let shmem_pre = ((2 * dk + 3) * 4) as u32;"),
        "cuda.rs の pass 1 smem 計算が k/v + scalars[3] と一致しない"
    );
    // dk == dv 前提のとき、共有領域は 4*dk + 5 floats。48KiB に収まること。
    for dk in [64i64, 128, 256] {
        let bytes = gdn_smem_floats(dk, dk) * 4;
        assert!(
            bytes <= 48 * 1024,
            "dk={dk} で共有メモリ {bytes}B が 48KiB を超える"
        );
        assert_eq!(
            gdn_smem_floats(dk, dk),
            4 * dk + 5,
            "共有領域の式が 4*dk+5 でない"
        );
    }

    // ---- (4) block = dk で、共有領域を全スレッドが書くこと ----
    assert!(
        CUDA_RS.contains(".launch_async(ctx.f_gdn, (nv as u32, 1, 1), (dk as u32, 1, 1), shmem"),
        "cuda.rs が gdn カーネルを grid=(nv) block=(dk) で起動していない"
    );
    assert!(
        count_in_code(GDN_SRC, "const int c = threadIdx.x;") >= 4,
        "gdn.cu の列担当が threadIdx.x でない"
    );
    assert!(
        GDN_SRC.contains("head_k_dim == head_v_dim"),
        "gdn.cu 先頭の dk == dv 前提コメントが消えた (dv のオフセットが成立しなくなる)"
    );
}
/// s100 (P3-3): `kernels/gdn.cu` のローカル列配列 (`Scol` / `Wcol` / `Mcol` /
/// `next`) の境界。これらは固定長 [256] のレジスタ配列で、ループは `dk` まで
/// 回る。dk を 256 より大きくするとスタックオーバーラン (不正メモリアクセス)
/// になるので、固定長とホスト側のゲート (dk <= 256) を突き合わせて固定する。
#[test]
fn gdn_kernel_column_arrays_fit_host_limits() {
    // カーネル側: 固定長 256 の列配列
    for (name, count) in [("Scol", 3usize), ("Wcol", 1), ("Mcol", 1), ("next", 1)] {
        let decl = format!("float {name}[256];");
        let n = GDN_SRC.matches(&decl).count();
        assert_eq!(
            n, count,
            "gdn.cu の `{decl}` が {n} 箇所 (期待 {count}): 列配列の長さ・追加を確認すること"
        );
    }
    // 書き込み側のループはすべて `i < dk` / `r < dk` / `r < dv` で、256 を直書きしていない
    assert_eq!(
        count_in_code(GDN_SRC, "[256]"),
        count_in_code(GDN_SRC, "float Scol[256];")
            + count_in_code(GDN_SRC, "float Wcol[256];")
            + count_in_code(GDN_SRC, "float Mcol[256];")
            + count_in_code(GDN_SRC, "float next[256];"),
        "gdn.cu に想定外の固定長 [256] がある (境界の再確認が必要)"
    );

    // ホスト側: dk / dv の上限ゲート (q35_chain_shapes_ok の形状条件)
    assert!(
        CUDA_RS.contains("dk == dv && d_conv == 4 && (1..=256).contains(&dk)"),
        "cuda.rs の gdn 形状ゲート (dk == dv / d_conv == 4 / dk <= 256) が変わった: \
         列配列 [256] を超えるとスタックを壊す"
    );
    assert!(
        GDN_SRC.contains("head_k_dim <= 256"),
        "gdn.cu 先頭の head_k_dim <= 256 前提コメントが消えた"
    );
    // 列配列への書き込み index は k 側ループ (dk) なので、dk == dv 前提で [256] に収まる。
    // ホストのゲート (dk <= 256) とカーネルの配列長 256 が同じ値であることを固定する。
    let array_len: i64 = 256;
    let host_limit: i64 = 256;
    assert_eq!(
        array_len, host_limit,
        "gdn.cu の列配列長 {array_len} とホストの dk <= 256 ゲートが食い違う"
    );
    // ゲートより配列が短いとスタックオーバーラン、長すぎると無駄なレジスタ消費。
    assert!(
        array_len >= 128,
        "dk=128 (qwen35 の実形状) が列配列長 {array_len} を超える"
    );
}

/// s100 (P3-3): `kernels/gdn.cu` の conv のチャネル index が全チャネルを被覆し、
/// 隣接カーネルと重複しないこと。
///
/// conv_dim = 2*nk*dk + nv*dv (q | k | v の連結)。スレッド c (0..dk) と head
/// (q/k は kh = h % nk・v は h) で
///   qch = kh*dk + c / kch = kd_total + kh*dk + c / vch = 2*kd_total + h*dv + c
/// を作る。h を 0..nv で動かしたとき q/k の被覆は [0, nk*dk) を **nv/nk 回**
/// 繰り返し、v は [2*kd_total, 2*kd_total + nv*dv) を 1 回ずつ覆う。
#[test]
fn gdn_kernel_conv_channel_index_covers_all_channels() {
    // ---- (1) index 式そのものが変わっていないこと ----
    assert!(
        GDN_SRC.contains("const int kh = h % nk;"),
        "gdn.cu の q/k head 写像 (h % nk・タイル repeat 規約) が変わった"
    );
    assert!(
        GDN_SRC.contains("const int qch = kh * dk + c;"),
        "gdn.cu の qch の式が変わった (kd_total の基点を確認)"
    );
    assert!(
        GDN_SRC.contains("const int kch = kd_total + kh * dk + c;"),
        "gdn.cu の kch の式が変わった (kd_total の基点を確認)"
    );
    assert!(
        GDN_SRC.contains("const int vch = 2 * kd_total + h * dv + c;"),
        "gdn.cu の vch の式が変わった (2*kd_total の基点を確認)"
    );

    // ---- (2) 全チャネル被覆 (9B / 27B の実形状 + 対称形) ----
    for &(nk, nv, dk, dv) in &[(16usize, 32usize, 128usize, 128usize),
                               (16, 48, 128, 128),
                               (16, 16, 128, 128),
                               (4, 8, 64, 64)] {
        let kd_total = nk * dk;
        let conv_dim = 2 * kd_total + nv * dv;
        let mut hits = vec![0usize; conv_dim];
        for h in 0..nv {
            let kh = h % nk;
            for c in 0..dk {
                let qch = kh * dk + c;
                let kch = kd_total + kh * dk + c;
                let vch = 2 * kd_total + h * dv + c;
                assert!(qch < conv_dim && kch < conv_dim && vch < conv_dim);
                hits[qch] += 1;
                hits[kch] += 1;
                hits[vch] += 1;
            }
        }
        // 全チャネルが少なくとも 1 回書かれる (未初期化チャネルが残らない)
        assert!(
            hits.iter().all(|&n| n > 0),
            "nk={nk} nv={nv} dk={dk}: 一度も書かれない conv チャネルがある"
        );
        // q/k は nv/nk 回・v は 1 回 (GQA の repeat 規約)
        let rep = nv / nk;
        assert_eq!(rep * nk, nv, "nv={nv} が nk={nk} の倍数でない (repeat 規約の前提)");
        for c in 0..kd_total {
            assert_eq!(hits[c], rep, "q/k チャネル {c} の書き込み回数が {rep} でない");
        }
        for c in 2 * kd_total..conv_dim {
            assert_eq!(hits[c], 1, "v チャネル {c} の書き込み回数が 1 でない");
        }
    }

    // ---- (3) conv 窓 (d_conv=4 / cs=3) が dk/dv と整合 ----
    assert!(
        GDN_SRC.contains("float wq[4], wk[4], wv[4];"),
        "gdn.cu の conv 重み窓が d_conv=4 でない (ホストの d_conv 前提と不一致)"
    );
    assert!(
        GDN_SRC.contains("float inq[3], ink[3], inv[3];"),
        "gdn.cu の conv 入力窓が cs=3 でない (ホストの d_conv-1 前提と不一致)"
    );
    // ローリング (新 state = 直近 d_conv-1 入力) が d_conv=4 版だけであること:
    // 3 要素のシフトが 3 本 (q/k/v) そろっていること。
    assert_eq!(
        count_in_code(GDN_SRC, "inq[0] = inq[1]; inq[1] = inq[2]; inq[2] = xq;"),
        3,
        "conv ウィンドウのローリング (3 変種) が揃っていない"
    );
}

// ---------------------------------------------------------------------------
// kernels/kernels_elem.cu
// ---------------------------------------------------------------------------

const ELEM_SRC: &str = include_str!("../kernels/kernels_elem.cu");

/// `kernels_elem.cu` の `dno_attention_prefill` の共有メモリ必要量 (float)。
/// Rust 側 `attn_pf_sh_f` と同じ式。
fn elem_attn_pf_sh_f(bq: i64, bk: i64, dh: i64) -> i64 {
    (bq + 2 * bk) * (dh + 1) + bq * bk + 2 * bq
}

/// s100 (P3-3): `PF_BQ` / `PF_BK` が Rust 側 `ATTN_PF_BQ` / `ATTN_PF_BK` と一致し、
/// 共有メモリのレイアウト・必要量がホストの計算と一致すること (s48/s70 で
/// ホスト側だけ古い値が残って error 700 になった事故の再発防止)。
#[test]
fn elem_attention_prefill_tile_matches_host() {
    let d = defines(ELEM_SRC);
    let bq = eval_define("PF_BQ", &d);
    let bk = eval_define("PF_BK", &d);
    assert_eq!(bq, 64, "PF_BQ が 64 でない (ホストの ATTN_PF_BQ と同時に更新すること)");
    assert_eq!(bk, 8, "PF_BK が 8 でない (ホストの ATTN_PF_BK と同時に更新すること)");
    assert_eq!(rust_const("ATTN_PF_BQ"), bq, "ATTN_PF_BQ がカーネルと Rust で不一致");
    assert_eq!(rust_const("ATTN_PF_BK"), bk, "ATTN_PF_BK がカーネルと Rust で不一致");

    // ---- 共有メモリの行連結レイアウト ----
    // q_sm[PF_BQ][dh+1] | k_sm[PF_BK][dh+1] | v_sm[PF_BK][dh+1]
    //   | s_sm[PF_BQ][PF_BK] | corr_sm[PF_BQ] | mnew_sm[PF_BQ]
    let layout: [(&str, &str); 6] = [
        ("float *q_sm = sh;", "先頭"),
        ("float *k_sm = sh + PF_BQ * (dh + 1);", "q_sm の直後"),
        ("float *v_sm = k_sm + PF_BK * (dh + 1);", "k_sm の直後"),
        ("float *s_sm = v_sm + PF_BK * (dh + 1);", "v_sm の直後"),
        ("float *corr_sm = s_sm + PF_BQ * PF_BK;", "s_sm の直後"),
        ("float *mnew_sm = corr_sm + PF_BQ;", "corr_sm の直後"),
    ];
    for (decl, where_) in layout {
        assert!(
            ELEM_SRC.contains(decl),
            "kernels_elem.cu の共有レイアウト宣言が変わった ({where_}): {decl}"
        );
    }
    // ホスト側の計算式が同じであること (attn_pf_sh_f)
    assert!(
        CUDA_RS.contains(
            "(ATTN_PF_BQ + 2 * ATTN_PF_BK) * (dh + 1) + ATTN_PF_BQ * ATTN_PF_BK + 2 * ATTN_PF_BQ"
        ),
        "cuda.rs の attn_pf_sh_f の式が変わった (カーネルのレイアウトと同時に更新すること)"
    );

    // ---- 必要量 (bytes) がホストの gate / 既定 48KiB と整合 ----
    let limit_default = rust_const("ATTN_PF_SMEM_DEFAULT");
    assert_eq!(limit_default, 48 * 1024, "ATTN_PF_SMEM_DEFAULT が 48KiB でない");
    // llama (dh=128) は既定 48KiB に収まり、qwen35 (dh=256) は opt-in が必要
    // (cuda.rs のコメントと同値であることを固定する)。
    assert_eq!(elem_attn_pf_sh_f(bq, bk, 128) * 4, 43_840, "dh=128 の smem が 43.8KB でない");
    assert_eq!(elem_attn_pf_sh_f(bq, bk, 256) * 4, 84_800, "dh=256 の smem が 84.8KB でない");
    assert!(elem_attn_pf_sh_f(bq, bk, 128) * 4 <= limit_default);
    assert!(elem_attn_pf_sh_f(bq, bk, 256) * 4 > limit_default);

    // block が PF_BQ 行 owner を持つ前提: dh >= PF_BQ (ホスト側ゲートと一致)
    assert!(
        CUDA_RS.contains("if dh < ATTN_PF_BQ || dh > 1024 {"),
        "cuda.rs の dh >= ATTN_PF_BQ ゲートが変わった"
    );
    assert!(
        ELEM_SRC.contains("dh >= PF_BQ"),
        "kernels_elem.cu の dh >= PF_BQ 前提コメントが消えた"
    );
}

/// s100 (P3-3): `kernels_elem.cu` の `float2` キャストのアラインメント。
///
/// `reinterpret_cast<const float2*>(&arr[idx])` は idx が **偶数** でないと
/// 8 バイト境界を跨ぎ、未定義動作 (実機では misaligned address) になる。
/// 対象は 2 つ:
///   - `corr_sm[rc * 2]`   : rc が整数なので常に偶数 (1 次元・歩幅 1)
///   - `s_sm[r0 * PF_BK + kk]` : kk が偶数かつ PF_BK が偶数なら常に偶数
#[test]
fn elem_float2_casts_are_aligned() {
    let d = defines(ELEM_SRC);
    let bk = eval_define("PF_BK", &d);
    let bq = eval_define("PF_BQ", &d);
    assert_eq!(bk % 2, 0, "PF_BK={bk} が偶数でない: s_sm の float2 荷載が未アラインになる");
    assert_eq!(bq % 2, 0, "PF_BQ={bq} が偶数でない: 2 行インターリーブが成り立たない");

    let body = kernel_body(ELEM_SRC, "void dno_attention_prefill");

    // ---- (1) corr_sm: `for (int rc = 0; rc < PF_BQ / 2; ++rc)` の rc*2 ----
    assert!(
        ELEM_SRC.contains("for (int rc = 0; rc < PF_BQ / 2; ++rc) {"),
        "corr_sm の 2 行ロードのループが変わった (rc*2 が偶数である前提)"
    );
    assert!(
        ELEM_SRC
            .contains("*reinterpret_cast<const float2 *>(&corr_sm[rc * 2]);"),
        "corr_sm の float2 荷載の式が変わった"
    );
    for rc in 0..(bq / 2) {
        assert_eq!((rc * 2) % 2, 0, "corr_sm の index が奇数");
        assert!(rc * 2 + 1 < bq, "corr_sm の float2 が [PF_BQ] を超えて読む");
    }

    // ---- (2) s_sm: `for (int kk = 0; kk < PF_BK; kk += 2)` の r0*PF_BK + kk ----
    assert!(
        ELEM_SRC.contains("for (int kk = 0; kk < PF_BK; kk += 2) {"),
        "s_sm の float2 荷載のループ (kk += 2) が変わった"
    );
    for pat in [
        "*reinterpret_cast<const float2 *>(&s_sm[r0 * PF_BK + kk]);",
        "*reinterpret_cast<const float2 *>(&s_sm[(r0 + 1) * PF_BK + kk]);",
    ] {
        assert!(ELEM_SRC.contains(pat), "s_sm の float2 荷載の式が変わった: {pat}");
    }
    for r0 in (0..bq).step_by(2) {
        for kk in (0..bk).step_by(2) {
            let idx = r0 * bk + kk;
            assert_eq!(idx % 2, 0, "s_sm の float2 index {idx} が奇数");
            assert!(idx + 1 < bq * bk, "s_sm の float2 が [PF_BQ*PF_BK] を超えて読む");
        }
    }

    // ---- (3) 動的 smem の基点は 16 バイト整列 (CUDA の保証) ----
    //      float2 (8B) の荷載は 8B 整列で足りるが、その前提コメントが残っていること。
    assert!(
        ELEM_SRC.contains("16B 整列"),
        "kernels_elem.cu の 16B 整列前提コメントが消えた"
    );
    // ---- (4) 共有領域は extern __shared__ 1 つのみ (静的 + 動的の二重取り禁止) ----
    // 例外: `__shared__ float s_inv;` (スカラー) は dno_add_rmsnorm 等の別カーネルの
    // もので、このカーネル本体の切り出しには入らない。ここで数えるのは本体だけ。
    let static_shm: Vec<&str> = code_lines(body)
        .map(|(_, l)| l)
        .filter(|l| l.contains("__shared__ float") && !l.contains("extern __shared__"))
        .collect();
    assert!(
        static_shm.is_empty(),
        "dno_attention_prefill に静的な __shared__ 宣言が混ざった \
         (extern __shared__ と二重に領域を取ることになる): {static_shm:?}"
    );
    assert!(
        count_in_code(body, "extern __shared__ float sh[];") == 1,
        "dno_attention_prefill の extern __shared__ 宣言が 1 つでない"
    );
}

/// s100 (P3-3): `kernels_elem.cu` の共有メモリ量と `reinterpret_cast` の
/// オフセット整合 (行ストライド dh+1 の pad が全参照で一貫していること)。
///
/// q_sm / k_sm / v_sm は `[行][dh+1]` で宣言され、本文の参照も
/// `*(dh + 1) + tid` の形で揃っていなければ、行の位置がずれて壊れる。
#[test]
fn elem_attention_prefill_smem_offsets_are_consistent() {
    let body = kernel_body(ELEM_SRC, "void dno_attention_prefill");

    // q_sm: 書き込み (dh スレッド) と読み出し (qa/qb) のストライド
    for pat in [
        "q_sm[r * (dh + 1) + tid]",
        "const float *qa = &q_sm[r0 * (dh + 1)];",
        "const float *qb = &q_sm[(r0 + 1) * (dh + 1)];",
    ] {
        assert!(body.contains(pat), "q_sm のストライド参照が変わった: {pat}");
    }
    // k_sm / v_sm: 書き込み・読み出しとも kk * (dh + 1) (+ tid)
    for pat in [
        "k_sm[kk * (dh + 1) + tid] = k_next[kk];",
        "v_sm[kk * (dh + 1) + tid] = v_next[kk];",
        "const float *ka = &k_sm[k0 * (dh + 1)];",
        "const float *kb = &k_sm[(k0 + 1) * (dh + 1)];",
        "const float v0 = v_sm[kk * (dh + 1) + tid];",
        "const float v1 = v_sm[(kk + 1) * (dh + 1) + tid];",
    ] {
        assert!(body.contains(pat), "k_sm/v_sm のストライド参照が変わった: {pat}");
    }
    // s_sm は pad 無し ([PF_BQ][PF_BK]) で、corr/mnew は 1 次元
    for pat in [
        "s_sm[r0 * PF_BK + k0]",
        "s_sm[r0 * PF_BK + k0 + 1]",
        "s_sm[(r0 + 1) * PF_BK + k0]",
        "s_sm[(r0 + 1) * PF_BK + k0 + 1]",
        "corr_sm[tid]",
        "mnew_sm[tid]",
    ] {
        assert!(body.contains(pat), "s_sm/corr_sm/mnew_sm の参照が変わった: {pat}");
    }
    // pad 幅 (dh + 1) 以外のストライドが混ざっていないこと: `+ 2)` 等の別 pad を作らない。
    assert_eq!(
        count_in_code(body, "(dh + 2)"),
        0,
        "共有メモリのストライドに (dh + 2) が混ざった (pad 幅が 2 種類ある)"
    );

    // ---- 共有領域の末尾 (mnew_sm + PF_BQ) がホストの必要量と一致 ----
    let d = defines(ELEM_SRC);
    let bq = eval_define("PF_BQ", &d);
    let bk = eval_define("PF_BK", &d);
    for dh in [128i64, 256] {
        let need = elem_attn_pf_sh_f(bq, bk, dh);
        // 末尾側から組み立てた値と一致するか (レイアウトの総和 = 各領域の和)
        let from_parts = bq * (dh + 1) + bk * (dh + 1) + bk * (dh + 1) + bq * bk + bq + bq;
        assert_eq!(need, from_parts, "共有レイアウトの総和が attn_pf_sh_f と不一致 (dh={dh})");
    }
}

/// s100 (P3-3) 既知の潜在不具合 (未修正・このテストでは固定しない): `dno_attention`
/// (decode 用) の block 間 reduction が共有メモリ `dh` floats に対して
/// `sh[threadIdx.x + off]` を `threadIdx.x < off` で読むため、block = dh では
/// threadIdx.x が最大 dh-1 になり **`sh[dh + off)` まで読む**。
/// 起動時の動的 smem は `dh * 4` バイト (cuda.rs) なので、off >= 1 の段で
/// 領域外を読む。ここでは「そう読む形になっていること」だけを検出して、
/// 修正されたらテストが落ちて気付けるようにする (REPORT に要修正として記載)。
#[test]
fn elem_decode_attention_reduction_reads_past_smem_known_issue() {
    let body = kernel_body(ELEM_SRC, "void dno_attention");
    // 現状の形 (領域外を読み得る)
    let cur = "for (int off = blockDim.x / 2; off > 0; off >>= 1) {\n        if (threadIdx.x < off) sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + off]);";
    let fixed_max = "off > 0; off >>= 1";
    let guarded = body.contains("threadIdx.x + off < blockDim.x");
    let reads_past = body.contains("sh[threadIdx.x + off]");
    assert!(
        reads_past,
        "dno_attention の reduction の形が変わった: 修正済みならこのテストを削除し REPORT を更新すること"
    );
    if !guarded {
        // 未修正であることを記録 (失敗させずに REPORT 用に明示する)
        println!(
            "KNOWN ISSUE (未修正): dno_attention の reduction が sh[threadIdx.x + off] を \
             読むが、動的 smem は dh floats・block も dh スレッドなので off >= 1 の段で \
             領域外を読む (cuda.rs の (dh * 4) と組み合わせて要修正)"
        );
        let _ = (cur, fixed_max);
    }
}

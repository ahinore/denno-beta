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
    // s239 (P36) が pass 1 の 4 レーン版 (`dno_gdn_chunk_pre_l4`) を足したので、
    // 同じ割り当てを持つカーネルは 2 本 (pre / pre_l4)。不変条件は「pass 1 の
    // sc[] は k_inv / eg / beta の 3 語」であって本数ではないため、期待値を
    // 2 に更新する (gdn.cu 自体は触っていない)。
    assert_eq!(
        GDN_SRC.matches("sc[0]=k_inv sc[1]=eg sc[2]=beta").count(),
        2,
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

// s104 (P3-3 横展開): smem の 4 バイトキャスト / 行列形参照の汎用静的検査
// ---------------------------------------------------------------------------
//
// ここから下は「カーネルごとに手書きした期待値」ではなく、`kernels/*.cu` の
// テキストを走査して **4 バイト要素型のキャスト** と **共有メモリの 2 次元参照**
// を機械的に列挙し、宛先オフセットのアラインメントと配列境界を検証する。
// s74/s75 の事故 (共有配列への `(int*)&sA[..][c]` が c 奇数で未アライン・
// 行ストライド 33 で全行ずれる) と同じクラスを、専用テストの無い箇所でも拾う。
//
// 方針:
//  - 走査対象はハードコードせず `kernels/` を列挙する (新規 .cu は自動で対象)。
//    ファイルが 1 つも見つからなければ**明示的に失敗**する (黙って skip しない)。
//  - コメント (`//`) は落とす。検査件数が想定下限を下回ったら失敗させ、
//    「緑だが何も見ていない」パーサの空回りを防ぐ。
//  - 値は `#define` と整数リテラルだけを解決する。裸の変数 (threadIdx.x 等) が
//    残った式は**未解決**として数え、アラインメントを主張しない (0 とみなさない)。
//  - 実カーネルの既知の違反は [`KNOWN_BAD_4B_CASTS`] に明示列挙する。.cu は
//    編集禁止 (PTX 再生成に nvcc が要る) なので、ここでは検出のみ行う。
//    表に無い違反が出たら失敗 (新規事故は必ず検出)。表の項目が消えた場合も
//    失敗 (修正されたのに表が残っている = テストが古い)。
//  - 一致判定は集合で行い (行番号を持たない)、表示だけソートする。HashMap の
//    反復順に依存させると、失敗の有無が実行ごとに変わる事故につながる。

/// smem オブジェクト (配列・スカラー・別名ポインタ) の宣言。
#[derive(Debug, Clone)]
struct SmemObj {
    /// 基点の名前 (別名ポインタなら元の配列名)
    name: String,
    /// 平坦化後の総要素数 (スカラーは 1・動的領域は i64::MAX)。
    /// 2 次元宣言は `先頭次元 * inner` をここに入れる (s105: 境界を要素数で判定する)。
    cap: i64,
    /// 1 行あたりの要素数 (2 次元宣言のときだけ Some)
    inner: Option<i64>,
    /// 平坦化オフセットに掛ける係数。signed char の共有配列は要素 = 1 バイトなので
    /// 平坦化した添字がそのままバイトオフセットになる (係数 1)。float の共有配列は
    /// 要素 = 4 バイトなので、`4 の倍数の要素境界` を判定する係数 4 を持つ。
    /// これが無いと `(float*)&sh[1]` (要素 4 番 = 整列済み) を未アラインと誤判定する。
    stride: i64,
}

/// 1 つの 4 バイトキャスト。
#[derive(Debug, Clone)]
struct CastSite {
    ty: String,
    /// 参照している名前 (配列名 or 別名ポインタ名)
    obj: String,
    /// 添字式 (配列なら `[..]` の中身・ポインタなら `p + ..` の形)
    idx: String,
}

fn strip_line_comments(src: &str) -> String {
    src.lines()
        .map(|l| match l.find("//") {
            Some(i) => &l[..i],
            None => l,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// 検出した smem 4 バイトキャスト (行番号は持たない: ファイル編集で揺れるため)。
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct SmemCast {
    file: String,
    ty: String,
    obj: String,
    off: i64,
}

#[derive(Debug, Default)]
struct Scan {
    /// アライン済み・範囲内と確認できたキャスト (添字が数値に解決できたもの)
    ok_casts: Vec<SmemCast>,
    /// 行添字が記号 (ループ変数・blockIdx 由来) でも、**列だけでアラインメントが
    /// 決まる**ため安全と言えるキャスト。行ストライドが 4 の倍数なら、行が何であっても
    /// 平坦化オフセットの 4 バイト整列は列だけで決まる。件数だけを持つ。
    symbolic_ok: usize,
    /// 行が記号でも列で安全と判定したキャストの内訳 (file/obj/ty)。
    /// 件数だけでは「特定ファイル・特定オブジェクトが検出されている」ことを
    /// 主張できないため、s105 で記録を追加した。
    symbolic_sites: Vec<SmemCast>,
    /// 記号行のキャストで、列側の条件を満たさなかったもの (本物の危険): (site, line, 行)
    bad_casts: Vec<(SmemCast, usize, String)>,
    /// 行も列も解決できなかったキャスト (file, line, 行テキスト)
    unresolved_casts: Vec<(String, usize, String)>,
    /// 平坦化できず検査できなかった 2 次元参照
    unchecked_rows: Vec<(String, usize, String)>,
    /// 検査した 2 次元参照 (キャスト内の `name[r][c]` を含む) の件数
    rows_checked: usize,
    /// 検査した 4 バイトキャストの総数
    casts_seen: usize,
    /// 2 次元宣言から作った行別名 (`a4` / `b4` 相当) の数
    aliases: usize,
}

/// `s[from..]` の `[` ... `]` の中身。
fn bracket_body(s: &str, from: usize) -> Option<&str> {
    let (open, close) = bracket_span(s, from)?;
    Some(&s[open + 1..close])
}

/// `s[from..]` で最初の `[` と対応する `]` の**バイト位置** (s の先頭からの絶対位置)。
/// `[` は open、`]` は close。返す値は `s[close + 1..]` が括弧の直後になる。
///
/// 旧実装は `bracket_body` の戻り値を「`[` から始まる文字列」と取り違えて
/// `find('[') + inner.len() + 2` で次の括弧を探しており、2 つ目の添字が必ず
/// 1 文字目で切れていた (`sA[tid][c]` の idx が `tid][c` に化ける原因)。
fn bracket_span(s: &str, from: usize) -> Option<(usize, usize)> {
    let rest = &s[from..];
    let open = rest.find('[')? + from;
    let mut depth = 0i32;
    for (i, ch) in s[open..].char_indices() {
        match ch {
            '[' => depth += 1,
            ']' => {
                depth -= 1;
                if depth == 0 {
                    return Some((open, open + i));
                }
            }
            _ => {}
        }
    }
    None
}

/// `名前[..]` の添字を `s[from..]` の先頭 (識別子の直後) から読む。
/// 2 次元なら `行][列` の形で返す。`[` が無ければ None。
fn read_index(s: &str, from: usize) -> Option<String> {
    let (open, close) = bracket_span(s, from)?;
    if !s[from..open].trim().is_empty() {
        return None; // `[` の前に別のトークンが挟まっている
    }
    let first = s[open + 1..close].trim().to_string();
    if let Some((o2, c2)) = bracket_span(s, close + 1) {
        if s[close + 1..o2].trim().is_empty() {
            return Some(format!("{first}][{}", s[o2 + 1..c2].trim()));
        }
    }
    Some(first)
}

/// 行から `名前[a][b]` をすべて取り出す (メンバ参照 `x.y[` は除く)。
fn find_double_index(line: &str) -> Vec<(String, String, String)> {
    let b = line.as_bytes();
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < b.len() {
        let c = b[i] as char;
        if !(c.is_ascii_alphabetic() || c == '_') {
            i += 1;
            continue;
        }
        let s = i;
        while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_') {
            i += 1;
        }
        let name = &line[s..i];
        if let Some(p) = line[..s].chars().next_back() {
            if p.is_ascii_alphanumeric() || p == '_' || p == '.' || p == '>' {
                continue;
            }
        }
        let Some((open1, close1)) = bracket_span(line, i) else {
            continue;
        };
        if !line[i..open1].trim().is_empty() {
            continue;
        }
        let first = line[open1 + 1..close1].to_string();
        let Some((open2, close2)) = bracket_span(line, close1 + 1) else {
            continue;
        };
        // 2 つ目の `[` が 1 つ目の `]` の直後にあるものだけを `name[a][b]` とみなす。
        if !line[close1 + 1..open2].trim().is_empty() {
            continue;
        }
        let second = line[open2 + 1..close2].to_string();
        out.push((
            name.to_string(),
            first.trim().to_string(),
            second.trim().to_string(),
        ));
        // 次の名前は 2 つ目の `]` の後ろから探す。
        i = close2 + 1;
    }
    out
}

/// 4 バイト要素型のキャストを 1 行から取り出す。
///   `*(int*)&sA[tid][c]`   / `*(const int*)&sA[..]` / `(*(const float2*)&x[i])`
///   `*(float*)(p + dk + 4)`
///   `*reinterpret_cast<const float2 *>(&s_sm[..])`   <- 第 2 文法 (s105 で追加)
/// C 形式のキャストは `*` の直後が `(`、その中が `型*` の形になっている。
/// `reinterpret_cast<T *>(&x)` は同じ 4 バイト幅のキャストなので同じ整列条件で
/// 検査する (kernels_elem.cu の float2 ロードがこれで、旧版は 1 件も拾えていなかった)。
fn find_casts4(line: &str) -> Vec<CastSite> {
    let mut out = c_style_casts4(line);
    out.extend(reinterpret_casts4(line));
    out
}

/// 型名から修飾子と空白を落とす。4 バイト幅の型かどうかは呼び出し側で判定する。
fn norm_ty(raw: &str) -> String {
    let mut ty = raw.trim().to_string();
    for p in ["const ", "volatile ", "__restrict__ ", "__restrict "] {
        ty = ty.replace(p, "");
    }
    ty.trim().to_string()
}

/// C 形式 (`*(int*)&..`) の 4 バイトキャスト。
fn c_style_casts4(line: &str) -> Vec<CastSite> {
    const TYS: [&str; 6] = ["float2", "int", "unsigned int", "unsigned", "float", "uint"];
    let star_paren = concat!("*", "(");
    let paren_star = concat!("*", ")");
    let b = line.as_bytes();
    let mut out = Vec::new();
    let mut search = 0usize;
    while let Some(off) = line[search..].find(star_paren) {
        let at = search + off;
        search = at + 1;
        // `*(` の直前が識別子や `)` なら別の演算子 (`a * (b)`) なので除外
        if let Some(p) = line[..at].chars().next_back() {
            if p.is_ascii_alphanumeric() || p == '_' || p == ')' {
                continue;
            }
        }
        let rest = &line[at + 2..];
        let close = match rest.find(paren_star) {
            Some(c) => c,
            None => continue,
        };
        let ty = norm_ty(&rest[..close]);
        if !TYS.contains(&ty.as_str()) {
            continue;
        }
        // `&名前[..]` / `(名前 + ..)` のどちらか
        let mut body = &rest[close + 2..];
        body = body.trim_start();
        if let Some(amp) = body.strip_prefix('&') {
            body = amp.trim_start();
        } else if body.starts_with('(') {
            body = body[1..].trim_start();
        } else {
            continue;
        }
        let name_len = body
            .char_indices()
            .take_while(|(_, c)| c.is_ascii_alphanumeric() || *c == '_')
            .count();
        if name_len == 0 {
            continue;
        }
        let name = &body[..name_len];
        let after = body[name_len..].trim_start();
        // 添字は元の行 `line` の絶対位置から読む (2 次元なら `行][列` の形で返る)。
        let after_at = line.len() - after.len();
        let idx = if after.starts_with('[') {
            match read_index(line, after_at) {
                Some(x) => x,
                None => continue,
            }
        } else if after.starts_with('+') {
            match after.find(')') {
                Some(c) => format!("{name} {}", &after[..c]),
                None => continue,
            }
        } else {
            continue;
        };
        let _ = b;
        out.push(CastSite {
            ty,
            obj: name.to_string(),
            idx,
        });
    }
    out
}

/// `reinterpret_cast<const float2 *>(&s_sm[..])` 形式の 4 バイトキャスト。
///
/// 型は `float` / `int` / `float2` など 4 バイト幅のものだけを対象にする
/// (`char*` など別幅の再解釈はここでは扱わない)。添字は C 形式と同じ形で取り出す。
fn reinterpret_casts4(line: &str) -> Vec<CastSite> {
    const TYS: [&str; 6] = ["float2", "int", "unsigned int", "unsigned", "float", "uint"];
    const TAG: &str = "reinterpret_cast<";
    let mut out = Vec::new();
    let mut search = 0usize;
    while let Some(off) = line[search..].find(TAG) {
        let at = search + off;
        search = at + TAG.len();
        let rest = &line[at + TAG.len()..];
        let Some(gt) = rest.find('>') else {
            continue;
        };
        let ty = norm_ty(rest[..gt].trim_end_matches('*').trim());
        if !TYS.contains(&ty.as_str()) {
            continue;
        }
        // `>` の直後は `(`。中身は `&名前[..]` か `名前 + ..`。
        let after_gt = rest[gt + 1..].trim_start();
        let Some(paren) = after_gt.strip_prefix('(') else {
            continue;
        };
        let body = paren.trim_start();
        let body = body.strip_prefix('&').unwrap_or(body).trim_start();
        let name_len = body
            .char_indices()
            .take_while(|(_, c)| c.is_ascii_alphanumeric() || *c == '_')
            .count();
        if name_len == 0 {
            continue;
        }
        let name = &body[..name_len];
        let after = body[name_len..].trim_start();
        let after_at = line.len() - after.len();
        let idx = if after.starts_with('[') {
            match read_index(line, after_at) {
                Some(x) => x,
                None => continue,
            }
        } else if after.starts_with('+') {
            match after.find(')') {
                Some(c) => format!("{name} {}", &after[..c]),
                None => continue,
            }
        } else {
            continue;
        };
        out.push(CastSite {
            ty,
            obj: name.to_string(),
            idx,
        });
    }
    out
}

/// `__shared__` / `extern __shared__` 宣言を読む。
///   `extern __shared__ float sh[];`     -> cap = i64::MAX (ホストがバイト数を渡す)
///   `__shared__ signed char sA[T][S];`  -> cap = T*S (総要素数), inner = S
///   `__shared__ float s_inv;`           -> cap = 1, inner = None
fn parse_smem_decl(line: &str, dynamic: bool, d: &HashMap<String, String>) -> Option<SmemObj> {
    let tag = if dynamic {
        "extern __shared__"
    } else {
        "__shared__"
    };
    let at = line.find(tag)?;
    if !dynamic && line[..at].contains("extern") {
        return None;
    }
    let rest = line[at + tag.len()..].trim();
    // 型は複数語になりうる (`signed char` 等)。名前は最後の語の先頭識別子で、
    // 次元はその語に空白無しで続く (`sA[TILE][SP];` / `sh[];`)。
    let words: Vec<&str> = rest.split_whitespace().collect();
    let ty_parts: String = words[..words.len().saturating_sub(1)].concat();
    if !(ty_parts.contains("float") || ty_parts.contains("int") || ty_parts.contains("char")) {
        return None;
    }
    let decl = match words.get(1..).and_then(|w| w.last()) {
        Some(d) if !d.is_empty() => *d,
        _ => return None,
    };
    let name_len = decl
        .char_indices()
        .take_while(|(_, c)| c.is_ascii_alphanumeric() || *c == '_')
        .count();
    if name_len == 0 {
        return None;
    }
    let name = decl[..name_len].to_string();
    let mut dims: Vec<String> = Vec::new();
    let mut rest_dims = &decl[name_len..];
    while let Some(inner) = bracket_body(rest_dims, 0) {
        dims.push(inner.trim().to_string());
        let used = rest_dims.find('[').unwrap() + inner.len() + 2;
        rest_dims = &rest_dims[used..];
    }
    // スカラー (`s_inv;`)、空次元の動的配列 (`sh[];`)、ホストがサイズを渡す
    // 動的領域は宣言長を上限にできない。
    // 要素 1 個のバイト数: `char`/`int8` は 1、それ以外 (float/int/float2) は 4。
    // 4 バイトキャストの整列は「flat * stride が 4 の倍数か」で判定する。
    let stride = if ty_parts.contains("char") && !ty_parts.ends_with("float") {
        1
    } else {
        4
    };
    if dynamic || dims.is_empty() || dims[0].is_empty() {
        return Some(SmemObj {
            name,
            cap: if dynamic || !dims.is_empty() { i64::MAX } else { 1 },
            inner: None,
            stride,
        });
    }
    let outer = eval(&dims[0], d)?;
    let inner = match dims.get(1) {
        Some(s) => Some(eval(s, d)?),
        None => None,
    };
    // 2 次元は平坦化後の**総要素数**で境界を判定する (s105: 旧実装は行数 `outer` を
    // 総要素数と取り違え、`sA[R][C]` の off = R*inner + C が正常でも違反扱いしていた)。
    // 総要素数・平坦化オフセットのどれも掛け算なので、飽和演算で overflow させない。
    let cap = outer.saturating_mul(inner.unwrap_or(1));
    Some(SmemObj {
        name,
        cap,
        inner,
        stride,
    })
}

/// 整数式の評価。`#define` と数値リテラルのみ (未解決は None)。
fn eval(expr: &str, d: &HashMap<String, String>) -> Option<i64> {
    let mut expanded = String::new();
    let b = expr.as_bytes();
    let mut i = 0usize;
    while i < b.len() {
        let c = b[i] as char;
        if c.is_ascii_alphabetic() || c == '_' {
            let s = i;
            while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_') {
                i += 1;
            }
            let ident = &expr[s..i];
            let raw = d.get(ident)?;
            let v = eval(raw, d)?;
            expanded.push_str(&format!("({v})"));
        } else {
            expanded.push(c);
            i += 1;
        }
    }
    let cleaned: String = expanded.chars().filter(|c| !c.is_whitespace()).collect();
    // 数字と四則演算・括弧以外が残っていたら未対応の式 (キャスト・関数呼び出し)
    if cleaned.is_empty()
        || !cleaned
            .chars()
            .all(|c| c.is_ascii_digit() || "+-*/%()".contains(c))
    {
        return None;
    }
    let cs: Vec<char> = cleaned.chars().collect();
    let mut pos = 0usize;
    let v = parse_sum(&cs, &mut pos);
    if pos == cs.len() {
        Some(v)
    } else {
        None
    }
}

/// `(int*)&sA[r][c]` の平坦化。2 次元宣言では `r * inner + c`。
/// 行と列は**別々に**解決する (行が記号でも列が数値ならアラインメントは列だけで決まる)。
///
/// 列が丸ごと数値に解決できなくても、列の式の中で**解決できる部分式**が
/// 「4 の倍数に固定されている」ことがある (`x * 4` / `ty * 8 + u` / `c0` など、
/// ループ変数を含む形)。その場合は列を「4 の倍数の錨 (anchor)」として扱い、
/// `unresolved` ではなく [`Flat::SymbolicRow`] に落とす。錨が無い (素の `tx + v` 等)
/// ときだけ未解決にする。
fn flatten(obj: &SmemObj, idx: &str, d: &HashMap<String, String>, body: &str) -> Flat {
    if let Some((a, b)) = idx.split_once("][") {
        // 行が数値、列も数値 -> 完全解決
        if let (Some(r), Some(c)) = (eval(a.trim(), d), eval(b.trim(), d)) {
            let off = r
                .saturating_mul(obj.inner.unwrap_or(1))
                .saturating_add(c);
            return Flat::Resolved(off);
        }
        // 行が記号 / 列が記号: 列の錨があれば列方向のアラインメントと範囲を検査できる。
        match col_anchor(b.trim(), d, body, obj.stride) {
            Some(col) => Flat::SymbolicRow {
                col,
                row_known: eval(a.trim(), d),
            },
            None => Flat::Unresolved,
        }
    } else if idx.starts_with('+') {
        // ポインタ形 (`p + EXPR`): 2 次元の行位置が分からないので解決扱いにしない
        match eval(idx.trim_start_matches('+').trim(), d) {
            Some(v) => Flat::Resolved(v),
            None => Flat::Unresolved,
        }
    } else {
        // 1 次元: `&sh[EXPR]` / スカラー。添字が記号でも列の錨 (4 の倍数) で
        // アラインメントだけは検査できる (`&s_sm[r0 * PF_BK + kk]` の `r0 * PF_BK`)。
        match eval(idx.trim(), d) {
            Some(v) => Flat::Resolved(v),
            None => match col_anchor(idx.trim(), d, body, obj.stride) {
                Some(col) => Flat::SymbolicRow {
                    col,
                    row_known: None,
                },
                None => Flat::Unresolved,
            },
        }
    }
}

/// 列の式から「4 の倍数に固定される項」を 1 つ取り出し、その値 (要素単位) を返す。
///   - 式全体が数値 -> その値
///   - `SUM` の加減算に分解し、各項を見る
///       - 数値項 -> その値
///       - `識別子 * N` / `N * 識別子` で N が 4 の倍数 -> N (ループ変数の倍数)
///       - 識別子単体 -> `#define` の値、またはローカル束縛 (下記) の錨
///   - 見つかった錨のうち**最大の 4 の倍数**を返す (列がどこまで進んでも
///     4 の倍数であることが保証される最小の保証幅)。
/// 錨が 4 の倍数でない、または見つからない場合は None。
fn col_anchor(expr: &str, d: &HashMap<String, String>, body: &str, stride: i64) -> Option<i64> {
    if let Some(v) = eval(expr, d) {
        return Some(v);
    }
    // 式を加減算で項に割る (括弧の外側の + / - だけ)
    let mut terms: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut depth = 0i32;
    for ch in expr.chars() {
        match ch {
            '(' | '[' => {
                depth += 1;
                cur.push(ch);
            }
            ')' | ']' => {
                depth -= 1;
                cur.push(ch);
            }
            '+' | '-' if depth == 0 => {
                terms.push(cur.clone());
                cur.clear();
            }
            c if c.is_whitespace() => {}
            c => cur.push(c),
        }
    }
    terms.push(cur);
    let mut best: Option<i64> = None;
    for t in terms {
        // 錨は「要素単位の刻み」。アラインメントを保証できるのは **バイトで 4 の倍数**
        // (要素数 * 要素幅) のときだけ。要素幅 4 の smem (float) なら要素 1 個が 4 バイト
        // なので要素数の刻み 1 でもよい (`rc * 2` の `2` は 8 バイト = 整列)。
        let cand = match term_anchor(&t, d, body, 0, stride) {
            Some(v) if v.saturating_mul(stride) % 4 == 0 => Some(v),
            _ => scaled_anchor(&t, d, body),
        };
        let Some(v) = cand.filter(|v| v.saturating_mul(stride) % 4 == 0) else {
            continue;
        };
        best = Some(best.map_or(v, |b: i64| b.max(v)));
    }
    // `(A + B) * N` の形 (括弧内が未解決でも N が 4 の倍数なら全体が 4 の倍数)。
    if best.is_none() {
        if let Some((_inner, n)) = split_mul_suffix(expr, d) {
            if n * stride % 4 == 0 {
                best = Some(n);
            }
        }
    }
    best
}

/// 項 1 つの錨。`x * 4` / `4 * x` / `N` (数値) / `define` 名 / ローカル束縛 を扱う。
fn term_anchor(term: &str, d: &HashMap<String, String>, body: &str, depth: u32, stride: i64) -> Option<i64> {
    let t = term.trim();
    if t.is_empty() {
        return Some(0); // 符号のみの項 (先頭の +)
    }
    if let Some(v) = eval(t, d) {
        return Some(v);
    }
    if let Some((_inner, n)) = split_mul_suffix(t, d) {
        // `識別子 * N`: N 自体が錨 (ループ刻み倍は scaled_anchor が見る)。
        return Some(n);
    }
    // 単独識別子: ローカル束縛 (`const int c = x * 4;` / `for (int c0 = 0; c0 < N; c0 += 4)`)
    // の錨を再帰的に見る。深さ制限で自己参照を止める。
    if depth < 4 && is_ident(t) {
        return local_anchor(t, d, body, depth + 1, stride);
    }
    None
}

/// `識別子 * N` (識別子 = ループ変数) で、1 周あたりの刻み `step * N`。
/// `&corr_sm[rc * 2]` のようにループ変数が定数倍されている形で、
/// 素の N が 4 の倍数でないとき (`rc * 2` で rc が 1 ずつ進む -> 2 要素 = 8 バイト) に使う。
fn scaled_anchor(term: &str, d: &HashMap<String, String>, body: &str) -> Option<i64> {
    let (_inner, n) = split_mul_suffix(term, d)?;
    let lhs = split_mul_ident(term)?;
    let step = loop_step(lhs, body)?;
    Some(step.abs().saturating_mul(n.abs()))
}

/// `識別子 * 何か` の左辺の識別子 (`(rc + 1) * PF_BK` の形は対象外)。
fn split_mul_ident(t: &str) -> Option<&str> {
    let (lhs, _) = t.split_once('*')?;
    let lhs = lhs.trim();
    if is_ident(lhs) {
        Some(lhs)
    } else {
        None
    }
}

/// ループ変数の刻み (`for (int v = A; v < B; v++)` -> 1 / `v += S` -> S)。
/// ループでない識別子は None。
fn loop_step(var: &str, body: &str) -> Option<i64> {
    let loop_decl = format!("for (int {var} = ");
    let pos = body.find(&loop_decl)?;
    let tail = &body[pos + loop_decl.len()..];
    let s1 = tail.find(';')?;
    let rest = &tail[s1 + 1..];
    let s2 = rest.find(';')?;
    let inc = rest[s2 + 1..rest[s2 + 1..].find(')')? + s2 + 1].trim();
    // 後置 (`v++`) と前置 (`++v`) の両方を受ける。
    if inc == format!("{var}++") || inc == format!("++{var}") {
        return Some(1);
    }
    if inc == format!("{var}--") || inc == format!("--{var}") {
        return Some(-1);
    }
    inc.strip_prefix(&format!("{var} +="))?
        .trim()
        .parse::<i64>()
        .ok()
}

/// 識別子のみからなる式か。
fn is_ident(t: &str) -> bool {
    !t.is_empty()
        && t.chars().next().is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && t.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// ローカル束縛の錨を求める。
///   `const int c = x * 4;`                 -> 右辺の錨 (4)
///   `for (int c0 = A; c0 < B; c0 += S)`    -> A と S の錨の最大値 (4)
///   `for (int r = 0; r < N; r++)`          -> A と 1 の錨 = 1 (4 の倍数でないので不適)
fn local_anchor(var: &str, d: &HashMap<String, String>, body: &str, depth: u32, stride: i64) -> Option<i64> {
    // ループ宣言を優先 (同じ名前が両方あることは無い)
    let loop_decl = format!("for (int {var} = ");
    if let Some(pos) = body.find(&loop_decl) {
        let tail = &body[pos + loop_decl.len()..];
        let s1 = tail.find(';')?;
        let from_expr = &tail[..s1];
        let rest = &tail[s1 + 1..];
        let s2 = rest.find(';')?;
        let inc = &rest[s2 + 1..rest[s2 + 1..].find(')')? + s2 + 1];
        let mut best = col_anchor(from_expr, d, body, stride)?;
        // 増分が `var++` / `var += S` でない形は扱わない (錨を断言できない)
        let inc = inc.trim();
        let step = if inc == format!("{var}++") {
            Some(1)
        } else {
            inc.strip_prefix(&format!("{var} +="))
                .and_then(|r| col_anchor(r, d, body, stride))
        };
        let step = step?;
        if step > best {
            best = step;
        }
        return Some(best);
    }
    // `const int c = <式>;` / `int c = <式>;`
    for prefix in ["const int ", "int ", "const "] {
        let decl = format!("{prefix}{var} = ");
        if let Some(pos) = body.find(&decl) {
            let tail = &body[pos + decl.len()..];
            let semi = tail.find(';')?;
            return col_anchor(tail[..semi].trim(), d, body, stride);
        }
    }
    None
}

/// `式 * N` / `N * 式` の N。`式` が数値に解決できれば (Some, N)、できなければ (None, N)。
/// N は数値リテラルか `#define` の値。乗算は**最後の** `*` で割る。
fn split_mul_suffix(expr: &str, d: &HashMap<String, String>) -> Option<(Option<i64>, i64)> {
    let t = expr.trim();
    // 外側の括弧を剥がす
    let t = t.strip_prefix('(').and_then(|x| x.strip_suffix(')')).unwrap_or(t);
    let (lhs, rhs) = t.rsplit_once('*')?;
    let lhs = lhs.trim();
    let rhs = rhs.trim();
    // 片側が数値 (リテラル or define) なら、もう片側は解決できなくてもよい
    // (`rc * 2` の `rc` はループ変数)。両側とも数値なら値そのもの。
    if let Some(n) = eval(rhs, d) {
        return Some((eval(lhs, d), n));
    }
    let n = eval(lhs, d)?;
    Some((eval(rhs, d), n))
}

/// [`flatten`] の結果。
#[derive(Debug, Clone, Copy)]
enum Flat {
    /// 平坦化オフセット (要素単位) が数値に解決できた
    Resolved(i64),
    /// 行が記号で未解決 (もしくは行は数値だが列が記号)。列の錨 `col` と、
    /// 行が数値に解決できたかどうか `row_known` を持つ。
    SymbolicRow {
        col: i64,
        row_known: Option<i64>,
    },
    /// 列もしくは 1 次元の添字が解決できなかった
    Unresolved,
}

/// `float *x = ...` の `x` を取り出す (左辺がポインタ宣言であること)。
fn ptr_name_before_eq(line: &str, eq: usize) -> Option<String> {
    let lhs = line[..eq].trim_end();
    let name_len = lhs
        .char_indices()
        .rev()
        .take_while(|(_, c)| c.is_ascii_alphanumeric() || *c == '_')
        .count();
    if name_len == 0 {
        return None;
    }
    let name = lhs[lhs.len() - name_len..].to_string();
    let head = lhs[..lhs.len() - name_len].trim_end();
    if !head.ends_with('*') || head.ends_with("**") {
        return None;
    }
    Some(name)
}

/// カーネル 1 本ぶんの smem 名表 (静的配列・動的配列・別名) を作る。
fn smem_table(
    src: &str,
    d: &HashMap<String, String>,
    aliases: &mut usize,
) -> HashMap<String, SmemObj> {
    let mut table: HashMap<String, SmemObj> = HashMap::new();
    // 動的領域はカーネルごとに 1 本。宣言順に「直近の extern 配列」を基点にする。
    let mut cur_dyn: Option<SmemObj> = None;
    for line in src.lines() {
        if let Some(obj) = parse_smem_decl(line, true, d) {
            cur_dyn = Some(obj.clone());
            table.insert(obj.name.clone(), obj);
            continue;
        }
        if let Some(obj) = parse_smem_decl(line, false, d) {
            table.insert(obj.name.clone(), obj);
            continue;
        }
        let Some(eq) = line.find('=') else {
            continue;
        };
        // 2 次元宣言の行そのものを指す別名 (`const int *row = &sA[r];`)
        if let Some((base_name, _, _)) = find_double_index(&line[..eq]).into_iter().next() {
            if let Some(parent) = table.get(&base_name).cloned() {
                if let (Some(inner), Some(alias)) = (parent.inner, ptr_name_before_eq(line, eq)) {
                    table.insert(
                        alias,
                        SmemObj {
                            name: parent.name.clone(),
                            // 別名は 1 行だけを指すので、上限は行幅・係数は要素のまま。
                            cap: inner,
                            inner: None,
                            stride: parent.stride,
                        },
                    );
                    *aliases += 1;
                }
            }
            continue;
        }
        // 1 次元ポインタ別名 (`float *x = sh + EXPR;`)
        let Some(ptr) = ptr_name_before_eq(line, eq) else {
            continue;
        };
        let rhs = line[eq + 1..].trim().trim_end_matches(';').trim();
        let head_len = rhs
            .char_indices()
            .take_while(|(_, c)| c.is_ascii_alphanumeric() || *c == '_')
            .count();
        let head = &rhs[..head_len];
        if head.is_empty() {
            continue;
        }
        let Some(base_obj) = table.get(head).cloned().or_else(|| cur_dyn.clone()) else {
            continue;
        };
        let tail = rhs[head_len..].trim();
        if !(tail.is_empty() || tail.starts_with('+') || tail.starts_with('-')) {
            continue;
        }
        // 別名の連鎖 (`float *s_sm = v_sm + PF_BK * (dh + 1);` の `v_sm` も別名) を
        // 1 回の走査で解決するため、走査を数回繰り返す。動的領域 (i64::MAX) では
        // オフセット式にカーネル引数 (`dh` 等) が入っていて評価できないことが多いが、
        // その場合でも「smem を指す名前」であることは確定なので cap = i64::MAX で登録する
        // (登録しないと `reinterpret_cast<const float2*>(&s_sm[..])` が丸ごと未検査になる)。
        let is_dyn = base_obj.cap == i64::MAX;
        let off = if tail.is_empty() {
            Some(0i64)
        } else {
            eval(tail.trim_start_matches('+').trim(), d)
        };
        let cap = match off {
            Some(off) if is_dyn => i64::MAX,
            Some(off) => (base_obj.cap - off).max(0),
            // 動的領域はホストがバイト数を渡すので、記号オフセットでも静的に上限は切れない。
            None if is_dyn => i64::MAX,
            None => continue, // 静的配列で記号オフセット: 上限が不明なので登録しない
        };
        table.insert(
            ptr,
            SmemObj {
                name: base_obj.name.clone(),
                cap,
                inner: None,
                stride: base_obj.stride,
            },
        );
    }
    table
}

/// カーネル群を走査する。
fn scan_kernels(sources: &[(String, String)]) -> Scan {
    let mut sc = Scan::default();
    for (file, raw) in sources {
        let src = strip_line_comments(raw);
        let d = defines(&src);
        let table = smem_table(&src, &d, &mut sc.aliases);

        // ---- 2 次元参照 (行列形) の列挙 ----
        // 数値に解決できるものだけをここで数える。記号添字の参照
        // (`sA[tid][c]` 等) は下の 4 バイトキャスト側の錨検査に一本化する
        // (s105: 実カーネルの 2 次元参照はキャストの内側にしか無いため、
        //  ここで未検査に積むと「実在しない構造を要求する」ことになる)。
        for (i, line) in src.lines().enumerate() {
            let ln = i + 1;
            if line.trim_start().contains("__shared__") {
                continue; // 宣言行は参照でない
            }
            for (name, a, b) in find_double_index(line) {
                let Some(obj) = table.get(&name) else {
                    continue;
                };
                let Some(inner) = obj.inner else {
                    continue;
                };
                if let (Some(row), Some(col)) = (eval(&a, &d), eval(&b, &d)) {
                    // s105: アライン違反は panic ではなく違反として記録する。
                    // 自己検証テストは「わざと壊した定義」を食わせるので、ここで
                    // 落ちると分類結果を検査できない。panic は行の範囲外だけに残す。
                    if col.saturating_mul(obj.stride) % 4 != 0 {
                        sc.bad_casts.push((
                            SmemCast {
                                file: file.clone(),
                                ty: "2d-ref".to_string(),
                                obj: obj.name.clone(),
                                off: col.saturating_mul(obj.stride),
                            },
                            ln,
                            format!(
                                "{name}[{a}][{b}] の列起点 {col} が未アライン \
                                 (行ストライド {inner} と列オフセットの整合を確認): {}",
                                line.trim()
                            ),
                        ));
                    }
                    assert!(
                        row >= 0 && col >= 0 && row < obj.cap / inner.max(1) && col < inner,
                        "{file}:{ln}: {name}[{row}][{col}] が宣言 [{}, {inner}] を越える: {}",
                        obj.cap / inner.max(1),
                        line.trim()
                    );
                    // s105: ここでは数えない。`*(int*)&sA[r][c]` は下のキャスト走査でも
                    // 数えるため、両方で足すと 1 参照が 2 件になる (自己検証テストの
                    // 期待値が実装の都合で揺れる)。数えるのはキャスト走査側に一本化。
                }
                // 記号行は下のキャスト走査で列の錨を検査する (unchecked には積まない)。
            }
        }

        // ---- 4 バイトキャストの列挙 ----
        for (i, line) in src.lines().enumerate() {
            let ln = i + 1;
            for c in find_casts4(line) {
                let Some(obj) = table.get(&c.obj) else {
                    continue; // smem 由来でない (引数・グローバル配列・レジスタ)
                };
                sc.casts_seen += 1;
                let site = |off: i64| SmemCast {
                    file: file.clone(),
                    ty: c.ty.clone(),
                    obj: obj.name.clone(),
                    off,
                };
                // バイトオフセット = 平坦化要素番号 * 要素幅 (char=1 / それ以外=4)
                match flatten(obj, &c.idx, &d, &src) {
                    Flat::Resolved(off) => {
                        if obj.inner.is_some() {
                            sc.rows_checked += 1; // 数値行の 2 次元参照も検査済みに数える
                        }
                        let bytes = off.saturating_mul(obj.stride);
                        let aligned = bytes % 4 == 0 && bytes >= 0;
                        let inbounds = off >= 0 && off + 4 <= obj.cap;
                        if aligned && inbounds {
                            sc.ok_casts.push(site(off));
                        } else {
                            sc.bad_casts.push((site(off), ln, line.trim().to_string()));
                        }
                    }
                    // 行が記号 (ループ変数・blockIdx 由来) でも、行ストライドが 4 の倍数なら
                    // 平坦化オフセットの 4 バイト整列は列だけで決まる。
                    // 条件: 行ストライドが 4 の倍数 / 列の錨 + 4 が行幅以内 /
                    //       列の錨のバイト位置が 4 の倍数。
                    // 行の上限は検査しない (ループ上下限はカーネルごとの別の不変条件。
                    // ここで見るのはアラインメントと列方向の境界だけ。REPORT に残件として記載)。
                    Flat::SymbolicRow { col, row_known } => {
                        sc.rows_checked += 1;
                        let col_bytes = col.saturating_mul(obj.stride);
                        let col_aligned = col >= 0 && col_bytes % 4 == 0;
                        // 2 次元なら行ストライドと列方向の境界も検査する。1 次元
                        // (動的 smem の別名など) は行の概念が無いのでアラインメントだけ。
                        let (row_ok, col_inbounds, why) = match obj.inner {
                            Some(inner) => (
                                (inner * obj.stride) % 4 == 0,
                                col + 4 <= inner,
                                format!(
                                    "行 {:?} / 行ストライド {} 要素 * {}B = {}B / 行幅 {inner}",
                                    row_known,
                                    inner,
                                    obj.stride,
                                    inner * obj.stride
                                ),
                            ),
                            None => (
                                true,
                                true,
                                "1 次元 (上限はホストが渡すバイト数なのでアラインのみ検査)"
                                    .to_string(),
                            ),
                        };
                        if row_ok && col_aligned && col_inbounds {
                            sc.symbolic_ok += 1;
                            sc.symbolic_sites.push(site(col_bytes));
                        } else {
                            sc.bad_casts.push((
                                site(col_bytes),
                                ln,
                                format!("{} [錨 {col} ({}B) / {why}]", line.trim(), col_bytes),
                            ));
                        }
                    }
                    Flat::Unresolved => sc
                        .unresolved_casts
                        .push((file.clone(), ln, line.trim().to_string())),
                }
            }
        }
    }
    sc
}

// ---------------------------------------------------------------------------
// 既知の違反 (未修正・.cu 編集禁止のため検出のみ)
// ---------------------------------------------------------------------------

/// 共有メモリ上の 4 バイトキャストについて、s104 の走査で見つかった既知違反。
/// `.cu` は PTX 再生成に nvcc が要るためこのコンテナでは直せない。ここに
/// 載っているものは失敗にせず、載っていないものは失敗させる。
/// 修正したら該当行を消すこと (残っているとテストが落ちる = 表の鮮度を強制)。
///
/// s104 時点では **空** (走査した 3 本すべてで、smem への 4 バイトキャストは
/// アライン済み・範囲内だった)。検出器の既知の粗さ: `gdn.cu` の
/// `s_col = (float*)ssm_in + hbase * dv + c;` は **グローバルメモリ** への
/// キャストだが、別名表の作り方によっては「直近の `extern __shared__` 宣言を
/// 基点」とみなして smem 扱いしうる。今回は `ssm_in` が引数 (smem 宣言でない)
/// なので別名表に入らず、対象外のままになっている。この粗さは REPORT_NEXT.md
/// の残件に記載した。
const KNOWN_BAD_4B_CASTS: &[(&str, &str, &str, i64)] = &[];

fn known_bad() -> Vec<SmemCast> {
    KNOWN_BAD_4B_CASTS
        .iter()
        .map(|(file, ty, obj, off)| SmemCast {
            file: (*file).to_string(),
            ty: (*ty).to_string(),
            obj: (*obj).to_string(),
            off: *off,
        })
        .collect()
}

/// `kernels/*.cu` を列挙する。見つからなければ失敗 (黙って skip しない)。
fn kernel_sources() -> Vec<(String, String)> {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels");
    assert!(
        dir.is_dir(),
        "カーネルディレクトリが見つからない: {}",
        dir.display()
    );
    let mut out: Vec<(String, String)> = Vec::new();
    for e in std::fs::read_dir(&dir).expect("kernels/ を読めない") {
        let p = e.expect("dir entry").path();
        if p.extension().and_then(|s| s.to_str()) == Some("cu") {
            let name = p.file_name().unwrap().to_string_lossy().to_string();
            let text = std::fs::read_to_string(&p).expect("read .cu");
            out.push((name, text));
        }
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    assert!(
        out.len() >= 3,
        "kernels/*.cu が {} 件しか見つからない (最少は gdn.cu / kernels_elem.cu / \
         kernels_int8.cu の 3 本)",
        out.len()
    );
    for want in ["gdn.cu", "kernels_elem.cu", "kernels_int8.cu"] {
        assert!(
            out.iter().any(|(n, _)| n == want),
            "{want} が kernels/ に無い (ファイル名を変えたら本テストを更新すること)"
        );
    }
    out
}

/// s104 (P3-3 横展開): `kernels/*.cu` の共有メモリへの 4 バイトキャストの宛先が
/// 4 の倍数で、宣言長を越えないこと。GPU 無し・cuda feature 無しで走る
/// (ファイルをテキストとして読むだけ)。
///
/// s74/s75 の事故は「CPU テストをすべて素通りして GPU 実機で初めて露呈」した。
/// ここではその類 (共有配列への int/float キャストの未アライン・範囲外) を、
/// 専用テストの無いカーネルも含めて機械的に検査する。
#[test]
fn kernel_source_4byte_smem_casts_are_aligned_on_all_kernels() {
    let sources = kernel_sources();
    let sc = scan_kernels(&sources);

    // ---- 既知の違反は表に無いものだけを失敗させる ----
    let known = known_bad();
    let mut unexpected: Vec<String> = sc
        .bad_casts
        .iter()
        .filter(|(s, _, _)| !known.contains(s))
        .map(|(s, ln, t)| format!("{}:{ln} `{}` {}[{}] :: {t}", s.file, s.ty, s.obj, s.off))
        .collect();
    unexpected.sort();
    assert!(
        unexpected.is_empty(),
        "共有メモリの 4 バイトキャストで未アライン/範囲外の宛先が {} 件ある \
         (s74 と同じクラス・.cu は編集禁止なので REPORT でメインに上げること):\n{}",
        unexpected.len(),
        unexpected.join("\n")
    );

    // ---- 表の項目が今も実際に違反として現れていること (表の鮮度) ----
    for k in &known {
        assert!(
            sc.bad_casts.iter().any(|(s, _, _)| s == k),
            "既知違反 {} `{}` {}[{}] がもう検出されない: 修正済みなら \
             KNOWN_BAD_4B_CASTS から消すこと",
            k.file,
            k.ty,
            k.obj,
            k.off
        );
    }

    // ---- パーサの空回り防止 ----
    // s105: 「合計 >= N 件」という件数のしきい値ではなく、**実在する具体的な
    // キャストが検出されていること**で担保する (`kernels_int8.cu` の
    // `*(int*)&sA[..][c]` は 6 箇所のみで、`kernels_elem.cu` の float2 は
    // `reinterpret_cast` 文法。件数の下限を実測に合わせずに置くと、退行ではなく
    // 見積もりの誤りで赤になる)。特定ファイル・特定オブジェクト・特定型の
    // 存在を要求する方が退行検出力が強い。
    assert!(
        sc.casts_seen >= 6,
        "4 バイトキャストを {} 件しか検出していない (kernels_int8.cu だけで 6 件あるはず)",
        sc.casts_seen
    );
    for (obj, ty) in [("sA", "int"), ("sB", "int")] {
        assert!(
            sc.ok_casts
                .iter()
                .chain(sc.symbolic_sites.iter())
                .any(|c| c.file == "kernels_int8.cu" && c.obj == obj && c.ty == ty),
            "kernels_int8.cu の `*({ty}*)&{obj}[..][..]` を 1 件も検出していない"
        );
    }
    assert!(
        sc.symbolic_ok >= 1,
        "行が記号のキャストを 1 件も「安全」と判定していない (kernels_int8.cu は 6 件とも行が \
         ループ変数/blockIdx 由来)"
    );
    // ---- 第 2 文法 (`reinterpret_cast<...>`) のパーサ検証 ----
    // 実カーネルの float2 キャストは動的 smem の別名 (`s_sm = v_sm + ..`) 越しで
    // 列も記号 (`rc * 2`) なので、走査器は列を数値に解決できず「未解決」に回す
    // (= アライン判定の対象外・REPORT に残件として記載)。ここでは走査結果ではなく
    // **パーサがその行を読めていること**を直接確かめる。
    let elem_src = sources
        .iter()
        .find(|(f, _)| f == "kernels_elem.cu")
        .map(|(_, src)| src.clone())
        .expect("kernels_elem.cu が見つからない");
    let mut parsed: Vec<(String, String)> = Vec::new();
    for line in elem_src.lines() {
        if line.contains("reinterpret_cast<const float2") {
            for c in find_casts4(line) {
                parsed.push((c.ty, c.obj));
            }
        }
    }
    for obj in ["s_sm", "corr_sm"] {
        assert!(
            parsed.iter().any(|(ty, o)| ty == "float2" && o == obj),
            "kernels_elem.cu の `reinterpret_cast<const float2*>(&{obj}[..])` を \
             パーサが 1 件も読めていない (第 2 文法が壊れている): {parsed:?}"
        );
    }
    assert!(
        sc.unresolved_casts.is_empty(),
        "キャストの列オフセットを解決できなかった箇所が {} 件 (列が未解決の 4 バイト\
         キャストは本物の危険なので 0 件を要求する):\n{}",
        sc.unresolved_casts.len(),
        sc.unresolved_casts
            .iter()
            .map(|(f, l, t)| format!("{f}:{l} {t}"))
            .collect::<Vec<_>>()
            .join("\n")
    );

    println!(
        "smem 4-byte casts: {} seen ({} numeric-ok, {} symbolic-row-ok, {} known-bad, \
         {} unresolved); {} double-index refs checked; {} row aliases",
        sc.casts_seen,
        sc.ok_casts.len(),
        sc.symbolic_ok,
        known.len(),
        sc.unresolved_casts.len(),
        sc.rows_checked,
        sc.aliases
    );
    }

/// s104 (P3-3 横展開): 2 次元共有配列の行ストライドと列オフセットの整合。
/// `sA[r][c]` の形で書かれた参照 (行別名経由を含む) の列起点が 4 の倍数で、
/// 宣言の行数・行幅を越えないことを検査する。
///
/// s105: 実カーネルの `name[r][c]` は **キャストの内側にしか無く**、行はすべて
/// ループ変数か blockIdx 由来の記号である (`kernels_int8.cu` の 6 箇所)。
/// `const int *row = &sA[r];` 形式の行別名は現存しないので、aliases の下限は要求せず
/// (0 件であることを固定する)、2 次元検査は「キャスト内の `name[row][col]` を
/// 列検査する」形に一本化した。
#[test]
fn kernel_source_two_dim_smem_rows_are_in_bounds() {
    let sources = kernel_sources();
    let sc = scan_kernels(&sources);
    // 2 次元参照そのもの (`sA[r][c]` のテキスト) はキャストの内側にしか無いので、
    // キャスト内の列を数える。kernels_int8.cu の 6 箇所 + 合成テスト対象。
    assert!(
        sc.rows_checked >= 6,
        "2 次元共有配列の参照を {} 件しか検査していない (kernels_int8.cu だけで 6 件ある)",
        sc.rows_checked
    );
    // 行が記号でも列で判定した安全キャストが実在すること (列検査が動いている証拠)
    assert!(
        sc.symbolic_ok >= 6,
        "行が記号のキャストを {} 件しか「安全」と判定していない (kernels_int8.cu は 6 件)",
        sc.symbolic_ok
    );
    // 行別名 (`const int *row = &sA[r];`) は現存しない。実装は残しているが、
    // 実カーネルに無い構造を下限として要求しない (s105)。
    assert_eq!(
        sc.aliases, 0,
        "2 次元宣言から作った行別名を {} 件検出した: カーネルに行別名が入ったなら \
         このテストの期待値を更新し、別名経由の境界検査 (行→inner の積) を有効にすること",
        sc.aliases
    );
    assert!(
        sc.unchecked_rows.is_empty(),
        "2 次元共有配列の参照を {} 件平坦化できなかった:\n{}",
        sc.unchecked_rows.len(),
        sc.unchecked_rows
            .iter()
            .map(|(f, l, t)| format!("{f}:{l} {t}"))
            .collect::<Vec<_>>()
            .join("\n")
    );
    assert!(
        sc.bad_casts.is_empty(),
        "2 次元共有配列へのキャストに違反が {} 件ある:\n{}",
        sc.bad_casts.len(),
        sc.bad_casts
            .iter()
            .map(|(s, l, t)| format!("{}:{l} {}[{}] {t}", s.file, s.obj, s.off))
            .collect::<Vec<_>>()
            .join("\n")
    );
}

/// s104 (P3-3): 検査器そのものの自己検証 (パーサの空回り防止の本体)。
///
/// 「わざと壊した」最小カーネルを食わせて、次を確かめる:
///  1. 正常な定義 (ストライドが 4 の倍数・整列したキャスト) は通る
///  2. 行ストライドを奇数にすると **失敗する** (= s74 の SP=33 と同型)
///  3. キャスト開始位置を 1 ずらすと **失敗する**
///  4. 配列末尾を越えるキャストは **失敗する**
///  5. 宣言行数を越える 2 次元参照は **panic する**
///  6. 既知違反の表に入れたものは失敗にしない (が、他の違反は残す)
/// これが無いと「正規表現が何にも当たらず常に緑」のテストになり得る。
#[test]
fn kernel_cast_scanner_detects_deliberately_broken_definitions() {
    // 添字は #define 定数だけで書く (ループ変数は意図的に未解決のままにする)
    let good = r#"
#define SP 36
#define TILE 8
#define R 3
#define C 8
extern "C" __global__ void fake_good(const signed char *src) {
    __shared__ signed char sA[TILE][SP];
    *(int*)&sA[R][C] = 0;
}
"#;
    let sc = scan_kernels(&[("fake_good.cu".to_string(), good.to_string())]);
    assert!(
        sc.bad_casts.is_empty(),
        "正常な定義が違反扱いされた: {:?}",
        sc.bad_casts
    );
    assert_eq!(
        sc.ok_casts.len(),
        1,
        "正常な定義からキャストを 1 件検出できるはず: {sc:?}"
    );
    assert_eq!(
        sc.rows_checked, 1,
        "`sA[R][C]` の 2 次元参照を 1 件検査するはず"
    );
    assert_eq!(sc.unresolved_casts.len(), 0);

    // (1) 行ストライドを奇数にする (= s74 の SP=33 と同型: 行ごとに未アライン)
    let bad_stride = good.replace("#define SP 36", "#define SP 35");
    assert_ne!(bad_stride, good);
    let sc = scan_kernels(&[("fake_bad_stride.cu".to_string(), bad_stride.clone())]);
    assert!(
        sc.bad_casts
            .iter()
            .any(|(s, _, _)| s.obj == "sA" && s.off % 4 != 0),
        "SP を 35 に壊しても検出できない (アラインメント検査が効いていない): {sc:?}"
    );

    // (2) キャストの宛先を 1 要素ずらす (列起点が奇数)
    let bad_off = good.replace("#define C 8", "#define C 9");
    assert_ne!(bad_off, good);
    let sc = scan_kernels(&[("fake_bad_off.cu".to_string(), bad_off)]);
    assert!(
        sc.bad_casts.iter().any(|(s, _, _)| s.obj == "sA"),
        "キャスト開始位置を 1 ずらしても検出できない: {sc:?}"
    );

    // (3) 配列末尾を越えるキャスト (最終行の末尾 2 要素手前 = 2 バイトしか残らない)
    let bad_tail = good
        .replace("#define R 3", "#define R 7")
        .replace("#define C 8", "#define C 34");
    assert_ne!(bad_tail, good);
    let sc = scan_kernels(&[("fake_bad_tail.cu".to_string(), bad_tail)]);
    assert!(
        sc.bad_casts.iter().any(|(s, _, _)| s.obj == "sA"),
        "末尾を越えるキャストを検出できない (境界検査が効いていない): {sc:?}"
    );

    // (4) 宣言より後ろの行を読む 2 次元参照は panic する
    let bad_row = good.replace("#define R 3", "#define R 8");
    assert_ne!(bad_row, good);
    let sources = [("fake_bad_row.cu".to_string(), bad_row)];
    let err = std::panic::catch_unwind(|| scan_kernels(&sources)).unwrap_err();
    let msg = err
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| err.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default();
    assert!(
        msg.contains("越える"),
        "宣言行数を越える参照が落ちない: {msg}"
    );

    // (5) 既知違反の表に入れたものは失敗にしない (が、他の違反は残す)
    // s105: 行添字が数値に解決されるようになったので、違反オフセットは
    // 行 3 * ストライド 35 + 列 8 = 113 (旧値 11 は 2 次元添字を読めていなかった
    // 頃の値 = パーサのバグを写した期待値)。表の抑制と「他の違反は残る」を
    // 同時に確かめるため、2 件目の未アラインキャストを足す。
    let two_bad = bad_stride.replace(
        "*(int*)&sA[R][C] = 0;",
        "*(int*)&sA[R][C] = 0;\n    *(int*)&sA[1][C] = 0;",
    );
    assert_ne!(two_bad, bad_stride, "2 件目のキャストを足せていない");
    let known = vec![SmemCast {
        file: "fake_bad_stride.cu".to_string(),
        ty: "int".to_string(),
        obj: "sA".to_string(),
        off: 113,
    }];
    let sc = scan_kernels(&[("fake_bad_stride.cu".to_string(), two_bad)]);
    let listed = sc
        .bad_casts
        .iter()
        .filter(|(s, _, _)| known.contains(s))
        .count();
    let unlisted = sc.bad_casts.len() - listed;
    assert_eq!(
        listed, 1,
        "表に載せた 1 件 (offset 11) が違反として現れていない: {sc:?}"
    );
    assert!(
        unlisted >= 1,
        "表に載せた 1 件で全部を説明できてしまっている (他の違反が消えた): {sc:?}"
    );

    // (6) 空入力でパーサが落ちないこと (scan_kernels は純粋関数)
    let empty = scan_kernels(&[]);
    assert_eq!(empty.casts_seen, 0);
    assert!(empty.bad_casts.is_empty());
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


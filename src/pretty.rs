//! Human-readable terminal output: per-file panel and corpus-level summary.
//!
//! Everything is hand-rolled ANSI escapes (no `colored` dep). The single-file
//! panel uses Unicode box-drawing chars; if you're piping into a tool that
//! mangles UTF-8 you'll want to strip them with `sed` on the way out.

use std::ops::Range;
use std::path::Path;

use crate::bench::FileScore;
use crate::wer::{
    AlignmentResult, Op, TimeSpan, TimingStats, Word, token_word_range, tokens_word_range,
};

// ANSI styling. Terminals supporting xterm-256 / truecolor render these fine.
pub const RESET: &str = "\x1b[0m";
pub const BOLD: &str = "\x1b[1m";
pub const DIM: &str = "\x1b[2m";
pub const RED: &str = "\x1b[31m";
pub const GREEN: &str = "\x1b[32m";
pub const YELLOW: &str = "\x1b[33m";
pub const CYAN: &str = "\x1b[36m";

/// Which side of the alignment is being rendered.
#[derive(Copy, Clone)]
pub enum Side {
    Ref,
    Hyp,
}

fn join_text(words: &[Word], range: Range<usize>) -> String {
    words[range]
        .iter()
        .map(|w| w.text.as_str())
        .collect::<Vec<_>>()
        .join(" ")
}

/// Render one side of the alignment as a single colored line.
pub fn fmt_words_colored(
    side: Side,
    result: &AlignmentResult,
    reference: &[Word],
    hypothesis: &[Word],
) -> String {
    let mut out = String::new();
    for op in &result.ops {
        let (text, color) = match (op, side) {
            (
                Op::Match {
                    ref_range,
                    hyp_range,
                },
                Side::Ref,
            ) => {
                let r = tokens_word_range(&result.ref_tokens[ref_range.clone()]);
                let _ = hyp_range;
                (join_text(reference, r), "")
            }
            (
                Op::Match {
                    ref_range,
                    hyp_range,
                },
                Side::Hyp,
            ) => {
                let h = tokens_word_range(&result.hyp_tokens[hyp_range.clone()]);
                let _ = ref_range;
                (join_text(hypothesis, h), "")
            }
            (Op::Sub { ref_idx, .. }, Side::Ref) => {
                let r = token_word_range(&result.ref_tokens[*ref_idx]);
                (join_text(reference, r), RED)
            }
            (Op::Sub { hyp_idx, .. }, Side::Hyp) => {
                let h = token_word_range(&result.hyp_tokens[*hyp_idx]);
                (join_text(hypothesis, h), YELLOW)
            }
            (Op::Del { ref_idx }, Side::Ref) => {
                let r = token_word_range(&result.ref_tokens[*ref_idx]);
                let txt = format!("{BOLD}{RED}{}{RESET}", join_text(reference, r));
                out.push_str(&txt);
                out.push(' ');
                continue;
            }
            (Op::Del { .. }, Side::Hyp) => {
                out.push_str(&format!("{DIM}—{RESET} "));
                continue;
            }
            (Op::Ins { .. }, Side::Ref) => {
                out.push_str(&format!("{DIM}—{RESET} "));
                continue;
            }
            (Op::Ins { hyp_idx }, Side::Hyp) => {
                let h = token_word_range(&result.hyp_tokens[*hyp_idx]);
                (join_text(hypothesis, h), GREEN)
            }
        };
        if color.is_empty() {
            out.push_str(&text);
        } else {
            out.push_str(&format!("{color}{text}{RESET}"));
        }
        out.push(' ');
    }
    out
}

/// Nearest-rank percentile of a pre-sorted ascending slice.
pub fn percentile(sorted_asc: &[f64], q: f64) -> f64 {
    if sorted_asc.is_empty() {
        return 0.0;
    }
    let idx = ((sorted_asc.len() as f64 - 1.0) * q.clamp(0.0, 1.0)).round() as usize;
    sorted_asc[idx]
}

const BOX_WIDTH: usize = 78;

fn print_box_top(title: &str) {
    println!(
        "\n{CYAN}┌─ {title} {}{RESET}",
        "─".repeat(BOX_WIDTH.saturating_sub(title.chars().count() + 4)),
    );
}

fn print_box_mid() {
    println!("{CYAN}├{}{RESET}", "─".repeat(BOX_WIDTH));
}

fn print_box_bottom() {
    println!("{CYAN}└{}{RESET}\n", "─".repeat(BOX_WIDTH));
}

fn print_wer_line(result: &AlignmentResult) {
    println!(
        "{CYAN}│{RESET} WER:       {wer:6.2} %   {CYAN}│{RESET} S={s} D={d} I={i}  N={n}",
        wer = result.wer() * 100.0,
        s = result.substitutions(),
        d = result.deletions(),
        i = result.insertions(),
        n = result.ref_token_count(),
    );
}

fn print_ref_hyp(result: &AlignmentResult, reference: &[Word], hypothesis: &[Word]) {
    println!(
        "{CYAN}│{RESET} {BOLD}REF{RESET}: {}",
        fmt_words_colored(Side::Ref, result, reference, hypothesis)
    );
    println!(
        "{CYAN}│{RESET} {BOLD}HYP{RESET}: {}",
        fmt_words_colored(Side::Hyp, result, reference, hypothesis)
    );
}

/// One-file panel: header → REF/HYP colored lines → alignment table.
#[allow(clippy::too_many_arguments)]
pub fn print_single(
    path: &Path,
    idx: Option<i32>,
    duration_s: f32,
    transcribe_s: f32,
    reference: &[Word],
    hypothesis: &[Word],
    have_truth: bool,
    result: &AlignmentResult,
    t: &TimingStats,
    drift_threshold_s: f64,
) {
    let title = match idx {
        Some(i) => format!("{} (idx={i})", path.display()),
        None => path.display().to_string(),
    };
    let rtf = if duration_s > 0.0 {
        transcribe_s / duration_s
    } else {
        0.0
    };

    print_box_top(&title);
    println!(
        "{CYAN}│{RESET} Duration:   {duration_s:6.2} s   {CYAN}│{RESET} Inference: {transcribe_s:5.2} s   {CYAN}│{RESET} RTF: {rtf:.3}x"
    );

    if have_truth {
        print_wer_line(result);
        if t.matched_pairs > 0 {
            let (med, p95) = t.percentiles_abs_mid();
            println!(
                "{CYAN}│{RESET} Timing:    matched={mp}  drift mean/median/p95 = {mean:.2}/{med:.2}/{p95:.2}s   high-drift(>{dt:.1}s)={hd}",
                mp = t.matched_pairs,
                mean = t.mean_abs_mid(),
                dt = drift_threshold_s,
                hd = t.high_drift_pairs,
            );
        }
    } else {
        println!("{CYAN}│{RESET} {YELLOW}WER:       (no ground truth for this idx){RESET}");
    }

    print_box_mid();

    if have_truth {
        print_ref_hyp(result, reference, hypothesis);
        print_box_mid();
        println!(
            "{CYAN}│{RESET} {BOLD}Alignment{RESET}  ({GREEN}match{RESET}, {RED}sub{RESET}, {BOLD}{RED}del{RESET}, {GREEN}ins{RESET}; ⚠ = drift>{:.1}s):",
            drift_threshold_s
        );
        for op in &result.ops {
            print_alignment_row(op, result, reference, hypothesis, drift_threshold_s);
        }
    } else {
        let joined: String = hypothesis
            .iter()
            .map(|w| w.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        println!("{CYAN}│{RESET} {BOLD}HYP{RESET}: {joined}");
    }
    print_box_bottom();
}

fn print_alignment_row(
    op: &Op,
    result: &AlignmentResult,
    reference: &[Word],
    hypothesis: &[Word],
    drift_threshold_s: f64,
) {
    let timing = result.op_timing(op);
    match op {
        Op::Match { ref_range, .. } => {
            let r = tokens_word_range(&result.ref_tokens[ref_range.clone()]);
            let r_span = timing.ref_span.expect("Match has ref_span");
            let h_span = timing.hyp_span.expect("Match has hyp_span");
            let dmid =
                (0.5 * (h_span.start + h_span.end) - 0.5 * (r_span.start + r_span.end)).abs();
            let flag = if dmid > drift_threshold_s {
                format!("{YELLOW}⚠{RESET} ")
            } else {
                "  ".to_string()
            };
            println!(
                "{CYAN}│{RESET}   {flag}{GREEN}OK {RESET}  {tx:<24}  {r_lo}  {h_lo}  Δmid={dmid:.2}s",
                tx = join_text(reference, r),
                r_lo = fmt_span("ref", &r_span),
                h_lo = fmt_span("hyp", &h_span),
            );
        }
        Op::Sub { ref_idx, hyp_idx } => {
            let r = token_word_range(&result.ref_tokens[*ref_idx]);
            let h = token_word_range(&result.hyp_tokens[*hyp_idx]);
            let r_span = timing.ref_span.expect("Sub has ref_span");
            let h_span = timing.hyp_span.expect("Sub has hyp_span");
            println!(
                "{CYAN}│{RESET}     {RED}SUB{RESET} {rtxt:<24}  →  {htxt:<24}  {r_lo}  {h_lo}",
                rtxt = join_text(reference, r),
                htxt = join_text(hypothesis, h),
                r_lo = fmt_span("ref", &r_span),
                h_lo = fmt_span("hyp", &h_span),
            );
        }
        Op::Del { ref_idx } => {
            let r = token_word_range(&result.ref_tokens[*ref_idx]);
            let r_span = timing.ref_span.expect("Del has ref_span");
            println!(
                "{CYAN}│{RESET}     {BOLD}{RED}DEL{RESET} {rtxt:<24}  →  {DIM}<missing>{RESET}                  {r_lo}",
                rtxt = join_text(reference, r),
                r_lo = fmt_span("ref", &r_span),
            );
        }
        Op::Ins { hyp_idx } => {
            let h = token_word_range(&result.hyp_tokens[*hyp_idx]);
            let h_span = timing.hyp_span.expect("Ins has hyp_span");
            println!(
                "{CYAN}│{RESET}     {GREEN}INS{RESET} {DIM}<missing>{RESET}                  →  {htxt:<24}  {h_lo}",
                htxt = join_text(hypothesis, h),
                h_lo = fmt_span("hyp", &h_span),
            );
        }
    }
}

fn fmt_span(label: &str, s: &TimeSpan) -> String {
    format!("{label}[{:>5.2}-{:>5.2}]", s.start, s.end)
}

/// WER-only sibling of [`print_single`]: same colored REF/HYP diff, but
/// the header omits timing fields and the alignment table drops the
/// per-op time spans + Δmid column. Use this for hypothesis sources
/// that don't emit word-level timestamps (e.g. the Python script bench).
pub fn print_single_wer_only(
    label: &Path,
    idx: Option<i32>,
    duration_s: f32,
    inference_s: f32,
    reference: &[Word],
    hypothesis: &[Word],
    result: &AlignmentResult,
) {
    let title = match idx {
        Some(i) => format!("{} (idx={i})", label.display()),
        None => label.display().to_string(),
    };

    let rtf = if duration_s > 0.0 {
        inference_s / duration_s
    } else {
        0.0
    };

    print_box_top(&title);
    println!(
        "{CYAN}│{RESET} Duration:   {duration_s:6.2} s   {CYAN}│{RESET} Inference:  {inference_s:6.2} s   {CYAN}│{RESET} RTF: {rtf:.3}x",
    );
    print_wer_line(result);
    print_box_mid();
    print_ref_hyp(result, reference, hypothesis);
    print_box_mid();
    println!(
        "{CYAN}│{RESET} {BOLD}Alignment{RESET}  ({GREEN}match{RESET}, {RED}sub{RESET}, {BOLD}{RED}del{RESET}, {GREEN}ins{RESET}):",
    );
    for op in &result.ops {
        print_alignment_row_no_timing(op, result, reference, hypothesis);
    }
    print_box_bottom();
}

fn print_alignment_row_no_timing(
    op: &Op,
    result: &AlignmentResult,
    reference: &[Word],
    hypothesis: &[Word],
) {
    match op {
        Op::Match { ref_range, .. } => {
            let r = tokens_word_range(&result.ref_tokens[ref_range.clone()]);
            println!(
                "{CYAN}│{RESET}     {GREEN}OK {RESET}  {tx}",
                tx = join_text(reference, r),
            );
        }
        Op::Sub { ref_idx, hyp_idx } => {
            let r = token_word_range(&result.ref_tokens[*ref_idx]);
            let h = token_word_range(&result.hyp_tokens[*hyp_idx]);
            println!(
                "{CYAN}│{RESET}     {RED}SUB{RESET} {rtxt:<24}  →  {htxt}",
                rtxt = join_text(reference, r),
                htxt = join_text(hypothesis, h),
            );
        }
        Op::Del { ref_idx } => {
            let r = token_word_range(&result.ref_tokens[*ref_idx]);
            println!(
                "{CYAN}│{RESET}     {BOLD}{RED}DEL{RESET} {rtxt:<24}  →  {DIM}<missing>{RESET}",
                rtxt = join_text(reference, r),
            );
        }
        Op::Ins { hyp_idx } => {
            let h = token_word_range(&result.hyp_tokens[*hyp_idx]);
            println!(
                "{CYAN}│{RESET}     {GREEN}INS{RESET} {DIM}<missing>{RESET}                  →  {htxt}",
                htxt = join_text(hypothesis, h),
            );
        }
    }
}

/// Corpus-level summary: overall WER (sum-based), per-file WER quantiles,
/// aggregate timing drift, and a worst-N table.
pub fn print_summary(
    per_file: &[FileScore],
    total_timing: &TimingStats,
    total_dur_s: f32,
    total_xt_s: f32,
    worst_n: usize,
    drift_threshold_s: f64,
) {
    let n = per_file.len();
    let subs: usize = per_file.iter().map(|f| f.subs).sum();
    let dels: usize = per_file.iter().map(|f| f.dels).sum();
    let ins: usize = per_file.iter().map(|f| f.ins).sum();
    let ref_total: usize = per_file.iter().map(|f| f.ref_len).sum();
    let overall_wer = if ref_total == 0 {
        0.0
    } else {
        (subs + dels + ins) as f64 / ref_total as f64
    } * 100.0;
    let rtf = if total_dur_s > 0.0 {
        total_xt_s / total_dur_s
    } else {
        0.0
    };

    let mut wers: Vec<f64> = per_file.iter().map(|f| f.wer * 100.0).collect();
    wers.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let hline = "═".repeat(78);
    println!("\n{CYAN}{hline}{RESET}");
    println!(" {BOLD}Summary over {n} files{RESET}");
    println!(
        "   Audio total:          {:.1}s   ({:.2}h)",
        total_dur_s,
        total_dur_s / 3600.0,
    );
    println!(
        "   Inference total:      {:.1}s   (mean {:.2}s/file)",
        total_xt_s,
        total_xt_s / n.max(1) as f32,
    );
    println!("   Mean RTF:             {rtf:.3}x");
    println!(
        "   {BOLD}Overall WER:          {overall_wer:.2}%{RESET}   (S={subs} D={dels} I={ins}  N={ref_total})",
    );
    println!(
        "   Per-file WER:         min {:.2}%  p25 {:.2}%  median {:.2}%  p75 {:.2}%  p95 {:.2}%  max {:.2}%",
        wers.first().copied().unwrap_or(0.0),
        percentile(&wers, 0.25),
        percentile(&wers, 0.50),
        percentile(&wers, 0.75),
        percentile(&wers, 0.95),
        wers.last().copied().unwrap_or(0.0),
    );

    if total_timing.matched_pairs > 0 {
        let drift_share =
            total_timing.high_drift_pairs as f64 / total_timing.matched_pairs.max(1) as f64 * 100.0;
        let (med, p95) = total_timing.percentiles_abs_mid();
        println!(
            "   Timing (matched={mp}):  drift mean/median/p95 = {mean:.2}/{med:.2}/{p95:.2}s   high-drift(>{dt:.1}s)={hd} ({share:.1}%)",
            mp = total_timing.matched_pairs,
            mean = total_timing.mean_abs_mid(),
            dt = drift_threshold_s,
            hd = total_timing.high_drift_pairs,
            share = drift_share,
        );
    } else {
        println!("   {YELLOW}Timing: no matched pairs (no ground truth aligned){RESET}");
    }

    let mut sorted = per_file.to_vec();
    sorted.sort_by(|a, b| {
        b.wer
            .partial_cmp(&a.wer)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let k = worst_n.min(sorted.len());
    if k > 0 {
        println!("\n   {BOLD}Worst {k} by WER:{RESET}");
        println!(
            "     {:>6}  {:>6}  {:>4} {:>4} {:>4} {:>5}  {:>6}  {:>6}   {:>5}  {:>6}  {:>6}   path",
            "idx", "WER%", "S", "D", "I", "N", "dur_s", "xt_s", "drift", "hd_pct", "matchN"
        );
        for f in sorted.iter().take(k) {
            let drift_pct = if f.matched_pairs > 0 {
                f.high_drift_pairs as f64 / f.matched_pairs as f64 * 100.0
            } else {
                0.0
            };
            println!(
                "     {idx:>6}  {wer:>6.2}  {s:>4} {d:>4} {i:>4} {n:>5}  {dur:>6.1}  {xt:>6.2}   {mid:>5.2}  {hd:>5.1}%  {mp:>6}   {p}",
                idx = f.idx.map(|i| i.to_string()).unwrap_or_else(|| "?".into()),
                wer = f.wer * 100.0,
                s = f.subs,
                d = f.dels,
                i = f.ins,
                n = f.ref_len,
                dur = f.duration_s,
                xt = f.transcribe_s,
                mid = f.mean_abs_mid_s,
                hd = drift_pct,
                mp = f.matched_pairs,
                p = f.path.display(),
            );
        }
    }
    println!("{CYAN}{hline}{RESET}\n");
}

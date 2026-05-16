//! Human-readable terminal output: per-file panel and corpus-level summary.
//!
//! Everything is hand-rolled ANSI escapes (no `colored` dep). The single-file
//! panel uses Unicode box-drawing chars; if you're piping into a tool that
//! mangles UTF-8 you'll want to strip them with `sed` on the way out.

use std::path::Path;

use crate::bench::FileScore;
use crate::wer::{AlignOp, TimingStats, WerResult, WerWord};

// ANSI styling. Terminals supporting xterm-256 / truecolor render these fine.
pub const RESET: &str = "\x1b[0m";
pub const BOLD: &str = "\x1b[1m";
pub const DIM: &str = "\x1b[2m";
pub const RED: &str = "\x1b[31m";
pub const GREEN: &str = "\x1b[32m";
pub const YELLOW: &str = "\x1b[33m";
pub const CYAN: &str = "\x1b[36m";

/// Which side of the alignment is being rendered: reference (ground truth)
/// or hypothesis (model output).
#[derive(Copy, Clone)]
pub enum Side {
    Ref,
    Hyp,
}

/// Render one side of the alignment as a single colored line, suitable for
/// the `REF:` / `HYP:` rows of the single-file panel.
pub fn fmt_words_colored(
    side: Side,
    alignment: &[AlignOp],
    reference: &[WerWord],
    hypothesis: &[WerWord],
) -> String {
    let mut out = String::new();
    for op in alignment {
        match (*op, side) {
            (AlignOp::Match { ref_idx, .. }, Side::Ref) => {
                out.push_str(&reference[ref_idx].text);
                out.push(' ');
            }
            (AlignOp::Match { hyp_idx, .. }, Side::Hyp) => {
                out.push_str(&hypothesis[hyp_idx].text);
                out.push(' ');
            }
            (AlignOp::Sub { ref_idx, .. }, Side::Ref) => {
                out.push_str(&format!("{RED}{}{RESET} ", reference[ref_idx].text));
            }
            (AlignOp::Sub { hyp_idx, .. }, Side::Hyp) => {
                out.push_str(&format!("{YELLOW}{}{RESET} ", hypothesis[hyp_idx].text));
            }
            (AlignOp::Del { ref_idx }, Side::Ref) => {
                out.push_str(&format!("{BOLD}{RED}{}{RESET} ", reference[ref_idx].text));
            }
            (AlignOp::Del { .. }, Side::Hyp) => {
                out.push_str(&format!("{DIM}—{RESET} "));
            }
            (AlignOp::Ins { .. }, Side::Ref) => {
                out.push_str(&format!("{DIM}—{RESET} "));
            }
            (AlignOp::Ins { hyp_idx }, Side::Hyp) => {
                out.push_str(&format!("{GREEN}{}{RESET} ", hypothesis[hyp_idx].text));
            }
        }
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

/// One-file panel: header → REF/HYP colored lines → alignment table.
#[allow(clippy::too_many_arguments)]
pub fn print_single(
    path: &Path,
    idx: Option<i32>,
    duration_s: f32,
    transcribe_s: f32,
    reference: &[WerWord],
    hypothesis: &[WerWord],
    have_truth: bool,
    w: &WerResult,
    t: &TimingStats,
    drift_threshold_s: f32,
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
    let hline = "─".repeat(78);

    println!(
        "\n{CYAN}┌─ {title} {}{RESET}",
        "─".repeat(78usize.saturating_sub(title.chars().count() + 4)),
    );
    println!(
        "{CYAN}│{RESET} Duration:   {duration_s:6.2} s   {CYAN}│{RESET} Inference: {transcribe_s:5.2} s   {CYAN}│{RESET} RTF: {rtf:.3}x"
    );

    if have_truth {
        println!(
            "{CYAN}│{RESET} WER:       {wer:6.2} %   {CYAN}│{RESET} S={s} D={d} I={i}  N={n}",
            wer = w.wer() * 100.0,
            s = w.substitutions,
            d = w.deletions,
            i = w.insertions,
            n = w.ref_len,
        );
        if t.matched_pairs > 0 {
            println!(
                "{CYAN}│{RESET} Timing:    matched={mp}  drift mean/median/p95 = {mean:.2}/{med:.2}/{p95:.2}s   high-drift(>{dt:.1}s)={hd}",
                mp = t.matched_pairs,
                mean = t.mean_abs_mid(),
                med = t.median_abs_mid(),
                p95 = t.p95_abs_mid(),
                dt = drift_threshold_s,
                hd = t.high_drift_pairs,
            );
        }
    } else {
        println!("{CYAN}│{RESET} {YELLOW}WER:       (no ground truth for this idx){RESET}");
    }

    println!("{CYAN}├{hline}{RESET}");

    if have_truth {
        println!(
            "{CYAN}│{RESET} {BOLD}REF{RESET}: {}",
            fmt_words_colored(Side::Ref, &w.alignment, reference, hypothesis)
        );
        println!(
            "{CYAN}│{RESET} {BOLD}HYP{RESET}: {}",
            fmt_words_colored(Side::Hyp, &w.alignment, reference, hypothesis)
        );
        println!("{CYAN}├{hline}{RESET}");
        println!(
            "{CYAN}│{RESET} {BOLD}Alignment{RESET}  ({GREEN}match{RESET}, {RED}sub{RESET}, {BOLD}{RED}del{RESET}, {GREEN}ins{RESET}; ⚠ = drift>{:.1}s):",
            drift_threshold_s
        );
        for op in &w.alignment {
            print_alignment_row(*op, reference, hypothesis, drift_threshold_s);
        }
    } else {
        let joined: String = hypothesis
            .iter()
            .map(|w| w.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        println!("{CYAN}│{RESET} {BOLD}HYP{RESET}: {joined}");
    }
    println!("{CYAN}└{hline}{RESET}\n");
}

fn print_alignment_row(
    op: AlignOp,
    reference: &[WerWord],
    hypothesis: &[WerWord],
    drift_threshold_s: f32,
) {
    match op {
        AlignOp::Match { ref_idx, hyp_idx } => {
            let r = &reference[ref_idx];
            let h = &hypothesis[hyp_idx];
            let dmid = (h.midpoint() - r.midpoint()).abs();
            let flag = if dmid > drift_threshold_s {
                format!("{YELLOW}⚠{RESET} ")
            } else {
                "  ".to_string()
            };
            println!(
                "{CYAN}│{RESET}   {flag}{GREEN}OK {RESET}  {tx:<24}  ref[{rs:>5.2}-{re:>5.2}]  hyp[{hs:>5.2}-{he:>5.2}]  Δmid={dmid:.2}s",
                tx = r.text,
                rs = r.start,
                re = r.end,
                hs = h.start,
                he = h.end,
            );
        }
        AlignOp::Sub { ref_idx, hyp_idx } => {
            let r = &reference[ref_idx];
            let h = &hypothesis[hyp_idx];
            println!(
                "{CYAN}│{RESET}     {RED}SUB{RESET} {rtxt:<24}  →  {htxt:<24}  ref[{rs:>5.2}-{re:>5.2}]  hyp[{hs:>5.2}-{he:>5.2}]",
                rtxt = r.text,
                htxt = h.text,
                rs = r.start,
                re = r.end,
                hs = h.start,
                he = h.end,
            );
        }
        AlignOp::Del { ref_idx } => {
            let r = &reference[ref_idx];
            println!(
                "{CYAN}│{RESET}     {BOLD}{RED}DEL{RESET} {rtxt:<24}  →  {DIM}<missing>{RESET}                  ref[{rs:>5.2}-{re:>5.2}]",
                rtxt = r.text,
                rs = r.start,
                re = r.end,
            );
        }
        AlignOp::Ins { hyp_idx } => {
            let h = &hypothesis[hyp_idx];
            println!(
                "{CYAN}│{RESET}     {GREEN}INS{RESET} {DIM}<missing>{RESET}                  →  {htxt:<24}  hyp[{hs:>5.2}-{he:>5.2}]",
                htxt = h.text,
                hs = h.start,
                he = h.end,
            );
        }
    }
}

/// Corpus-level summary: overall WER (sum-based), per-file WER quantiles,
/// aggregate timing drift, and a worst-N table.
pub fn print_summary(
    per_file: &[FileScore],
    totals: &WerResult,
    total_timing: &TimingStats,
    total_dur_s: f32,
    total_xt_s: f32,
    worst_n: usize,
    drift_threshold_s: f32,
) {
    let n = per_file.len();
    let overall_wer = totals.wer() * 100.0;
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
        "   {BOLD}Overall WER:          {overall_wer:.2}%{RESET}   (S={s} D={d} I={i}  N={n_ref})",
        s = totals.substitutions,
        d = totals.deletions,
        i = totals.insertions,
        n_ref = totals.ref_len,
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
            total_timing.high_drift_pairs as f32 / total_timing.matched_pairs.max(1) as f32 * 100.0;
        println!(
            "   Timing (matched={mp}):  drift mean/median/p95 = {mean:.2}/{med:.2}/{p95:.2}s   high-drift(>{dt:.1}s)={hd} ({share:.1}%)",
            mp = total_timing.matched_pairs,
            mean = total_timing.mean_abs_mid(),
            med = total_timing.median_abs_mid(),
            p95 = total_timing.p95_abs_mid(),
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
                f.high_drift_pairs as f32 / f.matched_pairs as f32 * 100.0
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

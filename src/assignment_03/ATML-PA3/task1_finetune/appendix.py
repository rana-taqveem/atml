"""Task 1 appendix tables: every collected result not already shown in the main report.

Reads only saved result files (same sources as summarize.py) and writes LaTeX tables to
results/task1/summary/appendix/. Called from summarize.run; can also be run alone:

    python -m task1_finetune.appendix
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common.plotting import T1_LABELS
from common.utils import ensure_dir, load_config, read_json, rel

OUT = "results/task1/summary/appendix"
VARIANTS = ["full", "l2sp", "lora4", "lora16", "oplora16"]
EXCERPT_HEAD, EXCERPT_TAIL = 170, 110   # characters kept from the start / end of long responses


def tex_escape(s: str) -> str:
    rep = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
           "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(rep.get(c, c) for c in s)


def tabular(spec, header, rows):
    return "\n".join([rf"\begin{{tabular}}{{{spec}}}", r"\toprule", header + r" \\", r"\midrule",
                      *[r + r" \\" for r in rows], r"\bottomrule", r"\end{tabular}"])


def config_table(cfg):
    """Shared training settings, data selection and per-variant hyperparameters."""
    t, d = cfg["train"], read_json(rel("results/task1/data/train_selection_stats.json"))
    lo = cfg["lora"]
    shared = [
        ("Model", tex_escape(cfg["model_id"]) + ", thinking disabled"),
        ("Training data", f"{d['n_selected']:,} MetaMathQA examples ({d['scanned']:,} scanned; "
                          f"{d['excluded_too_long']} too long, {d['excluded_gsm8k_overlap']} GSM8K overlaps, "
                          f"{d['excluded_empty']} empty removed)"),
        ("Tokens", f"{d['total_tokens']:,} total, {d['total_response_tokens']:,} response; "
                   f"mean length {d['mean_len']:.1f}, max {d['max_len']:,} (cap {cfg['data']['max_len']:,})"),
        ("Optimizer", rf"AdamW, $\beta=({t['betas'][0]}, {t['betas'][1]})$, $\epsilon=10^{{-8}}$, "
                      f"weight decay {t['weight_decay']}"),
        ("Schedule", f"{t['updates']:,} updates, cosine with {t['warmup']} warmup updates, clip {t['clip']}"),
        ("Batch", f"microbatch {t['microbatch']} $\\times$ {t['grad_accum']} accumulation = "
                  f"{t['microbatch'] * t['grad_accum']}; midpoint at update {t['midpoint_update']}"),
        ("Precision", "FP16 autocast, gradient checkpointing"),
        ("LoRA", f"dropout {lo['dropout']}; targets " + tex_escape(", ".join(lo["targets"]))),
        ("Hardware", "Tesla T4 (all runs)"),
    ]
    a = tabular("lp{0.72\\linewidth}", "Setting & Value", [f"{k} & {v}" for k, v in shared])
    rows = []
    for v in VARIANTS:
        c = cfg["variants"][v]
        lam = c.get("l2sp_lambda")
        lam_s = "--" if lam is None else ("0" if lam == 0 else f"$10^{{{int(np.log10(lam))}}}$")
        rows.append(f"{T1_LABELS[v]} & {sci(c['lr'])} & {lam_s} & {c.get('r', '--')} & "
                    f"{c.get('alpha', '--')} & {c.get('k', '--')}")
    b = tabular("lccccc", r"Method & LR & $\lambda$ & $r$ & $\alpha$ & $k$", rows)
    return a + "\n\n\\vspace{4pt}\n\n" + b


def eval_details_table():
    """Secondary metrics and evaluation checks not shown in the main benchmark table."""
    b = pd.read_csv(rel("results/task1/summary/benchmarks.csv"))
    rows = []
    for _, r in b.iterrows():
        merge = "--" if pd.isna(r["merge_max_abs_diff_fp32"]) else f"{r['merge_max_abs_diff_fp32']:.2e}"
        rows.append(f"{r['label']} & {r['gsm8k_strict']:.2f} & {r['gsm8k_truncation_rate']:.2f} & "
                    f"{r['gsm8k_extractable_rate']:.2f} & {r['gsm8k_mean_tokens']:.1f} & "
                    f"{r['hellaswag_acc']:.2f} & {r['arc_easy_acc']:.2f} & {merge}")
    return tabular("lccccccc",
                   r"Condition & \shortstack{GSM8K\\strict (\%)} & \shortstack{GSM8K\\trunc. (\%)} & "
                   r"\shortstack{GSM8K\\extract. (\%)} & \shortstack{GSM8K\\mean tok.} & "
                   r"\shortstack{HellaSwag\\acc (\%)} & \shortstack{ARC-E\\acc (\%)} & "
                   r"\shortstack{Merge max\\$|\Delta|$ (FP32)}", rows)


def transitions_table():
    t = pd.read_csv(rel("results/task1/summary/gsm8k_transitions.csv"))
    rows = [f"{T1_LABELS[r.condition]} & {r.wrong_to_right} & {r.right_to_wrong} & "
            f"{r.wrong_to_right - r.right_to_wrong:+d}" for r in t.itertuples()]
    return tabular("lccc", r"Condition & Wrong $\to$ right & Right $\to$ wrong & Net", rows)


def resource_details_table():
    rows = []
    for v in VARIANTS:
        s = read_json(rel(f"results/task1/train/{v}/summary.json"))
        sz = s["checkpoint_sizes"]
        if v == "l2sp":
            extra = f"$\\theta_0$ reference {s['l2sp_reference_bytes'] / 1e6:,.1f} MB"
        elif v == "oplora16":
            extra = f"bases {s['basis_bytes'] / 1e6:,.1f} MB, {s['basis_seconds']:.1f} s"
        else:
            extra = "--"
        rows.append(f"{T1_LABELS[v]} & {s['total_params']:,} & {s['train_seconds'] / s['updates']:.2f} & "
                    f"{s['peak_reserved_gb']:.2f} & {sz['recovery'] / 1e6:,.1f} & "
                    f"{s['skipped_updates']} & {extra}")
    return tabular("lrccccl", r"Method & Total params & s/update & \shortstack{Peak reserved\\(GiB)} & "
                              r"\shortstack{Recovery\\ckpt (MB)} & Skipped & Method-specific overhead", rows)


def small(x: float) -> str:
    """Three decimals, or LaTeX scientific notation for the near-zero OPLoRA values."""
    if abs(x) >= 1e-3:
        return f"{x:.3f}"
    m, e = f"{x:.2e}".split("e")
    return rf"${m}\times10^{{{int(e)}}}$"


def sci(x: float) -> str:
    m, e = f"{x:.0e}".split("e")
    return rf"${m}\times10^{{{int(e)}}}$"


def weight_space_table():
    w = pd.read_csv(rel("results/task1/diagnostics/weight_space.csv"))
    rows = []
    for c in T1_LABELS:
        for r in w[w["condition"] == c].itertuples():
            rows.append(f"{T1_LABELS[c]} & {r.layer} & {small(r.rho_k)} & {small(r.right_residual)} & "
                        f"{r.eta:.4f} & {r.dW_fro:.3f}")
    return tabular("lccccc", r"Condition & Layer & $\rho_k$ & Input share & $\eta$ & "
                             r"$\lVert\Delta W\rVert_F$", rows)


def activation_table():
    s = pd.read_csv(rel("results/task1/diagnostics/activation_summary.csv"))
    ratio = pd.read_csv(rel("results/task1/diagnostics/activation_ratio.csv"))
    rows = []
    for c in T1_LABELS:
        for r in ratio[ratio["condition"] == c].itertuples():
            g = s[(s["condition"] == c) & (s["layer"] == r.layer)].set_index("set")
            rows.append(f"{T1_LABELS[c]} & {r.layer} & {g.loc['math', 'mean']:.4f} ({g.loc['math', 'std']:.4f}) & "
                        f"{g.loc['nonmath', 'mean']:.4f} ({g.loc['nonmath', 'std']:.4f}) & "
                        f"{r.math_over_nonmath:.2f}")
    return tabular("lcccc", r"Condition & Layer & Math $R(z)$, mean (std) & Non-math $R(z)$, mean (std) & "
                            r"Math / non-math", rows)


def auto_flags_table():
    f = pd.read_csv(rel("results/task1/qualitative/auto_flags_by_condition.csv"))
    rows = [f"{T1_LABELS[r.condition]} & {r.truncated}/{r.n} & {r.mean_tokens:.1f} & "
            f"{round(r.json_valid * r.n / 3)}/{r.n // 3} & {round(r.template_filled * r.n / 3)}/{r.n // 3} & "
            f"{r.repeated_4gram:.3f} & {r.math_markers}" for r in f.itertuples()]
    return tabular("lcccccc", r"Condition & Truncated & \shortstack{Mean\\tokens} & \shortstack{JSON\\valid} & "
                              r"\shortstack{Template\\filled} & \shortstack{Repeated\\4-gram} & "
                              r"\shortstack{Math\\markers}", rows)


def excerpt(text: str) -> str:
    text = " ".join(str(text).split())
    if len(text) > EXCERPT_HEAD + EXCERPT_TAIL + 20:
        return tex_escape(text[:EXCERPT_HEAD]) + r" \textit{[\dots]} " + tex_escape(text[-EXCERPT_TAIL:])
    return tex_escape(text)


def excerpts_table(groups=(("what_is_your_wish", "plain"), ("enough_of_my_husband", "json"))):
    """Aligned excerpts for the groups the author selected in RQ5. Whitespace collapsed; long
    responses keep the first EXCERPT_HEAD and last EXCERPT_TAIL characters."""
    d = pd.read_csv(rel("results/task1/qualitative/annotation_sheet_42.csv"))
    out = []
    for q, fmt in groups:
        g = d[(d["question"] == q) & (d["format"] == fmt)]
        rows = [rf"\multicolumn{{2}}{{p{{0.92\linewidth}}}}{{\textit{{Prompt ({tex_escape(q)}, {fmt}):}} "
                rf"{tex_escape(' '.join(g['prompt'].iloc[0].split()))}}}", r"\midrule"]
        for c in T1_LABELS:
            r = g[g["condition"] == c].iloc[0]
            mark = r" \textsuperscript{T}" if bool(r["truncated"]) else ""
            rows.append(f"{T1_LABELS[c]}{mark} & \\texttt{{\\footnotesize {excerpt(r['response'])}}}")
        out.append("\n".join([r"\begin{tabular}{l>{\raggedright\arraybackslash}p{0.70\linewidth}}", r"\toprule",
                              r" \\ ".join(rows[:1]) + r" \\", rows[1],
                              *[x + r" \\[2pt]" for x in rows[2:]], r"\bottomrule", r"\end{tabular}"]))
    return "\n\n\\vspace{6pt}\n\n".join(out)


def write_all(cfg):
    out = ensure_dir(OUT)
    tables = {"config": config_table(cfg), "eval_details": eval_details_table(),
              "gsm8k_transitions": transitions_table(), "resource_details": resource_details_table(),
              "weight_space": weight_space_table(), "activations": activation_table(),
              "auto_flags": auto_flags_table(), "excerpts": excerpts_table()}
    for name, tex in tables.items():
        (out / f"{name}.tex").write_text(tex, encoding="utf-8")
    return list(tables)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task1.yaml")
    print(write_all(load_config(ap.parse_args().config)))


if __name__ == "__main__":
    main()

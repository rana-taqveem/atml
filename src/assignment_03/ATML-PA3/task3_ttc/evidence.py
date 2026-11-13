"""Task 3 report evidence not covered by summarize.py.

Reads only saved results (plus the supplied rollout text for example excerpts); no model, no GPU.
Writes results/task3/summary/{methods_diff,signals,live,adaptive,qual_main}_table.tex,
results/task3/summary/appendix/*.tex, results/task3/summary/qualitative_selection.json and
report/figures/task3/{adaptive_cost,trajectories}.{pdf,png}.

    python -m task3_ttc.evidence

Before writing, it checks that (1) per-question results reproduce the summary CSVs, (2) re-simulating
the selected stopping policies from the saved checkpoint states reproduces the saved evaluation
outcomes and selected_policies.json, and (3) the live summary is reproduced from generations.jsonl.

Example-selection rules (fixed, held-out questions in split order, reveal order 0, N = 32):
  * Selection disagreement: per dataset, the first question where self-consistency and the best RWC
    rule (self-certainty + RWC for MATH, VisConf + RWC for MathVista) choose different answers and
    exactly one is correct; one case for each direction. The main table shows the first of these.
  * Revision (live, 50 questions in live order): the first question of each kind: wrong to correct,
    answer changed but still wrong, answer kept. The main table shows the first changed answer.
  * Premature stopping: per dataset and selected controller, the first question whose run stops
    before 32 with a wrong answer that fixed-32 self-consistency gets right. The main table shows
    the MATH agreement-based case.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from common.plotting import INK, INK2, NEUTRAL, SERIES, TEXT_WIDTH, save, setup
from common.utils import ensure_dir, load_config, read_json, rel, write_json
from task1_finetune.appendix import tabular as _tabular, tex_escape
from task3_ttc.adaptive import simulate
from task3_ttc.summarize import COLORS, LABELS

R = "results/task3"
FIG = "report/figures/task3"
DS = ("math", "mathvista")
DNAME = {"math": "MATH", "mathvista": "MathVista"}
BEST_RWC = {"math": "cert_rwc", "mathvista": "visconf_rwc"}
ADAPT_SCORE = {"math": "self-certainty", "mathvista": "VisConf"}
Q_CHARS, TEXT_CHARS = 150, 220


def tabular(spec, header, rows):
    # a row ending in "\\ \midrule" would otherwise be followed by an empty row
    return _tabular(spec, header, rows).replace(" \\\\ \\midrule \\\\", " \\\\ \\midrule")


def pct(x, d=1):
    return f"{100 * x:.{d}f}"


def ci(m, lo, hi, d=1, sign=True):
    f = f"{{:{'+' if sign else ''}.{d}f}}"
    return f"{f.format(100 * m)} [{100 * lo:.{d}f}, {100 * hi:.{d}f}]"


UNICODE_MATH = {"⊙": r"$\odot$", "∥": r"$\parallel$", "∠": r"$\angle$", "△": r"$\triangle$", "°": r"$^\circ$",
                "π": r"$\pi$", "√": r"$\surd$", "×": r"$\times$", "÷": r"$\div$", "≤": r"$\leq$", "≥": r"$\geq$",
                "≠": r"$\neq$", "⊥": r"$\perp$", "≈": r"$\approx$", "−": "-", "θ": r"$\theta$", "α": r"$\alpha$",
                "β": r"$\beta$", "∘": r"$\circ$", "⌒": r"$\frown$", "′": "'", "“": "``", "”": "''", "’": "'", "–": "--", "—": "---"}


def excerpt(s, n):
    s = " ".join(str(s).split())
    t = tex_escape(s[:n])
    # characters the report's fonts cannot set: known math symbols mapped, anything else outside Latin-1 marked
    t = "".join(UNICODE_MATH.get(c, c if ord(c) < 0x250 else f"[U+{ord(c):04X}]") for c in t)
    return t + (r"\ldots" if len(s) > n else "")


def ans(a, n=40):
    """Answer string for a table cell: escaped, monospace, shortened."""
    if a is None or (isinstance(a, float) and np.isnan(a)):
        return r"\textit{no answer}"
    s = str(a).replace("RAW:", "")
    return r"\texttt{" + excerpt(s, n) + "}"


# ------------------------------------------------------------------------ loading and checks
def load(cfg):
    sp = read_json(rel(f"{R}/splits.json"))
    sums = {d: pd.read_csv(rel(f"{R}/offline/summary_{d}_eval.csv")) for d in DS}
    devs = {d: pd.read_csv(rel(f"{R}/offline/summary_{d}_dev.csv")) for d in DS}
    decs = {d: pd.read_csv(rel(f"{R}/offline/decomposition_{d}_eval.csv")) for d in DS}
    return sp, sums, devs, decs


def check_offline(sums):
    for d in DS:
        pq = pd.read_parquet(rel(f"{R}/offline/per_question_{d}_eval.parquet"))
        m = pq.groupby(["method", "N"])[["acc", "oracle", "cost"]].mean()
        s = sums[d].set_index(["method", "N"])
        for col in ("acc", "oracle", "cost"):
            assert np.allclose(m[col].loc[s.index], s[col], atol=1e-9), f"{d} {col}"


def check_adaptive(cfg, pol):
    sims = {}
    for d in DS:
        states = pd.read_parquet(rel(f"{R}/adaptive/states_{d}.parquet"))
        states["sample_id"] = states["sample_id"].astype(str)
        f32 = simulate(states, cfg).set_index(["sample_id", "order"])
        for fam in ("agreement", "quality"):
            p = pol[d][fam]
            sim = simulate(states, cfg, p["tau"], p["gamma"] if fam == "quality" else None)
            saved = pd.read_parquet(rel(f"{R}/adaptive/eval_{fam}_{d}.parquet"))
            saved["sample_id"] = saved["sample_id"].astype(str)
            ev = sim[sim.sample_id.isin(set(saved.sample_id))].set_index(["sample_id", "order"]).sort_index()
            sv = saved.set_index(["sample_id", "order"]).sort_index()
            assert (ev["n"].to_numpy() == sv["n"].to_numpy()).all() and (ev["sc_correct"].to_numpy() == sv["sc_correct"].to_numpy()).all(), f"{d} {fam}"
            assert abs(sv["sc_correct"].mean() - p["eval"]["acc"]) < 1e-9, f"{d} {fam} acc"
            sims[(d, fam)] = sv.join(f32[["sc_group", "sc_correct"]].rename(columns=lambda c: c + "_f32"))
    return sims


def check_live():
    from task3_ttc.live import summarize as live_summarize

    recs = [json.loads(x) for x in open(rel(f"{R}/live/generations.jsonl"), encoding="utf-8")]
    df = live_summarize(recs)
    saved = pd.read_csv(rel(f"{R}/live/summary.csv")).set_index("strategy")
    new = df.groupby("strategy").agg(accuracy=("correct", "mean"), output_tokens=("output_tokens", "mean"),
                                     seconds=("seconds", "mean"), any_truncated=("any_truncated", "mean"))
    for col in new.columns:
        assert np.allclose(new[col], saved.loc[new.index, col], atol=1e-9), f"live {col}"
    return recs, df


# ------------------------------------------------------------------------ main tables
METHOD_ORDER = ["first", "sc", "logp", "cert", "logp_rwc", "cert_rwc", "shortest",
                "vision", "vision_rwc", "visconf", "visconf_rwc", "tokvisconf", "tokvisconf_rwc"]


def methods_table(sums, N=32):
    rows = []
    for m in METHOD_ORDER:
        cells = []
        for d in DS:
            r = sums[d][(sums[d].method == m) & (sums[d].N == N)]
            if r.empty:
                cells += ["--", "--"]
                continue
            r = r.iloc[0]
            cells.append(ci(r.acc, r.acc_lo, r.acc_hi, sign=False))
            cells.append("(reference)" if m == "sc" else ci(r.diff_vs_sc, r.diff_lo, r.diff_hi))
        rows.append(f"{LABELS[m]} & " + " & ".join(cells))
    sc = {d: sums[d][(sums[d].method == "sc") & (sums[d].N == N)].iloc[0] for d in DS}
    fa = {d: sums[d][(sums[d].method == "first") & (sums[d].N == N)].iloc[0] for d in DS}
    rows[-1] += r" \\ \midrule"
    rows.append("Oracle coverage & " + " & ".join(f"{pct(sc[d].oracle)} & " for d in DS).rstrip(" &"))
    rows.append("Mean tokens (FirstAns / all 32) & " + " & ".join(f"{fa[d].cost:,.0f} / {sc[d].cost:,.0f} & " for d in DS).rstrip(" &"))
    head = (r"& \multicolumn{2}{c}{MATH} & \multicolumn{2}{c}{MathVista} \\ \cmidrule(lr){2-3}\cmidrule(lr){4-5}" "\n"
            r"Method ($N=32$) & Acc.\ [95\% CI] & $\Delta$ vs SC (pp) [95\% CI] & Acc.\ [95\% CI] & $\Delta$ vs SC (pp) [95\% CI]")
    return tabular("lcccc", head, rows)


SIGNALS = [("cert", "Self-certainty"), ("logp", "Log-probability"), ("neg_length", "Shortness (length control)"),
           ("vision", "Visual engagement"), ("visconf", "VisConf"), ("tokvisconf", "Tokenwise VisConf")]


def signal_rows(sig, subset):
    out = {}
    for d in DS:
        for r in sig[d]["quartiles"]:
            if r["subset"] == subset:
                out[(d, r["score"])] = r
    return out


def signals_table(sig, cv):
    q = signal_rows(sig, "mixed")
    n = {d: next(r["n_questions"] for r in sig[d]["quartiles"] if r["subset"] == "mixed") for d in DS}
    rows = []
    for s, name in SIGNALS:
        cells = []
        for d in DS:
            r = q.get((d, s))
            cells += (["--", "--"] if r is None else
                      [f"{100 * r['top_minus_bottom']:+.1f}", "--" if s == "neg_length" else f"{r['spearman_with_length']:+.2f}"])
        rows.append(f"{name} & " + " & ".join(cells))
    p = cv["pearson_c_v"]
    rows[-1] += r" \\ \midrule"
    rows.append(rf"\multicolumn{{5}}{{l}}{{Within-rollout corr$(c_t, v_t)$, MathVista ({len(cv):,} held-out rollouts): mean {p.mean():+.3f}, "
                rf"median {p.median():+.3f}, IQR [{p.quantile(.25):+.3f}, {p.quantile(.75):+.3f}]; mean Cov$_t(c,v)$ {cv['cov_c_v'].mean():+.4f}}}")
    head = (rf"& \multicolumn{{2}}{{c}}{{MATH ({n['math']} mixed questions)}} & \multicolumn{{2}}{{c}}{{MathVista ({n['mathvista']} mixed)}} \\ "
            r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}" "\n"
            r"Score ($N=32$) & Top$-$bottom quartile (pp) & $\rho$ with length & Top$-$bottom quartile (pp) & $\rho$ with length")
    return tabular("lcccc", head, rows)


def live_table(meta):
    s = pd.read_csv(rel(f"{R}/live/summary.csv")).set_index("strategy")
    cfg = meta["config"]
    names = {"single": f"Single ({cfg['single_cap']:,} cap)",
             "parallel": f"Parallel ({cfg['n_parallel']} $\\times$ {cfg['parallel_cap']:,}) + SC",
             "sequential": f"Sequential ({cfg['parallel_cap']:,} + {cfg['revision_cap']:,} revision)"}
    rows = []
    for k in ("single", "parallel", "sequential"):
        r = s.loc[k]
        rows.append(f"{names[k]} & {pct(r.accuracy, 0)} ({round(r.accuracy * r.n)}) & {pct(r.extraction_rate, 0)} & {r.input_tokens:,.1f} & "
                    f"{r.output_tokens:,.1f} & {r.seconds:.1f} & {pct(r.any_truncated, 0)} & {pct(r.final_truncated, 0)} & {r.peak_gb:.2f}")
    t = meta["transitions_initial_to_revised"]
    rows[-1] += r" \\ \midrule"
    rows.append(rf"\multicolumn{{9}}{{l}}{{Initial $\to$ revised (shared first parallel response): wrong$\to$correct {t['wrong_to_correct']}, "
                rf"correct$\to$wrong {t['correct_to_wrong']}, correct kept {t['correct_kept']}, wrong kept {t['wrong_kept']}; answer changed {t['answer_changed']}}}")
    head = (r"Strategy ($n=50$) & Acc.\ \% (count) & Extracted \% & Input tok. & Output tok. & Seconds & Any trunc.\ \% & "
            r"Final trunc.\ \% & Peak GiB")
    return tabular("lcccccccc", head, rows)


def adaptive_table(pol, sums):
    rows = []
    for d in DS:
        for fam in ("agreement", "quality"):
            p = pol[d][fam]
            e = p["eval"]
            sd = e["stop_dist"]
            par = rf"$\tau={p['tau']}$" + (rf", $\gamma={p['gamma']}$" if fam == "quality" else "")
            rows.append(f"{DNAME[d]} & {fam.capitalize()} ({par}) & {pct(p['dev_acc'])} & {pct(e['acc'])} & "
                        f"{ci(p['diff_vs_fixed32'], p['diff_lo'], p['diff_hi'], d=2)} & {e['cost']:,.0f} & {e['rollouts']:.1f} & "
                        + "/".join(pct(sd[k], 0) for k in ("4", "8", "16", "32")) +
                        f" & {pct(p['incorrect_early_stops'])} & {pct(p['incorrect_early_recoverable_by_fixed32'])}")
        for N in (8, 16, 32):
            f = pol[d]["fixed_sc"][str(N)]
            rows.append(f"{DNAME[d]} & Fixed SC, $N={N}$ & & {pct(f['acc'])} & & {f['cost']:,.0f} & {f['rollouts']:.0f} & & & ")
        rows[-1] += r" \\ \midrule" if d == "math" else ""
    head = (r"Data & Controller & Dev acc. & Acc. & $\Delta$ vs fixed 32 (pp) [95\% CI] & Tokens & Rollouts & "
            r"Stop @4/8/16/32 (\%) & Wrong early (\%) & Fixed 32 right (\% of those)")
    return tabular("llcccccccc", head, rows)


# ------------------------------------------------------------------------ examples
def raw_tables(cfg):
    out = {}
    for d in DS:
        g = pd.read_parquet(rel(f"{cfg['data']['raw_dir']}/{cfg['data']['datasets'][d]}/generations.parquet"),
                            columns=["sample_id", "rollout_id", "question_text", "gold_answer", "extracted_answer"])
        g["sample_id"] = g["sample_id"].astype(str)
        gr = pd.read_parquet(rel(f"{R}/grades_{d}.parquet"))
        gr["sample_id"] = gr["sample_id"].astype(str)
        m = gr.merge(g, on=["sample_id", "rollout_id"], how="left", validate="one_to_one")
        out[d] = m
    return out


def group_info(m, q):
    """Answer groups of one question: representative answer, size, correctness."""
    g = m[m.sample_id == q].sort_values("rollout_id")
    info = {}
    for gid, x in g.groupby("group", sort=False):
        a = None if x.no_answer.iloc[0] else (x.group_key.iloc[0] if not str(x.group_key.iloc[0]).isdigit() else x.extracted_answer.iloc[0])
        info[int(gid)] = {"answer": a, "size": len(x), "correct": bool(x.correct.iloc[0])}
    return g.question_text.iloc[0], g.gold_answer.iloc[0], info


def disagreements(sp, raw):
    out = {}
    for d in DS:
        s = pd.read_parquet(rel(f"{R}/offline/selections_{d}.parquet"))
        s["sample_id"] = s["sample_id"].astype(str)
        s = s[(s.N == 32) & (s.order == 0) & s.method.isin(["sc", BEST_RWC[d]])]
        w = s.pivot(index="sample_id", columns="method", values=["group", "correct"])
        found = {}
        for q in sp[d]["eval"]:
            r = w.loc[q]
            a, b = bool(r[("correct", "sc")]), bool(r[("correct", BEST_RWC[d])])
            if r[("group", "sc")] != r[("group", BEST_RWC[d])] and a != b:
                key = "rwc_right" if b else "sc_right"
                if key not in found:
                    text, gold, info = group_info(raw[d], q)
                    found[key] = {"sample_id": q, "question": text, "gold": gold, "groups": info,
                                  "sc_group": int(r[("group", "sc")]), "rwc_group": int(r[("group", BEST_RWC[d])])}
            if len(found) == 2:
                break
        out[d] = found
    return out


def live_problem(r):
    """Problem text from the rendered chat prompt of the first parallel call."""
    p = r["parallel"][0]["prompt"]
    p = p.split("<|im_start|>user\n", 1)[-1]
    return p.split("\n\nThink step by step", 1)[0]


def revisions(recs):
    kinds = {"first_changed": next(r for r in recs if r["answers"]["initial"] != r["answers"]["sequential"])}
    for r in recs:
        a, b = r["correct"]["initial"], r["correct"]["sequential"]
        changed = r["answers"]["initial"] != r["answers"]["sequential"]
        k = "wrong_to_correct" if (not a and b) else ("changed_still_wrong" if changed and not b else ("kept" if not changed else None))
        if k and k not in kinds:
            kinds[k] = r
    return kinds


def premature(sp, sims, raw):
    out = {}
    for d in DS:
        for fam in ("agreement", "quality"):
            s = sims[(d, fam)]
            for q in sp[d]["eval"]:
                r = s.loc[(q, 0)]
                if r["n"] < 32 and not r["sc_correct"] and r["sc_correct_f32"]:
                    text, gold, info = group_info(raw[d], q)
                    out[(d, fam)] = {"sample_id": q, "question": text, "gold": gold, "groups": info, "n": int(r["n"]),
                                     "lead_frac": float(r["lead_frac"]), "sc_group": int(r["sc_group"]),
                                     "f32_group": int(r["sc_group_f32"]), "pool_correct_frac": float(r["pool_correct_frac"])}
                    break
    return out


def votes(info, top=4):
    items = sorted(info.items(), key=lambda kv: -kv[1]["size"])[:top]
    return "; ".join(f"{ans(v['answer'], 24)} {v['size']}{r'$\checkmark$' if v['correct'] else ''}" for _, v in items)


def dis_cell(d, e):
    i = e["groups"]
    return (f"SC: {ans(i[e['sc_group']]['answer'])} ({i[e['sc_group']]['size']}/32, {'right' if i[e['sc_group']]['correct'] else 'wrong'}); "
            f"{LABELS[BEST_RWC[d]]}: {ans(i[e['rwc_group']]['answer'])} ({i[e['rwc_group']]['size']}/32, {'right' if i[e['rwc_group']]['correct'] else 'wrong'}). "
            f"Groups: {votes(i)}")


def rev_cell(r, n=TEXT_CHARS):
    init = r["parallel"][0]
    return (f"Initial: {ans(r['answers']['initial'])} ({init['output_tokens']} tok{', truncated' if init['truncated'] else ''}); "
            f"revised: {ans(r['answers']['sequential'])} ({r['revision']['output_tokens']} tok). Revision opens: ``{excerpt(r['revision']['response'], n)}''")


def stop_cell(e):
    i = e["groups"]
    return (f"Stopped at $n={e['n']}$ with {ans(i[e['sc_group']]['answer'])} ({100 * e['lead_frac']:.0f}\\% of revealed, wrong); "
            f"fixed 32: {ans(i[e['f32_group']]['answer'])} (right). Correct share of full pool {100 * e['pool_correct_frac']:.0f}\\%. "
            f"Groups: {votes(i)}")


def qual_main(dis, rev, pre):
    spec = r"@{}>{\raggedright\arraybackslash}p{0.14\linewidth}>{\raggedright\arraybackslash}p{0.28\linewidth}>{\raggedright\arraybackslash}p{0.5\linewidth}@{}"
    rows = []
    first = next(iter(dis["math"].values()))   # dict filled in split order, so this is the earliest case
    rows.append(f"Selection disagreement (MATH {tex_escape(first['sample_id'])}; gold {ans(first['gold'])}) & {excerpt(first['question'], Q_CHARS)} & {dis_cell('math', first)}")
    r = rev["first_changed"]
    rows.append(f"Revision (live {tex_escape(r['sample_id'])}; gold {ans(r['gold'])}) & {excerpt(live_problem(r), Q_CHARS)} & {rev_cell(r, 150)}")
    p = pre[("math", "agreement")]
    rows.append(f"Premature stop (MATH {tex_escape(p['sample_id'])}, agreement; gold {ans(p['gold'])}) & {excerpt(p['question'], Q_CHARS)} & {stop_cell(p)}")
    return tabular(spec, r"Behaviour (case) & Question (start) & Observed", rows)


def qual_appendix(dis, rev, pre):
    spec = r"@{}>{\raggedright\arraybackslash}p{0.16\linewidth}>{\raggedright\arraybackslash}p{0.28\linewidth}>{\raggedright\arraybackslash}p{0.48\linewidth}@{}"
    rows = []
    for d in DS:
        for k, e in dis[d].items():
            lab = "RWC right, SC wrong" if k == "rwc_right" else "SC right, RWC wrong"
            rows.append(f"Disagreement, {DNAME[d]} {tex_escape(e['sample_id'])} ({lab}); gold {ans(e['gold'])} & {excerpt(e['question'], Q_CHARS)} & {dis_cell(d, e)}")
    rows[-1] += r" \\ \midrule"
    for k in ("wrong_to_correct", "changed_still_wrong", "kept"):
        if k not in rev:
            continue
        r, lab = rev[k], {"wrong_to_correct": "wrong to correct", "changed_still_wrong": "changed, still wrong", "kept": "answer kept"}[k]
        rows.append(f"Revision, live {tex_escape(r['sample_id'])} ({lab}); gold {ans(r['gold'])} & {excerpt(live_problem(r), Q_CHARS)} & {rev_cell(r)}")
    rows[-1] += r" \\ \midrule"
    for (d, fam), e in pre.items():
        rows.append(f"Premature stop, {DNAME[d]} {tex_escape(e['sample_id'])} ({fam}); gold {ans(e['gold'])} & {excerpt(e['question'], Q_CHARS)} & {stop_cell(e)}")
    return tabular(spec, r"Case & Question (start) & Observed", rows)


# ------------------------------------------------------------------------ figures
def adaptive_figure(sums, pol):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.3), gridspec_kw={"wspace": 0.28})
    for ax, d in zip(axes, DS):
        s = sums[d]
        for m in ("sc", BEST_RWC[d]):
            x = s[s.method == m].sort_values("N")
            ax.plot(x.cost / 1000, 100 * x.acc, color=COLORS[m], marker="o", ms=2.5, label=f"{LABELS[m]} (fixed $N$)")
        x = s[s.method == "sc"].sort_values("N")
        for _, r in x[x.N >= 4].iterrows():   # N = 1, 2 sit on top of FirstAns
            ax.annotate(f"N={int(r.N)}", (r.cost / 1000, 100 * r.acc), textcoords="offset points", xytext=(0, -9),
                        ha="center", fontsize=6, color=INK2)
        f = s[(s.method == "first") & (s.N == 32)].iloc[0]
        ax.scatter([f.cost / 1000], [100 * f.acc], color=NEUTRAL, s=14, zorder=3, label="FirstAns")
        for fam, mk in (("agreement", "^"), ("quality", "s")):
            e = pol[d][fam]["eval"]
            ax.scatter([e["cost"] / 1000], [100 * e["acc"]], marker=mk, s=26, color=INK, zorder=4,
                       label=f"Adaptive, {fam}", edgecolor="white", linewidth=0.6)
        ax.set_xscale("log")
        ticks = [t for t in (0.25, 0.5, 1, 2, 5, 10, 20) if s.cost.min() / 1000 * 0.8 <= t <= s.cost.max() / 1000 * 1.2]
        ax.set_xticks(ticks, [f"{t:g}" for t in ticks]); ax.minorticks_off()
        ax.set_xlabel("Tokens per question (thousands, log)")
        ax.set_title(DNAME[d])
    axes[0].set_ylabel("Held-out accuracy (%)")
    h, l = [], []
    for ax in axes:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh); l.append(ll)
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 1.0), fontsize=6.5)
    save(fig, f"{FIG}/adaptive_cost")


def trajectory_figure(traj, w=15):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(traj), 2, figsize=(TEXT_WIDTH, 1.35 * len(traj)), sharey=True,
                             gridspec_kw={"hspace": 0.75, "wspace": 0.12})
    for row, t in zip(axes, traj):
        for ax, (key, role) in zip(row, ((t["visconf_top"], "VisConf top"), (t["tokvisconf_top"], "tokenwise top"))):
            s = t["series"][str(key)]
            for name, col, lab in (("c", SERIES[1], "calibrated confidence $c_t$"), ("v", SERIES[3], "visual engagement $v_t$")):
                y = pd.Series(s[name])
                ax.plot(y.index, y, color=col, alpha=0.18, lw=0.5)
                ax.plot(y.index, y.rolling(w, min_periods=1, center=True).mean(), color=col, lw=1.2, label=lab)
            ax.set_title(f"Q{t['sample_id']}, rollout {key} ({role}; {'correct' if s['correct'] else 'wrong'})", fontsize=7)
            ax.set_ylim(0, 1)
        row[0].set_ylabel("value")
    for ax in axes[-1]:
        ax.set_xlabel("Generated token")
    fig.legend(*axes[0][0].get_legend_handles_labels(), loc="lower center", ncol=2, bbox_to_anchor=(0.5, 1.0), fontsize=6.5)
    save(fig, f"{FIG}/trajectories")


# ------------------------------------------------------------------------ appendix tables
def setup_table(cfg, meta):
    st, gr = read_json(rel(f"{R}/data_stats.json")), read_json(rel(f"{R}/grader.json"))
    o = read_json(rel(f"{R}/offline/meta.json"))
    lc, env = meta["config"], meta["env"]
    rows = [("Rollouts", "; ".join(f"{DNAME[d]}: {st[d]['questions']:,} questions, {st[d]['rollouts']:,} rollouts, "
                                    f"{st[d]['capped']} capped, {st[d]['null_extracted']:,} missing extractions" for d in DS)),
            ("Splits", "sorted IDs permuted with default\\_rng(6304); 200 dev, 800 held-out per dataset; live = first 50 held-out MATH"),
            ("Grader", f"{tex_escape(gr['grader_version'])}, math-verify {gr['math_verify']}; MathVista: {tex_escape(gr['mathvista_rule'])}; "
                       f"missing extractions form one NO\\_ANSWER group, always wrong"),
            ("Grader outcome", "; ".join(f"{DNAME[d]}: rollout accuracy {pct(gr[d]['accuracy_all_rollouts'])}\\%, NO\\_ANSWER {pct(gr[d]['no_answer_rate'])}\\%, "
                                         f"{gr[d]['mean_groups_per_question']:.2f} groups/question, {gr[d]['group_vs_individual_label_disagreements']} group/label disagreements"
                                         for d in DS) + f"; MATH fallback grading for {', '.join(tex_escape(q) for q in gr['math']['fallback_questions'])}"),
            ("Offline", f"budgets {', '.join(map(str, o['budgets']))}; {o['orders']} reveal orders; {o['bootstrap']:,} paired question bootstrap resamples; seed {o['seed']}"),
            ("Adaptive", f"checks at {', '.join(map(str, cfg['adaptive']['checkpoints']))}, else 32; $\\tau \\in$ \\{{{', '.join(map(str, cfg['adaptive']['taus']))}\\}}, "
                         f"$\\gamma \\in$ \\{{{', '.join(map(str, cfg['adaptive']['gammas']))}\\}}; tolerance {cfg['adaptive']['tolerance_pp']} pp below dev fixed-32 SC"),
            ("Live model", f"{tex_escape(lc['model_id'])} @ {lc['revision'][:10]}, FP16, batch 1; T={lc['temperature']}, top-p {lc['top_p']}, top-k {lc['top_k']}, "
                           f"repetition penalty {lc['repetition_penalty']}; {env['device']}, torch {env['torch']}, transformers {env['transformers']}")]
    return tabular(r"lp{0.78\linewidth}", "Item & Value", [f"{k} & {v}" for k, v in rows])


def by_n_table(sums, devs, d):
    s = sums[d]
    Ns = sorted(s.N.unique())
    rows = []
    for m in METHOD_ORDER:
        x = s[s.method == m].set_index("N")
        if x.empty:
            continue
        dv = devs[d][(devs[d].method == m) & (devs[d].N == 32)].acc
        rows.append(f"{LABELS[m]} & " + " & ".join(pct(x.loc[n].acc) for n in Ns) + f" & {pct(dv.iloc[0])}")
    sc = s[s.method == "sc"].set_index("N")
    rows[-1] += r" \\ \midrule"
    rows.append("Oracle coverage & " + " & ".join(pct(sc.loc[n].oracle) for n in Ns) + " & ")
    rows.append("Distinct answer groups & " + " & ".join(f"{sc.loc[n].n_groups:.2f}" for n in Ns) + " & ")
    rows.append("Mean tokens (all $N$) & " + " & ".join(f"{sc.loc[n].cost:,.0f}" for n in Ns) + " & ")
    return tabular("l" + "c" * (len(Ns) + 1), "Held-out acc.\\ (\\%) & " + " & ".join(f"$N={n}$" for n in Ns) + r" & Dev, $N=32$", rows)


def decomposition_table(decs):
    rows = []
    for d in DS:
        for m in ("first", "sc", BEST_RWC[d]):
            x = decs[d][decs[d].method == m].set_index("N")
            rows.append(f"{DNAME[d]} & {LABELS[m]} & " + " & ".join(f"{pct(x.loc[n].missed)} / {pct(x.loc[n].none)}" for n in sorted(x.index)))
        rows[-1] += r" \\ \midrule" if d == "math" else ""
    Ns = sorted(decs["math"].N.unique())
    return tabular("ll" + "c" * len(Ns), "Data & Method & " + " & ".join(f"$N={n}$" for n in Ns), rows)


def signals_full(sig):
    rows = []
    for d in DS:
        for r in sig[d]["quartiles"]:
            rows.append(f"{DNAME[d]} & {dict(SIGNALS)[r['score']]} & {r['subset']} ({r['n_questions']}) & {pct(r['top_q_acc'])} & "
                        f"{pct(r['bottom_q_acc'])} & {100 * r['top_minus_bottom']:+.1f} & {r['spearman_with_length']:+.3f}")
        rows[-1] += r" \\ \midrule" if d == "math" else ""
    return tabular("lllcccc", r"Data & Score & Questions & Top-quartile acc. & Bottom-quartile acc. & Diff.\ (pp) & $\rho$ with length", rows)


def dev_grid_table(d, pol):
    g = pd.read_csv(rel(f"{R}/adaptive/dev_grid_{d}.csv"))
    rows = []
    for _, r in g.iterrows():
        chosen = any(r.family == fam and np.isclose(r.tau, pol[d][fam]["tau"]) and
                     (fam == "agreement" or np.isclose(r.gamma, pol[d][fam]["gamma"])) for fam in ("agreement", "quality"))
        cells = [r.family, "--" if pd.isna(r.tau) else f"{r.tau:g}", "--" if pd.isna(r.gamma) else f"{r.gamma:g}",
                 pct(r.acc), f"{r.cost:,.0f}", f"{r.rollouts:.1f}"]
        line = " & ".join(cells)
        rows.append(r"\textbf{" + line.replace(" & ", r"} & \textbf{") + "}" if chosen else line)
    return tabular("lccccc", r"Family & $\tau$ & $\gamma$ & Dev acc.\ (\%) & Tokens & Rollouts", rows)


def live_detail(recs):
    par = [c for r in recs for c in r["parallel"]]
    pairs = {}
    for a, b in (("single", "parallel_sc"), ("single", "sequential"), ("parallel_sc", "sequential")):
        c = pd.Series([(bool(r["correct"][a]), bool(r["correct"][b])) for r in recs]).value_counts()
        pairs[(a, b)] = c
    nm = {"single": "Single", "parallel_sc": "Parallel", "sequential": "Sequential"}
    rows = [f"{nm[a]} vs {nm[b]} & {c.get((True, True), 0)} & {c.get((True, False), 0)} & {c.get((False, True), 0)} & {c.get((False, False), 0)}"
            for (a, b), c in pairs.items()]
    rows[-1] += r" \\ \midrule"
    rows.append(rf"\multicolumn{{5}}{{p{{0.9\linewidth}}}}{{Parallel responses: {sum(c['truncated'] for c in par)} of {len(par)} hit the {recs[0]['parallel'][0]['cap']:,}-token cap; "
                rf"{sum(c['extracted'] is None for c in par)} had no extracted answer; vote returned no answer for {sum(r['answers']['parallel_sc'] is None for r in recs)} questions}}")
    rows.append(rf"\multicolumn{{5}}{{p{{0.9\linewidth}}}}{{At least one of three parallel responses correct: {sum(any(c['correct'] for c in r['parallel']) for r in recs)} questions; "
                rf"initial answer missing: {sum(r['answers']['initial'] is None for r in recs)}, revised answer missing: {sum(r['answers']['sequential'] is None for r in recs)}}}")
    return tabular("lcccc", r"Pair (A vs B) & Both right & Only A right & Only B right & Both wrong", rows)


def run(cfg):
    setup()
    sp, sums, devs, decs = load(cfg)
    check_offline(sums)
    pol = read_json(rel(f"{R}/adaptive/selected_policies.json"))
    sims = check_adaptive(cfg, pol)
    recs, _ = check_live()
    meta = read_json(rel(f"{R}/live/meta.json"))
    sig = read_json(rel(f"{R}/signals/summary.json"))
    cv = pd.read_csv(rel(f"{R}/signals/conf_vision_per_rollout.csv"))
    traj = read_json(rel(f"{R}/signals/trajectories.json"))
    raw = raw_tables(cfg)
    dis, rev, pre = disagreements(sp, raw), revisions(recs), premature(sp, sims, raw)
    out, app = ensure_dir(f"{R}/summary"), ensure_dir(f"{R}/summary/appendix")
    write_json({"rules": "see task3_ttc/evidence.py docstring",
                "disagreement": {d: {k: v["sample_id"] for k, v in dis[d].items()} for d in DS},
                "revision": {k: r["sample_id"] for k, r in rev.items()},
                "main_table": {"disagreement": next(iter(dis["math"].values()))["sample_id"],
                               "revision": rev["first_changed"]["sample_id"], "premature": pre[("math", "agreement")]["sample_id"]},
                "premature": {f"{d}/{f}": e["sample_id"] for (d, f), e in pre.items()}}, out / "qualitative_selection.json")
    files = {out / "methods_diff_table.tex": methods_table(sums), out / "signals_table.tex": signals_table(sig, cv),
             out / "live_table.tex": live_table(meta), out / "adaptive_table.tex": adaptive_table(pol, sums),
             out / "qual_main.tex": qual_main(dis, rev, pre),
             app / "setup.tex": setup_table(cfg, meta), app / "by_n_math.tex": by_n_table(sums, devs, "math"),
             app / "by_n_mathvista.tex": by_n_table(sums, devs, "mathvista"), app / "decomposition.tex": decomposition_table(decs),
             app / "signals.tex": signals_full(sig), app / "dev_grid_math.tex": dev_grid_table("math", pol),
             app / "dev_grid_mathvista.tex": dev_grid_table("mathvista", pol), app / "live_detail.tex": live_detail(recs),
             app / "qual.tex": qual_appendix(dis, rev, pre)}
    for p, t in files.items():
        p.write_text(t, encoding="utf-8")
    adaptive_figure(sums, pol)
    trajectory_figure(traj)
    print("checks passed (offline, adaptive re-simulation, live); wrote", out)
    print(json.dumps(read_json(out / "qualitative_selection.json"), indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task3.yaml")
    run(load_config(ap.parse_args().config))


if __name__ == "__main__":
    main()

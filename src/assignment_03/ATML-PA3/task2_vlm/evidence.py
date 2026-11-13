"""Task 2 report evidence not covered by summarize.latex_final.

Reads only saved results (no model, no GPU) and writes to results/task2/summary/:
  type_table.tex, resources_table.tex, qual_main.tex, by_type.csv, qualitative_selection.json
  appendix/{settings,curriculum,data,dev_log,by_type,swap,diagnostics,qual_<type>}.tex
plus report/figures/task2/swap_evidence.{pdf,png} and thumbnails in report/figures/task2/examples/.
Called from summarize.run; can also be run alone (this path never touches the run store):

    python -m task2_vlm.evidence

Before writing anything, overall metrics and the pair bootstrap are recomputed from
predictions.csv and compared with metrics.csv / bootstrap.json. Per-type intervals use the same
pair_bootstrap function, seed and resample count as the saved overall intervals.

Qualitative selection rule (fixed, no manual choice). Each test pair falls into exactly one
outcome from its two correct-image responses: both_right (both match their canonical answers),
identical (same normalised response for both images), or changed_wrong (responses differ, not
both right). The appendix shows the first pair in file order for every type x outcome. The main
table shows, per type, the first pair of that type's most frequent outcome.
"""
from __future__ import annotations

import argparse
import collections
import io
import urllib.request

import numpy as np
import pandas as pd

from common.plotting import INK2, NEUTRAL, SERIES, TEXT_WIDTH, save, setup
from common.utils import ensure_dir, load_config, read_json, read_jsonl, rel, write_json
from task1_finetune.appendix import tabular, tex_escape
from task2_vlm.evaluate import CONDS, metrics, pair_bootstrap
from task2_vlm.train import stage_bounds

R = "results/task2"
THUMB_DIR = "report/figures/task2/examples"
THUMB_TEX = "task2/examples"              # relative to \graphicspath{{figures/}}
THUMB_SIDE = 256
TYPES = [("grounding", "presence"), ("grounding", "counting"), ("vqa", "yes/no"), ("vqa", "number"), ("vqa", "other")]
SUITE = {"grounding": "Grounding", "vqa": "VQAv2"}
COND_LABEL = {"correct": "Correct", "complementary": "Partner", "mismatched": "Unrelated", "none": "None"}
OUTCOMES = ["both_right", "identical", "changed_wrong"]
OUTCOME_LABEL = {"both_right": "both right", "identical": "identical", "changed_wrong": "changed, not both right"}
SWAP_KINDS = [("follows", "Follows partner", SERIES[6]), ("keeps", "Keeps original", SERIES[3]), ("other", "Other", NEUTRAL)]
NONE_CHARS = 60                           # characters of the no-image response kept in the main table


def load_predictions():
    # keep_default_na=False: "none", "nan" and empty responses are text, not missing values
    return pd.read_csv(rel(f"{R}/eval/predictions.csv"), keep_default_na=False)


def verify(p, cfg):
    """Stop if recomputation from predictions.csv disagrees with the saved evaluation files."""
    saved = pd.read_csv(rel(f"{R}/eval/metrics.csv")).set_index(["suite", "condition"])
    new = metrics(p).set_index(["suite", "condition"])
    for col in ("accuracy", "pair_accuracy", "identical_answer_rate", "mean_ref_nll"):
        d = (new[col] - saved[col]).abs().max()
        assert d < 1e-9, f"metrics.csv mismatch in {col}: {d}"
    boot = read_json(rel(f"{R}/eval/bootstrap.json"))["bootstrap"]
    for suite, d in p.groupby("suite"):
        b = pair_bootstrap(d, cfg["eval"]["n_boot"], cfg["seed"])
        for k, v in b.items():
            for q in ("mean", "lo", "hi"):
                assert abs(v[q] - boot[suite][k][q]) < 1e-9, f"bootstrap mismatch {suite} {k} {q}"
    return boot


def type_stats(p, cfg):
    """One row per (suite, type) plus an overall row per suite, for each condition."""
    rows = []
    groups = [(s, "all", d) for s, d in p.groupby("suite")] + [(s, t, p[(p.suite == s) & (p.qtype == t)]) for s, t in TYPES]
    for suite, qtype, d in groups:
        b = pair_bootstrap(d, cfg["eval"]["n_boot"], cfg["seed"])
        for cond in CONDS:
            g = d[d.condition == cond]
            pair = g.groupby("pair_id").agg(both=("canon_match", "all"), same=("pred_norm", lambda s: s.nunique() == 1))
            rows.append({"suite": suite, "qtype": qtype, "condition": cond, "pairs": g.pair_id.nunique(),
                         "accuracy": g.score.mean(), "acc_lo": b[f"score_{cond}"]["lo"], "acc_hi": b[f"score_{cond}"]["hi"],
                         "pair_accuracy": pair.both.mean(), "identical": pair.same.mean(), "ref_nll": g.nll.mean(),
                         "score_adv": b["score_advantage"]["mean"], "score_adv_lo": b["score_advantage"]["lo"],
                         "score_adv_hi": b["score_advantage"]["hi"], "nll_adv": b["nll_advantage"]["mean"],
                         "nll_adv_lo": b["nll_advantage"]["lo"], "nll_adv_hi": b["nll_advantage"]["hi"]})
    return pd.DataFrame(rows)


def type_label(suite, qtype):
    return f"{SUITE[suite]}, all" if qtype == "all" else qtype.capitalize()


def type_table(st):
    head = (r"Type & Pairs & Correct [95\% CI] & Partner & Unrelated & None & Pair acc. & Identical & Ref.\ NLL & "
            r"Score adv.\ (pp) [95\% CI] & NLL adv.\ [95\% CI]")
    rows, order = [], [("grounding", "presence"), ("grounding", "counting"), ("grounding", "all"),
                       ("vqa", "yes/no"), ("vqa", "number"), ("vqa", "other"), ("vqa", "all")]
    for s, t in order:
        d = st[(st.suite == s) & (st.qtype == t)].set_index("condition")
        c = d.loc["correct"]
        row = (f"{type_label(s, t)} & {int(c.pairs)} & {100 * c.accuracy:.1f} [{100 * c.acc_lo:.1f}, {100 * c.acc_hi:.1f}] & "
               f"{100 * d.loc['complementary'].accuracy:.1f} & {100 * d.loc['mismatched'].accuracy:.1f} & {100 * d.loc['none'].accuracy:.1f} & "
               f"{100 * c.pair_accuracy:.1f} & {100 * c.identical:.1f} & {c.ref_nll:.3f} & "
               f"{100 * c.score_adv:+.1f} [{100 * c.score_adv_lo:.1f}, {100 * c.score_adv_hi:.1f}] & "
               f"{c.nll_adv:.2f} [{c.nll_adv_lo:.2f}, {c.nll_adv_hi:.2f}]")
        rows.append((r"\textbf{" + row.replace(" & ", r"} & \textbf{") + "}") if t == "all" else row)
    return tabular("lccccccccc" + "c", head, rows)


def swap_counts(p):
    c = p[p.condition == "complementary"]
    kind = np.where(c.follows_partner.astype(bool), "follows", np.where(c.canon_match.astype(bool), "keeps", "other"))
    return pd.crosstab([c.suite, c.qtype], kind).reindex(columns=["follows", "keeps", "other"], fill_value=0)


def swap_figure(cfg, counts):
    import matplotlib.pyplot as plt

    dev = pd.DataFrame(read_jsonl(rel(f"{R}/train/dev_log.jsonl")))
    fig, (a, b) = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.0), gridspec_kw={"wspace": 0.42})
    for suite, col in (("grounding", SERIES[0]), ("vqa", SERIES[1])):
        a.plot(dev["update"], dev[f"{suite}_swap_penalty"], color=col, marker="o", ms=3, label=SUITE[suite])
    for x in [s[1] for s in stage_bounds(cfg)][:-1]:
        a.axvline(x, color=INK2, lw=0.6, ls=":")
    a.axhline(0, color="#bbbbbb", lw=0.8)
    a.set_xlabel("Update"); a.set_ylabel(r"$\Delta$NLL (partner $-$ correct)")
    a.set_title("Dev swap penalty"); a.legend()

    labels = [t.capitalize() for _, t in TYPES][::-1]
    keys = TYPES[::-1]
    left = np.zeros(len(keys))
    for kind, name, col in SWAP_KINDS:
        n = np.array([counts.loc[k, kind] for k in keys])
        pct = 100 * n / np.array([counts.loc[k].sum() for k in keys])
        b.barh(labels, pct, left=left, color=col, edgecolor="white", linewidth=1.0, label=name, height=0.72)
        for y, (w, x, cnt) in enumerate(zip(pct, left, n)):
            if cnt:
                b.text(x + w / 2, y, str(cnt), ha="center", va="center", fontsize=6.5,
                       color="#0b0b0b" if col == SERIES[3] else "white")
        left += pct
    b.set_xlim(0, 100); b.set_xlabel("Responses under partner image (%; labels: counts)")
    b.set_title("Final-test swap behaviour"); b.grid(axis="y", visible=False)
    b.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.42, 1.09), columnspacing=0.8, handlelength=1.0, fontsize=6.5)
    save(fig, "report/figures/task2/swap_evidence")


def resources_table(cfg, p):
    s = read_json(rel(f"{R}/train/summary.json"))
    fc = read_json(rel(f"{R}/resources_feature_cache.json")) if rel(f"{R}/resources_feature_cache.json").exists() \
        else read_json(rel(f"{R}/summary/resources.json"))["feature_cache"]
    gpu = read_json(rel(f"{R}/eval/bootstrap.json"))["gpu"]
    head = r"Phase & Work & Time & Peak GPU (GiB) & Trainable parameters"
    rows = [f"Feature caching & {fc['n_images']:,} images, {fc['feature_bytes'] / 1e9:.2f} GB & "
            f"{fc['download_seconds']:.1f} s download, {fc['feature_seconds']:.1f} s encoding & {fc['peak_allocated_gb']:.3f} & 0"]
    for i, st in enumerate(cfg["train"]["stages"], 1):
        n, sec = st["updates"], s["stage_seconds"][st["name"]]
        rows.append(f"Stage {i} ({st['name']}) & {n:,} updates & {sec:,.1f} s ({sec / n:.3f} s/update) & "
                    f"{s['stage_peak_gb'][st['name']]:.3f} & {s['trainable_by_stage'][st['name']]:,}")
    rows.append(f"Training total & {s['updates']:,} updates & {s['train_hours']:.3f} h & {max(s['stage_peak_gb'].values()):.3f} & --")
    t = p.groupby("condition").gen_s_per_answer.agg(["min", "max"])
    img = t.drop(index="none")
    rows.append(f"Final evaluation & {len(p):,} responses & {img['min'].min():.3f}--{img['max'].max():.3f} s/answer with image, "
                f"{t.loc['none', 'min']:.3f}--{t.loc['none', 'max']:.3f} without & {gpu['peak_allocated_gb']:.3f} & 0")
    return tabular("llllr", head, rows)


# --------------------------------------------------------------------- qualitative examples
def outcomes(p):
    rows = []
    for (s, t, pid), g in p[p.condition == "correct"].groupby(["suite", "qtype", "pair_id"], sort=False):
        o = "both_right" if g.canon_match.astype(bool).all() else ("identical" if g.pred_norm.nunique() == 1 else "changed_wrong")
        rows.append({"suite": s, "qtype": t, "pair_id": pid, "outcome": o})
    return pd.DataFrame(rows)


def select(o):
    grid, main = {}, {}
    for s, t in TYPES:
        d = o[(o.suite == s) & (o.qtype == t)]
        grid[t] = {k: d[d.outcome == k].pair_id.iloc[0] for k in OUTCOMES if (d.outcome == k).any()}
        top = d.outcome.value_counts().idxmax()
        main[t] = {"outcome": top, "pair_id": d[d.outcome == top].pair_id.iloc[0],
                   "outcome_counts": {k: int((d.outcome == k).sum()) for k in OUTCOMES}}
    return grid, main


def fetch_thumbnails(cfg, pair_ids, p):
    from PIL import Image

    out = ensure_dir(THUMB_DIR)
    ids = sorted({int(i) for i in p[p.pair_id.isin(pair_ids)].image_id})
    for i in ids:
        path = out / f"{i}.jpg"
        if path.exists():
            continue
        url = cfg["data"]["coco_image_url"].format(split="val2014", id=i)   # all evaluation images are val2014
        with urllib.request.urlopen(url, timeout=60) as r:
            im = Image.open(io.BytesIO(r.read())).convert("RGB")
        im.thumbnail((THUMB_SIDE, THUMB_SIDE))
        im.save(path, quality=82)
    return ids


def references(suite_rows):
    """(pair_id, image_id) -> reference text: the answer, or the ten VQA answers summarised."""
    refs = {}
    for it in suite_rows["grounding"]:
        for x in it["items"]:
            refs[(it["pair_id"], x["image_id"])] = {"short": tex_escape(str(x["answer"])), "full": tex_escape(str(x["answer"]))}
    for it in suite_rows["vqa"]:
        for x in it["items"]:
            c = collections.Counter(x["answers"]).most_common()
            refs[(it["pair_id"], x["image_id"])] = {
                "short": f"{tex_escape(x['consensus'])} ({dict(c).get(x['consensus'], 0)}/10)",
                "full": ", ".join(f"{tex_escape(a)}$\\times${n}" for a, n in c)}
    return refs


def thumb(image_id, h="0.46in", w="0.56in"):
    return rf"\includegraphics[height={h},width={w},keepaspectratio]{{{THUMB_TEX}/{image_id}.jpg}}"


def resp(g, cond, k):
    v = g[(g.condition == cond) & (g.k == k)].prediction.iloc[0]
    return tex_escape(str(v)) if str(v).strip() else r"\textit{(empty)}"


def qual_main(p, main, refs):
    spec = (r"@{}>{\raggedright\arraybackslash}m{0.075\linewidth}>{\raggedright\arraybackslash}m{0.15\linewidth}"
            r"m{0.105\linewidth}m{0.105\linewidth}>{\raggedright\arraybackslash}m{0.105\linewidth}"
            r">{\raggedright\arraybackslash}m{0.075\linewidth}>{\raggedright\arraybackslash}m{0.075\linewidth}"
            r">{\raggedright\arraybackslash}m{0.15\linewidth}@{}")
    head = r"Type (outcome) & Question & Image A & Image B & Reference A / B & Correct A / B & Unrelated A / B & No image (first " + str(NONE_CHARS) + " chars)"
    rows = []
    for _, t in TYPES:
        pid, g = main[t]["pair_id"], p[p.pair_id == main[t]["pair_id"]]
        a, b = g[g.k == 0].image_id.iloc[0], g[g.k == 1].image_id.iloc[0]
        none = str(g[g.condition == "none"].prediction.iloc[0])
        none = tex_escape(none[:NONE_CHARS]) + (r"\ldots" if len(none) > NONE_CHARS else "")
        rows.append(f"{t.capitalize()} ({OUTCOME_LABEL[main[t]['outcome']]}) & {tex_escape(g.question.iloc[0])} & {thumb(a)} & {thumb(b)} & "
                    rf"{refs[(pid, a)]['short']} / {refs[(pid, b)]['short']} & {resp(g, 'correct', 0)} / {resp(g, 'correct', 1)} & "
                    rf"{resp(g, 'mismatched', 0)} / {resp(g, 'mismatched', 1)} & {none}")
    return tabular(spec, head, rows)


def qual_appendix(p, grid, refs, t):
    spec = (r"@{}>{\raggedright\arraybackslash}m{0.14\linewidth}m{0.105\linewidth}>{\raggedright\arraybackslash}m{0.17\linewidth}"
            r">{\raggedright\arraybackslash}m{0.08\linewidth}>{\raggedright\arraybackslash}m{0.08\linewidth}"
            r">{\raggedright\arraybackslash}m{0.08\linewidth}>{\raggedright\arraybackslash}m{0.235\linewidth}@{}")
    head = r"Outcome, question & Image & References & Correct & Partner & Unrelated & No image"
    rows = []
    for outcome, pid in grid[t].items():
        g = p[p.pair_id == pid]
        for k in (0, 1):
            img = g[g.k == k].image_id.iloc[0]
            lead = f"\\textit{{{OUTCOME_LABEL[outcome]}}}: {tex_escape(g.question.iloc[0])}" if k == 0 else ""
            rows.append(f"{lead} & {thumb(img)} & {refs[(pid, img)]['full']} & {resp(g, 'correct', k)} & {resp(g, 'complementary', k)} & "
                        f"{resp(g, 'mismatched', k)} & {resp(g, 'none', k)}")
        rows[-1] += r" \\ \addlinespace" if outcome != list(grid[t])[-1] else ""
    # a row ending in "\\ \addlinespace" would otherwise be followed by an empty row
    return tabular(spec, head, rows).replace(" \\\\ \\addlinespace \\\\", " \\\\ \\addlinespace")


# --------------------------------------------------------------------- appendix tables
def settings_table(cfg):
    t, lo, cal = cfg["train"], cfg["train"]["lora"], read_json(rel(f"{R}/train/connector_calibration.json"))
    rows = [("Vision encoder", tex_escape(cfg["vision_id"]) + f", frozen; {cfg['n_visual_tokens']} tokens of size 768, FP16 cache"),
            ("Connector", r"Linear 768$\to$1024, GELU, Linear 1024$\to$1024"),
            ("Language model", tex_escape(cfg["model_id"]) + ", original weights frozen, thinking disabled"),
            ("LoRA (Stages 2--3)", rf"$r={lo['r']}$, $\alpha={lo['alpha']}$, dropout {lo['dropout']}; targets " + tex_escape(", ".join(lo["targets"]))),
            ("Visual scale", f"{cal['scale']:.5f} = text norm {cal['text_norm']:.4f} / initial visual norm {cal['visual_norm_init']:.4f}"),
            ("Optimizer", f"AdamW (new instance each stage), weight decay {t['weight_decay']}, clip {t['clip']}, FP16 autocast with GradScaler"),
            ("Schedule", f"{100 * t['warmup_frac']:.0f}\\% warmup, cosine to {100 * t['min_lr_frac']:.0f}\\% of peak, per stage"),
            ("Batch", f"microbatch {t['microbatch']} $\\times$ {t['grad_accum']} accumulation = {t['microbatch'] * t['grad_accum']}"),
            ("Token caps", ", ".join(f"{k} {v}" for k, v in t["caps"].items())),
            ("Caption prompt", tex_escape(cfg["caption_prompt"])),
            ("Generation", f"greedy, {cfg['eval']['max_new_tokens']} new tokens, batch {cfg['eval']['batch_size']}"),
            ("Bootstrap", f"{cfg['eval']['n_boot']:,} resamples of complete pairs, seed {cfg['seed']}"),
            ("Optional 49-token run", "not run")]
    return tabular(r"lp{0.72\linewidth}", "Setting & Value", [f"{k} & {v}" for k, v in rows])


def sci(x):
    if x is None:
        return "--"
    m, e = f"{x:.0e}".split("e")
    return rf"${m}\times 10^{{{int(e)}}}$" if m != "1" else rf"$10^{{{int(e)}}}$"


def curriculum_table(cfg):
    log = pd.DataFrame(read_jsonl(rel(f"{R}/train/log.jsonl")))
    summ = read_json(rel(f"{R}/train/summary.json"))
    rows = []
    for i, st in enumerate(cfg["train"]["stages"], 1):
        g = log[log.stage == st["name"]]
        m = collections.Counter()
        for d in g["mix"]:
            m.update(d)
        total = sum(m.values())
        keys = list(st["mix"])
        target = ", ".join(f"{k} {100 * st['mix'][k]:.0f}" for k in keys)
        real = ", ".join(f"{k} {100 * m[k] / total:.1f}" for k in keys)
        skipped = g[g["skipped"]]["update"].tolist()
        rows.append(f"{i} & {st['updates']:,} & {sci(st['connector_lr'])} & {sci(st['lora_lr'])} & {target} & {real} & "
                    f"{g['loss'].head(50).mean():.3f} $\\to$ {g['loss'].tail(50).mean():.3f} & "
                    f"{len(skipped)}{' (' + ', '.join(map(str, skipped)) + ')' if skipped else ''}")
    assert summ["skipped_updates"] == log["skipped"].sum()
    return tabular(r"cccc>{\raggedright\arraybackslash}p{0.17\linewidth}>{\raggedright\arraybackslash}p{0.19\linewidth}cl",
                   r"Stage & Updates & Conn.\ LR & LoRA LR & Target mix (\%) & Realised mix (\%) & Train loss (first $\to$ last 50) & Skipped", rows)


def data_table():
    s = read_json(rel(f"{R}/data/stats.json"))
    J = lambda f: read_json(rel(f"{R}/data/{f}.json"))
    def images(ps):
        return len({x["image_id"] for q in ps for x in q["items"]})
    mm = J("mismatched_test")
    rows = [("Images", f"{s['n_images']:,} total: {s['n_train_images']:,} train2014, {s['n_eval_images']} val2014 (evaluation)"),
            ("Captions", f"{s['captions_train_examples']:,} training examples; {len({c['image_id'] for c in J('captions_dev')})} dev images"),
            ("Grounding pools", ", ".join(f"{k} {v:,}" for k, v in s["grounding_pool"].items()) + " training pairs"),
            ("Grounding dev / test", f"{sum(s['grounding_dev'].values())} / {sum(s['grounding_test'].values())} pairs "
                                     f"({images(J('grounding_dev'))} / {images(J('grounding_test'))} images)"),
            ("VQAv2 candidates", f"train {s['vqa_candidates']['train']:,}, val {s['vqa_candidates']['val']:,}"),
            ("VQAv2 train", ", ".join(f"{tex_escape(k)} {v}" for k, v in s["vqa_train"].items()) + " pairs"),
            ("VQAv2 dev / test", f"{sum(s['vqa_dev'].values())} / {sum(s['vqa_test'].values())} pairs "
                                 f"({images(J('vqa_dev'))} / {images(J('vqa_test'))} images)"),
            ("Mismatched mapping", f"{len(mm)} fixed entries (unrelated image from the same suite)")]
    return tabular(r"lp{0.72\linewidth}", "Item & Value", [f"{k} & {v}" for k, v in rows])


def dev_table(cfg):
    d = pd.DataFrame(read_jsonl(rel(f"{R}/train/dev_log.jsonl")))
    head = (r"Update & Caption loss & \multicolumn{4}{c}{Grounding NLL} & \multicolumn{4}{c}{VQAv2 NLL} \\ \cmidrule(lr){3-6}\cmidrule(lr){7-10}"
            "\n" r" & & Correct & Partner & None & $\Delta$ & Correct & Partner & None & $\Delta$")
    rows = [f"{int(r['update']):,} & {r['dev_caption_loss']:.3f} & " + " & ".join(
        f"{r[f'{s}_{c}']:.3f}" for s in ("grounding", "vqa") for c in ("nll_correct", "nll_complementary", "nll_none", "swap_penalty"))
        for _, r in d.iterrows()]
    return tabular("rccccccccc", head, rows)


def by_type_table(st):
    rows, order = [], [("grounding", "presence"), ("grounding", "counting"), ("grounding", "all"), ("vqa", "yes/no"), ("vqa", "number"), ("vqa", "other"), ("vqa", "all")]
    for s, t in order:
        d = st[(st.suite == s) & (st.qtype == t)].set_index("condition")
        for c in CONDS:
            r = d.loc[c]
            rows.append(f"{type_label(s, t) if c == 'correct' else ''} & {COND_LABEL[c]} & {100 * r.accuracy:.1f} [{100 * r.acc_lo:.1f}, {100 * r.acc_hi:.1f}] & "
                        f"{100 * r.pair_accuracy:.1f} & {100 * r.identical:.1f} & {r.ref_nll:.3f}")
    return tabular("llcccc", r"Type & Image & Score [95\% CI] & Pair acc. & Identical & Ref.\ NLL", rows)


def swap_table(counts):
    rows = []
    for s, t in TYPES:
        c = counts.loc[(s, t)]
        n = c.sum()
        rows.append(f"{SUITE[s]} & {t.capitalize()} & {n} & " + " & ".join(f"{c[k]} ({100 * c[k] / n:.1f})" for k in ("follows", "keeps", "other")))
    for s in ("grounding", "vqa"):
        c = counts.loc[s].sum()
        n = c.sum()
        rows.append(rf"{SUITE[s]} & \textbf{{All}} & {n} & " + " & ".join(f"{c[k]} ({100 * c[k] / n:.1f})" for k in ("follows", "keeps", "other")))
    return tabular("llcccc", r"Suite & Type & Responses & Follows partner (\%) & Keeps original (\%) & Other (\%)", rows)


def first_word(s):
    w = str(s).split()
    return w[0].strip(",.!?\"'").lower() if w else ""


def diagnostics_table(p):
    rows = []
    n = p[p.condition == "none"].assign(first=lambda d: d.prediction.map(first_word))
    for s, t in TYPES:
        g = n[(n.suite == s) & (n.qtype == t)]
        top = ", ".join(f"{tex_escape(a)} {b}" for a, b in g["first"].value_counts().head(3).items())
        rows.append(f"No-image first word & {t.capitalize()} ({len(g)} responses) & {top}; median length {int(g.prediction.str.len().median())} chars")
    for cond in ("correct", "mismatched"):
        g = p[(p.condition == cond) & (p.qtype == "presence")]
        acc = {r: 100 * g[g.canonical == r].score.mean() for r in ("yes", "no")}
        pr = ", ".join(f"{tex_escape(a)} {b}" for a, b in g.pred_norm.value_counts().head(3).items())
        rows.append(f"Presence by reference & {COND_LABEL[cond]} image & score {acc['yes']:.1f}\\% on ``yes'', {acc['no']:.1f}\\% on ``no''; responses: {pr}")
    g = p[(p.condition == "correct") & (p.qtype == "counting")]
    dist = g.canonical.astype(str).value_counts(normalize=True).sort_index()
    rows.append("Counting references & Canonical answers & " + ", ".join(f"``{a}'' {100 * b:.1f}\\%" for a, b in dist.items()))
    for s in ("grounding", "vqa"):
        c = p[(p.suite == s) & (p.condition == "correct")].set_index(["pair_id", "image_id"]).pred_norm
        x = p[(p.suite == s) & (p.condition == "complementary")].set_index(["pair_id", "partner_image_id"]).pred_norm
        x.index.names = ["pair_id", "image_id"]
        j = c.to_frame("c").join(x.to_frame("x"))
        rows.append(f"Swap vs.\\ correct & {SUITE[s]} & partner-image response equals that image's correct-image response in {(j.c == j.x).sum()} of {len(j)} cases")
    return tabular(r"ll>{\raggedright\arraybackslash}p{0.47\linewidth}", "Check & Subset & Result", rows)


def run(cfg):
    setup()
    p = load_predictions()
    verify(p, cfg)
    out = ensure_dir(f"{R}/summary")
    app = ensure_dir(f"{R}/summary/appendix")
    st = type_stats(p, cfg)
    st.to_csv(out / "by_type.csv", index=False)
    counts = swap_counts(p)
    swap_figure(cfg, counts)
    grid, main = select(outcomes(p))
    ids = {q for g in grid.values() for q in g.values()} | {m["pair_id"] for m in main.values()}
    write_json({"rule": "see task2_vlm/evidence.py docstring", "grid": grid, "main": main}, out / "qualitative_selection.json")
    fetch_thumbnails(cfg, ids, p)
    refs = references({"grounding": read_json(rel(f"{R}/data/grounding_test.json")), "vqa": read_json(rel(f"{R}/data/vqa_test.json"))})
    (out / "type_table.tex").write_text(type_table(st), encoding="utf-8")
    (out / "resources_table.tex").write_text(resources_table(cfg, p), encoding="utf-8")
    (out / "qual_main.tex").write_text(qual_main(p, main, refs), encoding="utf-8")
    for name, text in (("settings", settings_table(cfg)), ("curriculum", curriculum_table(cfg)), ("data", data_table()),
                       ("dev_log", dev_table(cfg)), ("by_type", by_type_table(st)), ("swap", swap_table(counts)),
                       ("diagnostics", diagnostics_table(p))):
        (app / f"{name}.tex").write_text(text, encoding="utf-8")
    for _, t in TYPES:
        (app / f"qual_{t.replace('/', '_')}.tex").write_text(qual_appendix(p, grid, refs, t), encoding="utf-8")
    print("verified against metrics.csv and bootstrap.json; wrote", out)
    print(pd.DataFrame(main).T[["outcome", "pair_id", "outcome_counts"]].to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task2.yaml")
    run(load_config(ap.parse_args().config))


if __name__ == "__main__":
    main()

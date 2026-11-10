#!/usr/bin/env python3
"""Print every number the papers report, recomputed from the collected runs.

This is the whole results surface in one place: for each group it prints one row per
(sequence, system), with each metric as a mean over the runs and the run-to-run standard
deviation beside it. The papers lay the same numbers out differently from one part to the next;
rather than reproduce four LaTeX layouts, this prints one uniform table you can read any of them
against.

    python3 print_tables.py                  # all four groups
    python3 print_tables.py --group 3        # one group
    python3 print_tables.py --runs           # per-run values instead of aggregates
    python3 print_tables.py --csv            # machine-readable, one row per (seq, system)
    python3 print_tables.py --runs --csv     # machine-readable, one row per run

Standard deviations are POPULATION (NumPy's default, dividing by n), matching the papers'
captions. At n=3 the sample convention would be 1.22x larger.

"ATE unal." is absolute trajectory error WITHOUT the SE(3) alignment the other ATE column
applies. It is only meaningful for group 4, where both systems start from a ground-truth seed
pose in a fixed map frame, so an unaligned score is a real measurement. Groups 1-3 start at the
origin in their own frame, so their unaligned numbers measure that frame offset, not tracking --
expect values in the thousands there, and ignore them.

Partial runs: a run whose recorded trajectory spans less than --coverage of the bag is marked
with a dagger and listed under the table. They are INCLUDED by default, because the papers do
not treat them uniformly -- Part 1 reports FAST-LIO2's truncated urban runs with a footnote,
Part 3 excludes RTAB-Map's truncated forest run and reports that cell as n=2. Pass --drop-partial
to exclude them and reproduce Part 3's n=2 cell.
"""
import argparse
import sys

import numpy as np

import metrics_lib as ml
from collect_metrics import ALGORITHMS, SEQUENCES

# (column header, metric key, unit scale, decimals). Every metric either paper reports.
COLUMNS = [
    ("drift/10m",  "drift_trans_rmse_m",        1.0,  3),
    ("RPE tr.",    "rpe_trans_rmse_m",          1.0,  3),
    ("RPE rot",    "rpe_rot_rmse_deg",          1.0,  3),
    ("ATE",        "ate_trans_rmse_m",          1.0,  3),
    ("ATE unal.",  "ate_trans_rmse_unaligned_m", 1.0, 3),
    ("median",     "ate_trans_median_m",        1.0,  3),
    ("max",        "ate_trans_max_m",           1.0,  2),
    ("RSS GB",     "peak_rss_mb",               1e-3, 2),
    ("cores",      "avg_cpu_cores",             1.0,  2),
    ("RTF",        "rtf",                       1.0,  4),
]

GROUPS = {}
for _algo, _cfg in ALGORITHMS.items():
    GROUPS.setdefault(_cfg["group"], []).append(_algo)


def coverage(group, algo, seq, run):
    """What fraction of the bag the recorded trajectory spans.

    A run that died partway through is still scored over the poses it produced, so it can post a
    respectable error while covering half the route. That makes coverage the first thing to look
    at, not an afterthought -- see the note in the module docstring.
    """
    meta = ml.load_meta(group, algo, seq, run)
    if meta is None:
        return None
    est = np.loadtxt(meta["estimated_trajectory_path"])
    gt = np.loadtxt(meta["ground_truth_path"])
    if est.ndim != 2 or len(est) < 2:
        return 0.0
    return float((est[-1, 0] - est[0, 0]) / (gt[-1, 0] - gt[0, 0]))


def collect(group, algo, seq, min_coverage, drop_partial):
    """Usable runs as (run number, metrics dict, coverage), newest layout first."""
    out = []
    for run in (1, 2, 3):
        data = ml.load_run(group, algo, seq, run)
        if data is None:
            continue
        cov = coverage(group, algo, seq, run)
        partial = cov is not None and cov < min_coverage
        if partial and drop_partial:
            continue
        out.append((run, data, cov, partial))
    return out


def fmt(value, decimals):
    return "--" if value is None else "%.*f" % (decimals, value)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", choices=["1", "2", "3", "4"], action="append",
                    help="print only this group; repeatable (default: all four)")
    ap.add_argument("--runs", action="store_true", help="per-run values instead of aggregates")
    ap.add_argument("--csv", action="store_true", help="machine-readable output")
    ap.add_argument("--drop-partial", action="store_true",
                    help="exclude runs below --coverage instead of flagging them")
    ap.add_argument("--coverage", type=float, default=0.90, metavar="FRAC",
                    help="a run below this fraction of the bag counts as partial (default 0.90)")
    args = ap.parse_args()

    wanted = [g for g in sorted(GROUPS)
              if not args.group or any(g.startswith("group" + n) for n in args.group)]

    if args.csv:
        head = ["group", "sequence", "system"]
        head += ["run", "coverage"] if args.runs else ["n"]
        for name, _, _, _ in COLUMNS:
            head += [name] if args.runs else [name, name + "_sd"]
        print(",".join(head))

    footnotes = []
    for group in wanted:
        if not args.csv:
            print("\n=== %s ===" % group)
            header = "%-9s %-22s %3s %s" % (
                "sequence", "system", "n", " ".join("%12s" % c[0] for c in COLUMNS))
            print(header)
            print("-" * len(header))

        for algo in GROUPS[group]:
            for seq in SEQUENCES:
                rows = collect(group, algo, seq, args.coverage, args.drop_partial)
                if not rows:
                    continue
                if args.runs:
                    for run, data, cov, partial in rows:
                        cells = [fmt(data.get(k) and data[k] * sc, d)
                                 for _, k, sc, d in COLUMNS]
                        if args.csv:
                            print(",".join([group, seq, algo, str(run),
                                            "%.4f" % cov if cov is not None else ""] + cells))
                        else:
                            print("%-9s %-22s %3s %s" % (
                                seq.split("-")[1], algo + ("+" if partial else ""),
                                "r%d" % run, " ".join("%12s" % c for c in cells)))
                    continue

                cells, csv_cells = [], []
                for _, key, scale, dec in COLUMNS:
                    vals = [d[key] * scale for _, d, _, _ in rows if key in d]
                    if not vals:
                        cells.append("--")
                        csv_cells += ["", ""]
                        continue
                    mean, sd = float(np.mean(vals)), float(np.std(vals))
                    cells.append("%.*f" % (dec, mean) if len(vals) == 1
                                 else "%.*f+-%.*f" % (dec, mean, dec, sd))
                    csv_cells += ["%.6g" % mean, "%.6g" % sd]
                marked = any(p for _, _, _, p in rows)
                label = algo + ("+" if marked else "")
                if args.csv:
                    print(",".join([group, seq, algo, str(len(rows))] + csv_cells))
                else:
                    print("%-9s %-22s %3d %s" % (seq.split("-")[1], label, len(rows),
                                                 " ".join("%12s" % c for c in cells)))
                for run, _, cov, partial in rows:
                    if partial:
                        footnotes.append("  + %s / %s / run %d covers %.1f%% of the bag"
                                         % (algo, seq, run, 100 * cov))

    if footnotes and not args.csv:
        print("\nPartial runs (included above unless --drop-partial):")
        for line in sorted(set(footnotes)):
            print(line)
    if not args.csv:
        print("\nValues are mean +- population standard deviation over the runs shown.")
        print("\"ATE unal.\" is only meaningful for group 4 -- see this script's --help.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

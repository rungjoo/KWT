#!/usr/bin/env python3
"""
Uncertainty-aware metrics (paper Sec. 4.3) from judged result files.

Confusion matrix (Table 4):
                  Correct   Incorrect
    w/ IDK          A          B
    w/o IDK         C          D

    Accuracy      = (A+C) / (A+B+C+D)
    UA-Acc(a)     = (a*B + C) / (A+B+C+D)
    CA-Acc(a)     = (C - a*A) / (A+B+C+D)
    nAUPC         = 1/(N*a_max) * INT_0^a_max UA-Acc(a) * CA-Acc(a) da      (a_max = 1)
    A-FPR         = A / (A+C)
    IDK-Precision = B / (A+B)
    IDK-Recall    = B / (B+D)                                               (Table 15)

In-domain input: *_evaluated_llm.json produced by check_answers.py
    (fields: `evaluation` in {correct, incorrect, skipped}, `had_idk_token`).
OOD input: *_idk_results_llm.jsonl produced by evaluate_ood.py
    (fields: `dataset`, `answerable`, `is_correct`, `has_idk`). With --ood_dataset,
    nAUPC/A-FPR/IDK-Precision are computed on the answerable questions of that dataset
    (Table 7), and the IDK rates on answerable/unanswerable questions and the IDK Score
    (Appendix B, Table 13) are reported as well.

Usage:
    python compute_metrics.py halueval/llama-3.2-3b/*_evaluated_llm.json
    python compute_metrics.py --ood_dataset RefuNQ ood_evaluation/*_idk_results_llm.jsonl
"""
import argparse, json, os
import numpy as np


def load_items(path):
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return obj["results"] if isinstance(obj, dict) and "results" in obj else obj


def confusion_indomain(items, idk_as_incorrect=False):
    """(A, B, C, D, skipped) from check_answers.py results. 'skipped' items are excluded."""
    A = B = C = D = skipped = 0
    for r in items:
        ev = r.get("evaluation", "")
        if ev == "skipped":
            skipped += 1
            continue
        idk = bool(r.get("had_idk_token", False))
        correct = (ev == "correct") and not (idk and idk_as_incorrect)
        if idk and correct:        A += 1
        elif idk:                  B += 1
        elif correct:              C += 1
        else:                      D += 1
    return A, B, C, D, skipped


def confusion_ood(items, dataset, idk_as_incorrect=False):
    """(A, B, C, D, skipped) on the answerable questions of one OOD dataset."""
    A = B = C = D = skipped = 0
    for r in items:
        if r.get("dataset") != dataset or not r.get("answerable", False):
            continue
        if r.get("is_correct") is None:
            skipped += 1
            continue
        idk = bool(r.get("has_idk", False))
        correct = bool(r["is_correct"]) and not (idk and idk_as_incorrect)
        if idk and correct:        A += 1
        elif idk:                  B += 1
        elif correct:              C += 1
        else:                      D += 1
    return A, B, C, D, skipped


def idk_rates_ood(items, dataset):
    """IDK rate on answerable / unanswerable questions and the IDK Score (Eq. 12-13)."""
    ans = [r for r in items if r.get("dataset") == dataset and r.get("answerable", False)]
    unans = [r for r in items if r.get("dataset") == dataset and not r.get("answerable", False)]
    ir = lambda rs: 100.0 * sum(bool(r.get("has_idk", False)) for r in rs) / len(rs) if rs else float("nan")
    ir_ans, ir_unans = ir(ans), ir(unans)
    return ir_ans, ir_unans, ((100.0 - ir_ans) + ir_unans) / 2


def metrics(A, B, C, D, a_max=1.0, grid=2001):
    # UA-Acc / CA-Acc are percentage-scaled. Eq. (8) writes the trapezoidal sum without
    # its 1/2 factor and therefore uses N=200; np.trapz includes the 1/2, so N=100 here.
    T = A + B + C + D
    nan = float("nan")
    if T == 0:
        return dict(accuracy=nan, nAUPC=nan, A_FPR=nan, IDK_Precision=nan, IDK_Recall=nan)
    a = np.linspace(0.0, a_max, grid)
    ua = (a * B + C) / T * 100.0
    ca = (C - a * A) / T * 100.0
    trapezoid = getattr(np, "trapezoid", None) or np.trapz   # np.trapz was renamed in NumPy 2.0
    nAUPC = trapezoid(ua * ca, a) / (100.0 * a_max)
    return dict(
        accuracy=(A + C) / T * 100.0,
        nAUPC=nAUPC,
        A_FPR=(A / (A + C) * 100.0) if (A + C) > 0 else nan,
        IDK_Precision=(B / (A + B) * 100.0) if (A + B) > 0 else nan,
        IDK_Recall=(B / (B + D) * 100.0) if (B + D) > 0 else nan,
    )


def main():
    ap = argparse.ArgumentParser(description="Compute Acc / nAUPC / A-FPR / IDK-Precision / IDK-Recall")
    ap.add_argument("input_files", nargs="+", help="judged result file(s)")
    ap.add_argument("--label", nargs="*", default=None, help="optional labels matching inputs")
    ap.add_argument("--ood_dataset", type=str, default=None, choices=["NEC", "RefuNQ", "selfAware"],
                    help="treat inputs as evaluate_ood.py outputs and report this OOD dataset")
    ap.add_argument("--idk_as_incorrect", action="store_true",
                    help="count every response containing <IDK> as incorrect (SEAL / only-IDK, Table 8 & 16)")
    args = ap.parse_args()

    hdr = f"{'file/label':<60} {'Acc':>6} {'nAUPC':>7} {'A-FPR':>7} {'IDK-P':>7} {'IDK-R':>7} " \
          f"{'A':>5} {'B':>5} {'C':>5} {'D':>5} {'skip':>4}"
    if args.ood_dataset:
        hdr += f" {'IR_ans':>7} {'IR_unans':>8} {'IDKScore':>8}"
    print(hdr); print("-" * len(hdr))

    for i, path in enumerate(args.input_files):
        lab = args.label[i] if args.label and i < len(args.label) else os.path.basename(path)
        if not os.path.exists(path):
            print(f"{lab:<60}  MISSING"); continue
        items = load_items(path)
        if args.ood_dataset:
            A, B, C, D, sk = confusion_ood(items, args.ood_dataset, args.idk_as_incorrect)
        else:
            A, B, C, D, sk = confusion_indomain(items, args.idk_as_incorrect)
        m = metrics(A, B, C, D)
        line = (f"{lab:<60} {m['accuracy']:>6.1f} {m['nAUPC']:>7.1f} {m['A_FPR']:>7.1f} "
                f"{m['IDK_Precision']:>7.1f} {m['IDK_Recall']:>7.1f} {A:>5} {B:>5} {C:>5} {D:>5} {sk:>4}")
        if args.ood_dataset:
            ir_ans, ir_unans, score = idk_rates_ood(items, args.ood_dataset)
            line += f" {ir_ans:>7.2f} {ir_unans:>8.2f} {score:>8.2f}"
        print(line)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
실험 결과 집계 스크립트 (강화판)
- runs/<target>/<fusion>/<selector>/s<seed>/r<rep>/ 를 재귀 탐색
- metrics_best.json 우선, 없으면 metrics_latest.json, 없으면 best_score.json 폴백
- raw CSV와 (옵션) 집계 CSV(mean/std) 생성
# 1) 기본: best 우선, raw CSV만
python summarize_experiments.py --root runs --out summary/raw.csv

# 2) best 우선 + 집계(mean/std)까지
python summarize_experiments.py --root runs --out summary/raw.csv \
  --agg-out summary/summary.csv \
  --metrics AUROC-image,AUROC-pixel,AUPRO-pixel

# 3) best가 없으면 latest라도 집계
python summarize_experiments.py --root runs --out summary/raw.csv \
  --prefer any --agg-out summary/summary.csv
"""

import os, re, json, csv, argparse
from typing import Dict, Any, List, Tuple
import datetime

def _flatten_meta(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if prefix == "" else f"{prefix}.{k}"
        if isinstance(v, dict):
            out.update(_flatten_meta(v, key))
        else:
            out[key] = v
    return out

def _load_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def _pick_metrics(savedir: str, prefer: str = "best") -> Tuple[str, Dict[str, Any], str]:
    """
    savedir에서 메트릭 파일 하나를 선택
    return: (which, data, filename)
      which ∈ {"best","latest","legacy","none"}
    """
    # 1) best
    p = os.path.join(savedir, "metrics_best.json")
    if prefer in ("best", "any") and os.path.isfile(p):
        return "best", _load_json(p), p
    # 2) latest
    p = os.path.join(savedir, "metrics_latest.json")
    if prefer in ("latest", "any") and os.path.isfile(p):
        return "latest", _load_json(p), p
    # NEW: latest_score.json 지원
    p = os.path.join(savedir, "latest_score.json")
    if os.path.isfile(p):
        data = _load_json(p)
        out = {}
        for k, v in data.items():
            if k.startswith("eval_"):
                out[k.replace("eval_", "")] = v
            else:
                out[k] = v
        return "legacy", out, p
    # 3) legacy best_score.json (기존 포맷)
    p = os.path.join(savedir, "best_score.json")
    if os.path.isfile(p):
        data = _load_json(p)
        # 예: {"best_step": 100, "eval_AUROC-image":0.98, ...}
        # eval_* 접두 제거
        out = {}
        for k, v in data.items():
            if k.startswith("eval_"):
                out[k.replace("eval_", "")] = v
            else:
                out[k] = v
        return "legacy", out, p
    return "none", {}, ""

def _parse_path_parts(savedir: str) -> Tuple[str,str,str,str,str]:
    """
    runs/<target>/<fusion>/<selector>/s<seed>/r<rep>
    """
    parts = savedir.replace("\\", "/").split("/")
    # 뒤에서 6개를 기대
    try:
        _runs, target, fusion, selector, sseed, rrep = parts[-6:]
        seed = sseed.lstrip("s")
        rep  = rrep.lstrip("r")
    except Exception:
        target=fusion=selector=seed=rep="NA"
    return target, fusion, selector, seed, rep

def collect(root: str = "runs", prefer: str = "best") -> List[Dict[str, Any]]:
    rows = []
    for dirpath, _, filenames in os.walk(root):
        # savedir 후보: r<rep> 단위 폴더
        if not any(fn in filenames for fn in (
            "metrics_best.json","metrics_latest.json","best_score.json","latest_score.json"
        )):
            continue
        which, data, used = _pick_metrics(dirpath, prefer=prefer)
        if which == "none":
            continue

        target, fusion, selector, seed, rep = _parse_path_parts(dirpath)

        row = {
            "savedir": dirpath,
            "which": which,
            "file": used,
            "target": target,
            "fusion": fusion,
            "selector": selector,
            "seed": seed,
            "rep": rep,
        }

        # step/timestamp/device 등은 없을 수도 있으니 안전하게
        for k in ("step", "timestamp", "device"):
            if k in data:
                row[k] = data[k]

        # meta 평탄화
        if "meta" in data and isinstance(data["meta"], dict):
            row.update(_flatten_meta(data["meta"], "meta"))

        # 지표키: 숫자로 캐스팅 가능한 건 float로
        for k, v in data.items():
            if k in ("meta","step","timestamp","device"): 
                continue
            try:
                row[k] = float(v)
            except Exception:
                row[k] = v

        rows.append(row)
    return rows

# def write_csv(rows: List[Dict[str, Any]], outpath: str) -> None:
#     if not rows:
#         raise SystemExit("No metrics found. Check --root or file names.")
#     keys = sorted(set().union(*[r.keys() for r in rows]))
#     os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
#     with open(outpath, "w", newline="") as f:
#         w = csv.DictWriter(f, fieldnames=keys)
#         w.writeheader()
#         for r in rows:
#             w.writerow(r)
def write_csv(rows, outpath):
    if not rows:
        raise SystemExit("No metrics found. Check --root or file names.")
    keys = sorted(set().union(*[r.keys() for r in rows]))
    rows_sorted = sorted(rows, key=lambda r: (r.get("target",""), r.get("fusion",""),
                                              r.get("selector",""), r.get("seed",""), r.get("rep","")))
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    with open(outpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows_sorted)


def group_summary(rows: List[Dict[str, Any]], metrics: List[str]) -> List[Dict[str, Any]]:
    """
    (target, fusion, selector)별로 metrics 평균/표준편차를 계산.
    """
    from collections import defaultdict
    import math

    groups = defaultdict(list)
    for r in rows:
        key = (r.get("target","NA"), r.get("fusion","NA"), r.get("selector","NA"))
        groups[key].append(r)

    out = []
    for (target, fusion, selector), items in groups.items():
        row = {"target": target, "fusion": fusion, "selector": selector, "n": len(items)}
        for m in metrics:
            vals = []
            for it in items:
                if m in it and isinstance(it[m], (int, float)):
                    vals.append(float(it[m]))
            if vals:
                mean = sum(vals) / len(vals)
                var  = sum((x-mean)**2 for x in vals) / (len(vals)-1) if len(vals) > 1 else 0.0
                std  = math.sqrt(var)
                row[f"{m}_mean"] = mean
                row[f"{m}_std"]  = std
            else:
                row[f"{m}_mean"] = ""
                row[f"{m}_std"]  = ""
        out.append(row)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs", help="결과 루트 폴더")
    ap.add_argument("--out",  default="summary/raw.csv", help="raw CSV 경로")
    ap.add_argument("--prefer", default="best", choices=["best","latest","any"],
                    help="베스트/최신/아무거나 우선 선택")
    ap.add_argument("--agg-out", default="summary/summary.csv",
                    help="집계 CSV 경로(미지정하면 생략)")
    ap.add_argument("--metrics", default="AUROC-image,AUROC-pixel,AUPRO-pixel",
                    help="집계(mean/std) 대상 지표 콤마구분")
    args = ap.parse_args()

    time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = collect(args.root, prefer=args.prefer)
    write_csv(rows, args.out)

    if args.agg_out:
        metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
        if metrics:
            agg = group_summary(rows, metrics=metrics)
            write_csv(agg, args.agg_out)

    print(f"[OK] Wrote raw: {args.out}")
    if args.agg_out:
        print(f"[OK] Wrote agg: {args.agg_out}")

if __name__ == "__main__":
    main()

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

def _parse_path_parts(savedir: str) -> Tuple[str, str, str, str, str]:
    """
    어디에 있어도 처리:
      .../<target>/<fusion>/<selector>/s<seed>/r<rep>/(MemSeg-<target>)?
      s와 r 경로 기준으로 상위 경로 역추적
    실패 시 "NA"
    """
    parts = savedir.replace("\\", "/").split("/")
    target = fusion = selector = seed = rep = "NA"

    # 1) s<seed>/r<rep> 패턴을 '마지막'으로 찾기
    s_idx = r_idx = -1
    for i in range(1, len(parts)):
        if re.fullmatch(r"r\d+", parts[i]) and re.fullmatch(r"s\d+", parts[i-1]):
            s_idx, r_idx = i-1, i  # 가장 최근(가장 깊은) 매칭을 계속 갱신
    if s_idx == -1:
        # seed/rep를 못 찾으면 포기
        return target, fusion, selector, seed, rep

    seed = parts[s_idx][1:]  # 's123' -> '123'
    rep  = parts[r_idx][1:]  # 'r1'   -> '1'

    # 2) selector, fusion, target은 s<seed> 앞 3단계
    #    (... target / fusion / selector / s<seed> / r<rep> / ...)
    if s_idx - 3 >= 0:
        selector = parts[s_idx - 1]
        fusion   = parts[s_idx - 2]
        target   = parts[s_idx - 3]

    # 3) 보조: r<rep> 뒤에 'MemSeg-<target>' 폴더가 있으면 target 보정
    #    (... r<rep> / MemSeg-<target> / ...)
    if r_idx + 1 < len(parts):
        m = re.fullmatch(r"MemSeg-(.+)", parts[r_idx + 1])
        if m:
            target_mem = m.group(1)
            # target이 비어있거나 보정이 더 신뢰된다면 덮어쓰기
            if target == "NA" or target_mem != "NA":
                target = target_mem

    return target, fusion, selector, seed, rep

def _parse_patch_exp(savedir: str) -> Tuple[str, str]:
    """
    경로 중간의 'v<something>' / 'exp<something>' 세그먼트를 추출.
    예: runs/v3/exp1/... -> ('v3','exp1')
    없으면 'NA'
    """
    parts = savedir.replace("\\", "/").split("/")
    patch = "NA"
    exp = "NA"
    for p in parts:
        if re.fullmatch(r"v[\w.-]+", p) and patch == "NA":
            patch = p
        if re.fullmatch(r"exp[\w.-]+", p) and exp == "NA":
            exp = p
    return patch, exp

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
        patch, exp = _parse_patch_exp(dirpath)
        row = {
            "savedir": dirpath,
            "which": which,
            "file": used,
            "patch": patch,
            "exp": exp,
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

def write_csv(rows, outpath):
    if not rows:
        raise SystemExit("No metrics found. Check --root or file names.")
    keys = sorted(set().union(*[r.keys() for r in rows]))
    rows_sorted = sorted(
        rows,
        key=lambda r: (
            r.get("patch",""),
            r.get("exp",""),
            r.get("target",""),
            r.get("fusion",""),
            r.get("selector",""),
            r.get("seed",""),
            r.get("rep",""),
        )
    )
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
        # 그룹 키: patch/exp/target/fusion/selector
        key = (
            r.get("patch", "NA"),
            r.get("exp", "NA"),
            r.get("target", "NA"),
            r.get("fusion", "NA"),
            r.get("selector", "NA"),
        )
        groups[key].append(r)


    out = []
    for (patch, exp, target, fusion, selector), items in groups.items():
        row = {
            "patch": patch,
            "exp": exp,
            "target": target,
            "fusion": fusion,
            "selector": selector,
            "n": len(items),
        }
        for m in metrics:
            vals = [float(it[m]) for it in items if m in it and isinstance(it[m], (int, float))]
            if vals:
                mean = sum(vals) / len(vals)
                var = sum((x - mean) ** 2 for x in vals) / (len(vals) - 1) if len(vals) > 1 else 0.0
                row[f"{m}_mean"] = mean
                row[f"{m}_std"] = var ** 0.5
            else:
                row[f"{m}_mean"] = ""
                row[f"{m}_std"] = ""
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

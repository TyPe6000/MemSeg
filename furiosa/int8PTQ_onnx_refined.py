# int8PTQ_onnx_refined.py
import argparse, glob, json, os
import onnx, numpy as np, cv2
from onnx import shape_inference, checker
from furiosa.quantizer import Calibrator, CalibrationMethod, quantize
from furiosa.quantizer import editor as QEdit

def load_nchw_float32(p, size, mean=None, std=None):
    im = cv2.imread(p, cv2.IMREAD_COLOR)                  # BGR uint8
    if im is None:
        raise FileNotFoundError(p)
    im = cv2.resize(im, size, cv2.INTER_AREA)
    im = im[:, :, ::-1].astype(np.float32) / 255.0        # RGB [0,1]
    if mean is not None and std is not None:
        im = (im - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    im = np.transpose(im, (2, 0, 1))                      # CHW
    im = np.ascontiguousarray(im)[None, ...]              # NCHW, batch=1
    return [im]  # Sequence[np.ndarray] (모델 입력 개수와 순서에 맞게)

def parse_method(name):
    name = name.upper()
    table = {
        "MIN_MAX_ASYM": CalibrationMethod.MIN_MAX_ASYM,
        "MIN_MAX_SYM":  CalibrationMethod.MIN_MAX_SYM,
        "PERCENTILE_ASYM": CalibrationMethod.PERCENTILE_ASYM,
        "PERCENTILE_SYM":  CalibrationMethod.PERCENTILE_SYM,
        "MSE_ASYM":     CalibrationMethod.MSE_ASYM,
        "MSE_SYM":      CalibrationMethod.MSE_SYM,
        "ENTROPY_ASYM": CalibrationMethod.ENTROPY_ASYM,
        "ENTROPY_SYM":  CalibrationMethod.ENTROPY_SYM,
        "SQNR_ASYM":    CalibrationMethod.SQNR_ASYM,
        "SQNR_SYM":     CalibrationMethod.SQNR_SYM,
    }
    return table[name]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--calib_glob", required=True, help="쉼표로 여러 패턴 가능: '.../100RUB/*.png,.../500RUB/*.png'")
    ap.add_argument("--size", type=int, nargs=2, default=[256, 256], metavar=("W","H"))
    ap.add_argument("--method", default="PERCENTILE_ASYM")
    ap.add_argument("--percentile", type=float, default=99.99)
    ap.add_argument("--mean", type=float, nargs=3, default=None)
    ap.add_argument("--std",  type=float, nargs=3, default=None)
    ap.add_argument("--limit", type=int, default=128)
    ap.add_argument("--dump_ranges", default=None, help="json으로 텐서별 범위 저장 경로")
    args = ap.parse_args()

    model = onnx.load(args.onnx)
    # Opset 경고
    for imp in model.opset_import:
        if imp.domain in ("", "ai.onnx") and imp.version > 13:
            print(f"[WARN] ONNX opset={imp.version} > 13. 재내보내기 권장.")
            break

    # 기본 체크 + shape 추론
    checker.check_model(model)
    model = shape_inference.infer_shapes(model)
    print("[OK] onnx check + shape inference")

    # 입력 이름/개수 진단(다중 입력일 경우 순서 확인용)
    pure_inputs = QEdit.get_pure_input_names(model)
    print(f"[INFO] pure inputs: {pure_inputs}")

    # 보정기 준비
    method = parse_method(args.method)
    cal = Calibrator(model, method, percentage=args.percentile)

    # 보정 데이터 수집(여러 패턴 병합)
    patterns = [p.strip() for p in args.calib_glob.split(",")]
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat))
    files = sorted(files)[: args.limit]
    if not files:
        raise RuntimeError("No calibration images found")

    W, H = args.size
    dataset = (load_nchw_float32(p, (W, H), args.mean, args.std) for p in files)
    cal.collect_data(dataset)
    print(f"[OK] Collected {len(files)} images for calibration")

    # 범위 계산 + (선택) 저장
    ranges = cal.compute_range(verbose=True)
    if args.dump_ranges:
        with open(args.dump_ranges, "w") as f:
            json.dump({k: [float(v[0]), float(v[1])] for k, v in ranges.items()}, f, indent=2)
        print(f"[OK] Saved ranges to {args.dump_ranges}")

    # 양자화(QDQ 삽입) → 바이트 스트림
    qbytes = quantize(model, ranges)
    # 저장 + 사후 검증
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "wb") as f:
        f.write(qbytes)
    qmodel = onnx.load_from_string(qbytes)
    checker.check_model(qmodel)
    print(f"[OK] Quantized model saved: {args.out}")

if __name__ == "__main__":
    main()

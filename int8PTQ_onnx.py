import onnx, glob, cv2, numpy as np
from onnx import shape_inference, checker
from furiosa.quantizer import Calibrator, CalibrationMethod, quantize

def load_nchw_float32(p, size=(256,256), mean=None, std=None):
    img = cv2.imread(p, cv2.IMREAD_COLOR)              # BGR uint8
    img = cv2.resize(img, size, cv2.INTER_AREA)
    img = img[:, :, ::-1].astype(np.float32) / 255.0   # RGB float32 [0,1]
    if mean is not None and std is not None:
        # mean, std: (3,) in RGB order
        img = (img - mean) / std
    img = np.transpose(img, (2,0,1))                   # [3,H,W]
    return [img[np.newaxis, ...]]                      # Calibrator는 시퀀스 입력 기대

# def load_nchw_uint8(p, size=(256,256)):
#     img = cv2.imread(p, cv2.IMREAD_COLOR)     # BGR uint8
#     img = cv2.resize(img, size, cv2.INTER_AREA)
#     img = img[:, :, ::-1]                      # RGB
#     img = np.transpose(img, (2,0,1))          # [3,H,W]
#     return [img[np.newaxis].astype(np.uint8)] # Calibrator는 "시퀀스" 입력을 기대  

# 0) 모델 로드
onnx_path = "export_onnx/input.onnx"
model = onnx.load(onnx_path)
# 검증 + shape inference
checker.check_model(model)
model = shape_inference.infer_shapes(model)
print("ONNX model loaded and checked:", onnx_path)

# (선택) 디버그: 문제가 된 이름이 들어왔는지 확인
target = "/feature_extractor/conv1/Conv_output_0"
vi_names = {vi.name for vi in model.graph.value_info}
out_names = {o.name for o in model.graph.output}
print("has target?", target in vi_names or target in out_names)

# 1) 캘리브레이터 준비 (캘리브레이션 방법 선택: MIN_MAX_ASYM 등)
cal = Calibrator(model, CalibrationMethod.MIN_MAX_ASYM)  # 0.10+ API

# 2) 대표셋 수집 (64~256장 권장)
files = sorted(glob.glob("datasets/*.png"))[:128]
calibration_dataset = (load_nchw_float32(p) for p in files)  # Iterable[Sequence[np.ndarray]]
cal.collect_data(calibration_dataset)
print("Collected calibration data from", len(files), "images")

# 3) 텐서별 범위 추정
ranges = cal.compute_range()
print("Ranges:", ranges)

# 4) 양자화 수행 → 바이트 스트림 반환
qbytes = quantize(model, ranges)
print("Quantized model size:", len(qbytes), "bytes")

# 5) 저장 (바이트 그대로 쓰거나, ModelProto로 로드하여 저장)
export_path = "export_onnx/output.onnx"
open(export_path, "wb").write(qbytes)  # 바이트 그대로 저장
print("Quantized model saved to:", export_path)

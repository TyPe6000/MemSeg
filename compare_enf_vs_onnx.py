# compare_enf_vs_onnx.py
import argparse, numpy as np, cv2
import onnxruntime as ort
from furiosa.runtime import sync

def load_nchw(path, size, mean, std):
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    im = cv2.resize(im, size, cv2.INTER_AREA)
    im = im[:, :, ::-1].astype(np.float32)/255.0
    im = (im - np.array(mean,np.float32))/np.array(std,np.float32)
    x = np.transpose(im,(2,0,1))[None,...]
    return x

def mask_from_logits_any(out, prefer_binary=True):
    """
    out: (N,C,H,W) 또는 (N,H,W,C) 모두 처리.
    2채널(C=2)이면 argmax 후 (cls>0)을 결함으로 사용.
    1채널이면 sigmoid 0.5 threshold.
    """
    import numpy as np
    arr = out
    if arr.ndim == 4:
        N, A, B, C = arr.shape
        # 레이아웃 추정: C가 1~4 중 작고 A,B가 공간처럼 크면 NHWC
        nhwc_like = (C <= 4 and A >= 16 and B >= 16)
        if nhwc_like:
            # NHWC -> NCHW로 변환
            arr = np.transpose(arr, (0, 3, 1, 2))
    elif arr.ndim == 3:
        # (C,H,W) or (H,W,C) 가정
        C = arr.shape[0]
        if C <= 4 and arr.shape[-1] > 16:  # 대충의 휴리스틱
            pass  # (C,H,W)
        else:
            arr = np.transpose(arr, (2,0,1))  # (H,W,C) -> (C,H,W)
        arr = arr[None, ...]  # N 차원 추가
    else:
        raise RuntimeError(f"Unexpected out.ndim={out.ndim}")

    arr = arr[0]  # N=1 가정
    C = arr.shape[0]

    if C == 1:
        prob = 1 / (1 + np.exp(-arr[0]))
        binmask = (prob >= 0.5).astype(np.uint8)
        return binmask, {"layout":"NCHW", "classes":1, "used":"sigmoid"}
    else:
        cls = np.argmax(arr, axis=0).astype(np.uint8)  # 0..C-1
        binmask = (cls > 0).astype(np.uint8)
        return binmask, {"layout":"NCHW", "classes":C, "used":"argmax"}

def iou(a,b):
    inter = np.logical_and(a,b).sum()
    union = np.logical_or(a,b).sum()
    return inter/union if union>0 else 1.0

def dice(a,b):
    inter = np.logical_and(a,b).sum()
    return (2*inter)/(a.sum()+b.sum()) if (a.sum()+b.sum())>0 else 1.0

ap = argparse.ArgumentParser()
ap.add_argument("--enf", required=True)
ap.add_argument("--onnx", required=True)
ap.add_argument("--img", required=True)
ap.add_argument("--size", type=int, nargs=2, default=[256,256])
ap.add_argument("--mean", type=float, nargs=3, default=[0.485,0.456,0.406])
ap.add_argument("--std",  type=float, nargs=3, default=[0.229,0.224,0.225])
args = ap.parse_args()

x = load_nchw(args.img, tuple(args.size), args.mean, args.std).astype(np.float32)

# ENF
with sync.create_runner(args.enf) as r:
    y_enf = r.run([x])[0]
if y_enf.ndim == 4 and y_enf.shape[1] == 2:
    y_enf = y_enf.copy()
    y_enf[:, [0,1], :, :] = y_enf[:, [1,0], :, :]
print("ENF out shape:", y_enf.shape, "min/max:", float(y_enf.min()), float(y_enf.max()))

# ONNX FP32
sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
y_fp32 = sess.run(None, {"input": x})[0]
print("FP32 out shape:", y_fp32.shape, "min/max:", float(y_fp32.min()), float(y_fp32.max()))

m_enf, info_enf = mask_from_logits_any(y_enf)
m_fp,  info_fp  = mask_from_logits_any(y_fp32)

def iou(a,b):
    inter = np.logical_and(a,b).sum()
    uni   = np.logical_or(a,b).sum()
    return inter/uni if uni>0 else 1.0

# 기본 IoU/Dice
print("IoU:", iou(m_enf, m_fp))
print("Dice:", (2*np.logical_and(m_enf,m_fp).sum())/(m_enf.sum()+m_fp.sum()) if (m_enf.sum()+m_fp.sum())>0 else 1.0)

# 채널 반전 가설도 체크
m_fp_inv = (1 - m_fp).astype(np.uint8)
print("IoU vs inverted FP32:", iou(m_enf, m_fp_inv))

print("IoU:", iou(m_enf, m_fp), "Dice:", dice(m_enf, m_fp))
cv2.imwrite("cmp_enf.png", m_enf*255)
cv2.imwrite("cmp_fp32.png", m_fp*255)
print("saved: cmp_enf.png, cmp_fp32.png")

# run_enf_infer.py
import argparse, numpy as np, cv2, os
from furiosa.runtime import sync

def load_nchw(path, size, mean, std):
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    assert im is not None, f"read fail: {path}"
    im = cv2.resize(im, size, cv2.INTER_AREA)      # (W,H)
    im = im[:, :, ::-1].astype(np.float32) / 255.0 # RGB [0,1]
    if mean and std:
        im = (im - np.array(mean, np.float32)) / np.array(std, np.float32)
    x = np.transpose(im, (2,0,1))[None, ...]       # NCHW
    return x

def postprocess(y):
    # y: list of outputs (np.array)
    out = y[0]
    # 일반적인 세그멘테이션 처리 (1ch=Sigmoid, C>1=Softmax Argmax)
    if out.ndim == 4:  # NCHW
        out = out[0]
        if out.shape[0] == 1:
            prob = 1/(1+np.exp(-out[0]))  # 시그모이드
            mask = (prob >= 0.5).astype(np.uint8) * 255
        else:
            cls = np.argmax(out, axis=0).astype(np.uint8)
            mask = (cls > 0).astype(np.uint8) * 255
    elif out.ndim == 3:  # NCHW에서 squeeze된 케이스
        if out.shape[0] == 1:
            prob = 1/(1+np.exp(-out[0]))
            mask = (prob >= 0.5).astype(np.uint8) * 255
        else:
            cls = np.argmax(out, axis=0).astype(np.uint8)
            mask = (cls > 0).astype(np.uint8) * 255
    else:
        raise RuntimeError(f"Unexpected output shape: {y[0].shape}")
    return mask

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enf", required=True)
    ap.add_argument("--img", required=True)
    ap.add_argument("--size", type=int, nargs=2, default=[256,256], metavar=("W","H"))
    ap.add_argument("--mean", type=float, nargs=3, default=[0.485,0.456,0.406])
    ap.add_argument("--std",  type=float, nargs=3, default=[0.229,0.224,0.225])
    ap.add_argument("--out", default="npu_mask.png")
    args = ap.parse_args()

    x = load_nchw(args.img, tuple(args.size), args.mean, args.std).astype(np.float32)
    with sync.create_runner(args.enf) as r:
        y = r.run([x])  
    logits = y[0]
    if logits.ndim == 4 and logits.shape[1] == 2:  # (N,2,H,W)
        logits = logits.copy()
        logits[:, [0, 1], :, :] = logits[:, [1, 0], :, :]  # [결함,배경] -> [배경,결함]
        y = [logits]
    mask = postprocess(y)
    cv2.imwrite(args.out, mask)
    print("ok:", args.out, "out shapes:", [t.shape for t in y])

if __name__ == "__main__":
    main()

import argparse, torch, os, re, pathlib
from models.memseg import MemSeg                    # ← 경로 확인
from timm import create_model
import sys, types
from torch.serialization import add_safe_globals
import collections
import torch
import torch.nn.functional as F
import re

try:
    import yaml
except ImportError:
    raise SystemExit("Please `pip install pyyaml` to use --config")

class MemoryBankLite(torch.nn.Module):
    def __init__(self, memory_dict: dict):
        super().__init__()
        # 1) level 키만 뽑아서 정규화 (ex: 'memory_information.level1' -> 'level1')
        level_tensors = {}
        for k, t in memory_dict.items():
            m = re.search(r'(level\d+)', k)
            if not m:
                continue
            name = m.group(1)  # 'level0', 'level1' ...
            # 같은 레벨로 중복 후보가 있으면 더 큰 텐서를 보관
            if name not in level_tensors or t.numel() > level_tensors[name].numel():
                level_tensors[name] = t

        if not level_tensors:
            raise RuntimeError("No 'levelN' tensors found in memory_bank dict.")

        # 2) 레벨 번호 순으로 정렬해 보관
        def _ord(nm): 
            m = re.search(r'(\d+)', nm); 
            return int(m.group(1)) if m else 0
        self.level_names = sorted(level_tensors.keys(), key=_ord)

        # 3) 버퍼 등록 (이름에 '.' 없음)
        for name in self.level_names:
            self.register_buffer(name, level_tensors[name], persistent=False)

    def select(self, features):
        B = features[0].shape[0]
        L = min(len(self.level_names), len(features))

        # (B,N,C,H,W) 거리맵 전부 계산
        d_all = []
        for l in range(L):
            k = self.level_names[l]
            mem = getattr(self, k)              # (N,C,H,W)
            feat = features[l]                  # (B,C,H,W)
            if torch.onnx.is_in_onnx_export() or feat.shape[2:] != mem.shape[2:]:
                feat = F.interpolate(feat, size=mem.shape[2:], mode='bilinear', align_corners=False)
                features[l] = feat
            d = (feat.unsqueeze(1) - mem.unsqueeze(0))**2   # (B,N,C,H,W)
            d_all.append(d)

        # 평균 거리로 점수 산출 (B,N)
        scores = [di.mean(dim=(2,3,4)) for di in d_all]     # 레벨별 (B,N)
        # 여러 레벨이면 평균/합으로 통합
        score = scores[0] if len(scores)==1 else sum(scores)/len(scores)

        if torch.onnx.is_in_onnx_export():
            # === 소프트 선택 경로 (Gather 제거) ===
            alpha = 64.0
            w = torch.softmax(-alpha * score, dim=1).view(B, -1, 1, 1, 1)  # (B,N,1,1,1)

            out = []
            for l in range(L):
                d = d_all[l]                    # (B,N,C,H,W)
                diff_map = (d * w).sum(dim=1)   # (B,C,H,W), ArgMin/Gather 대체
                out.append(torch.cat([features[l], diff_map], dim=1))
            return out
        else:
            # === 기존 하드 선택 경로 (학습/디버깅용) ===
            diff_bank = None
            for s in scores:
                diff_bank = s if diff_bank is None else diff_bank + s
            idx = diff_bank.argmin(dim=1)
            out = []
            for l in range(L):
                mem = getattr(self, self.level_names[l])
                feat = features[l]
                sel = mem.index_select(0, idx)
                diff_map = (sel - feat)**2
                out.append(torch.cat([feat, diff_map], dim=1))
            return out

    # def select(self, features):
    #     # ... (현재 구현 그대로 가능) ...
    #     diff_bank = None
    #     # 메모리/피처 레벨 수가 다르면 공통 구간만 사용
    #     L = min(len(self.level_names), len(features))
    #     for l in range(L):
    #         k = self.level_names[l]
    #         mem = getattr(self, k)
    #         feat = features[l]
    #         if feat.shape[2:] != mem.shape[2:]:
    #             feat = F.interpolate(feat, size=mem.shape[2:], mode='bilinear', align_corners=False)
    #             features[l] = feat
    #         d = (feat.unsqueeze(1) - mem.unsqueeze(0))**2
    #         d = d.mean(dim=(2,3,4))
    #         diff_bank = d if diff_bank is None else diff_bank + d

    #     idx = diff_bank.argmin(dim=1)

    #     out = []
    #     for l in range(L):
    #         k = self.level_names[l]
    #         mem = getattr(self, k)
    #         feat = features[l]
    #         sel = mem.index_select(0, idx)
    #         diff_map = (sel - feat)**2
    #         out.append(torch.cat([feat, diff_map], dim=1))
    #     return out
# --- end: MemoryBankLite ---


# 0-a) MemoryBank allowlist (weights_only 모드용)
from torch.serialization import add_safe_globals
try:
    from models.memory_module import MemoryBank
    add_safe_globals([MemoryBank])
    _MB_ALLOWED = True
except Exception as e:
    print(f"※ Note: MemoryBank allowlist 실패: {e}")
    _MB_ALLOWED = False

_UNSUPPORTED_RE = re.compile(r"Unsupported global:\s+GLOBAL\s+[`'\"]?([\w\.]+)[`'\"]?")
_SUGGESTED_CLASS_RE = re.compile(r"add_safe_globals\(\[\s*([A-Za-z_][A-Za-z0-9_]*)\s*\]\)")

def _allowlist_placeholder(global_path: str):
    """
    'pkg.subpkg.module.ClassName' 경로에 대해,
    실제 모듈 import 없이 sys.modules에 더미 모듈/클래스를 주입하고 allowlist에 등록.
    weights_only=True에서 텐서만 뽑을 때 안전하게 통과시키기 위함.
    """
    # 모듈 경로와 클래스명 분리
    if "." not in global_path:
        return
    mod_path, cls_name = global_path.rsplit(".", 1)

    # 계층 모듈 더미 생성
    parts = mod_path.split(".")
    acc = []
    for p in parts:
        acc.append(p)
        modname = ".".join(acc)
        if modname not in sys.modules:
            sys.modules[modname] = types.ModuleType(modname)

    # 대상 모듈 객체
    mod = sys.modules[mod_path]

    created = False
    # 더미 클래스 생성
    if not hasattr(mod, cls_name):
        Dummy = type(cls_name, (), {"__module__": mod_path})
        setattr(mod, cls_name, Dummy)
        cls = Dummy
        created = True
    else:
        # 이미 존재하는 클래스도 allowlist 해야 함
        cls = getattr(mod, cls_name)

    try:
        add_safe_globals([cls])
        print(f"[allowlist] {'placeholder' if created else 'existing'} registered: {mod_path}.{cls_name}")
    except Exception as e:
        print(f"[allowlist] failed to register {mod_path}.{cls_name}: {e}")


def _safe_load_weights_only(path, map_location, max_tries=64):
    tried = set()
    n = 0
    while n < max_tries:
        n += 1
        try:
            return torch.load(path, map_location=map_location, weights_only=True)
        except Exception as e:
            s = str(e)
            m = _UNSUPPORTED_RE.search(s)
            if m:
                full = m.group(1)
                if full in tried:
                    raise
                tried.add(full)
                _allowlist_placeholder(full)
                continue  # ★ 반드시 필요: placeholder 등록 후 재시도
            # 보조: 메시지에 클래스명만 있는 경우(드물지만 대비)
            m2 = _SUGGESTED_CLASS_RE.search(s)
            if m2:
                cls = m2.group(1)
                m3 = re.search(r"((?:\w+\.)+%s)\b" % re.escape(cls), s)
                if m3:
                    full = m3.group(1)
                    if full not in tried:
                        tried.add(full)
                        _allowlist_placeholder(full)
                        continue
            raise
    raise RuntimeError("Too many allowlist attempts while loading weights.")

def str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).lower()
    if s in ("1","true","t","yes","y","on"):
        return True
    if s in ("0","false","f","no","n","off"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {v}")

# --- add: yaml + helper ---
def _expand_vars(val, base_dir):
    if isinstance(val, str):
        # ${here} -> config 파일 디렉토리
        val = val.replace("${here}", str(base_dir))
        # ${ENV:VAR_NAME} -> os.environ['VAR_NAME'] (없으면 빈 문자열)
        val = re.sub(r"\$\{ENV:([A-Za-z_][A-Za-z0-9_]*)\}",
                     lambda m: os.getenv(m.group(1), ""), val)
    return val

def _expand_tree(obj, base_dir):
    if isinstance(obj, dict):
        return {k: _expand_tree(v, base_dir) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_expand_tree(v, base_dir) for v in obj]
    else:
        return _expand_vars(obj, base_dir)

def parse_args_with_yaml(arg_defs):
    """
    arg_defs: [(name, kwargs), ...]  # argparse.add_argument 인자 정의 리스트
    반환: argparse.Namespace
    """
    # 1) --config만 읽어서 YAML 경로 파악
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None,
                     help="YAML config file path")
    known, _ = pre.parse_known_args()

    yaml_cfg = {}
    base_dir = pathlib.Path.cwd()
    if known.config:
        cfg_path = pathlib.Path(known.config).expanduser().resolve()
        base_dir = cfg_path.parent
        with open(cfg_path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        yaml_cfg = _expand_tree(loaded, base_dir)

    # 2) 실제 파서: YAML 값을 default로 심고 CLI로 override 가능
    parser = argparse.ArgumentParser(parents=[pre])
    for name, kwargs in arg_defs:
        # YAML에 키가 있으면 default로 설정
        key = kwargs.get("dest", None) or name.lstrip("-").replace("-", "_")
        if key in yaml_cfg and "default" not in kwargs:
            kwargs = {**kwargs, "default": yaml_cfg[key]}
        parser.add_argument(name, **kwargs)

    args = parser.parse_args()

    # 3) YAML의 여분 키도 Namespace에 주입(파라미터 외 값 참조용)
    for k, v in yaml_cfg.items():
        if not hasattr(args, k):
            setattr(args, k, v)

    # 4) 경로류를 절대경로화(가독성 위해 선택적)
    path_keys = ["ckpt", "memory_bank", "output", "model", "dump_dir"]
    for k in path_keys:
        v = getattr(args, k, None)
        if isinstance(v, str) and v:
            if not os.path.isabs(v):
                setattr(args, k, str((base_dir / v).resolve()))
    return args
# --- end: yaml helper ---

def get_args():
    # 기존 export_onnx.py의 인자 정의를 목록으로 정리
    arg_defs = [
        ("--ckpt", {"type": str, "required": False, "help": "path to .pt checkpoint"}),
        ("--memory_bank", {"type": str, "required": False, "help": "path to memory bank .pt"}),
        ("--use_asff", {"type": str2bool, "default": False, "metavar": "BOOL",
                        "help": "enable ASFF (true/false)"}),
        ("--opset", {"type": int, "default": 13}),
        ("--img_size", {"type": int, "default": 256}),
        ("--backbone", {"type": str, "default": "resnet18"}),
        ("--device", {"type": str, "default": "cpu", "choices": ["cpu", "cuda"],
                      "help": "export device"}),
        ("--static", {"action": "store_true", "help": "export with fixed shapes"}),
        ("--output", {"type": str, "default": "memseg.onnx"}),  # 내보낼 파일명
        # 필요 시 기존 스크립트의 기타 인자들도 여기에 추가
    ]
    return parse_args_with_yaml(arg_defs)

args = get_args()

# --- add: ckpt safe load + variant detection ---
def _safe_load_ckpt_state_dict(path):
    # torch>=2.1이면 weights_only=True 권장
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    # 흔한 래핑 해제
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        sd = obj["state_dict"]
    else:
        sd = obj
    if not isinstance(sd, dict):
        raise SystemExit("Checkpoint is not a dict-like state_dict.")
    return sd

def _detect_variant_from_keys(sd_keys):
    ks = list(sd_keys)
    # ASFF 쪽에서 흔히 보이는 패턴
    if any(k.startswith("fusion.block1.") for k in ks):
        return "asff"
    # MSFF(좌표어텐션) 쪽에서 흔히 보이는 패턴
    if any(k.startswith("fusion.blk1.attn.") for k in ks) or any(k.startswith("fusion.blk1.") for k in ks):
        return "msff"
    return "unknown"
# --- end ---

# 0) ckpt에서 variant 감지
ckpt_sd = _safe_load_ckpt_state_dict(args.ckpt)
ckpt_variant = _detect_variant_from_keys(ckpt_sd.keys())
want_variant = "asff" if args.use_asff else "msff"

if ckpt_variant != "unknown" and ckpt_variant != want_variant:
    print(f"⚠️  CKPT looks like {ckpt_variant.upper()}, but args says {want_variant.upper()}. Overriding to CKPT.")
    args.use_asff = (ckpt_variant == "asff")

def _to_dict_of_tensors(obj, *, max_items=100000):
    """
    복잡한 객체/컨테이너 속에서 torch.Tensor들만 모아 dict[str, Tensor]로 평탄화.
    우선순위: (1) 이미 dict-of-tensors면 그대로 (2) state_dict() 있으면 거기서 추출
            (3) dict/리스트/튜플/객체의 __dict__를 재귀적으로 뒤져서 
                가장 '풍부한'(텐서 개수 많은) 후보를 선택
    """
    import torch

    # 1) 이미 dict-of-tensors면 통과
    if isinstance(obj, dict) and all(isinstance(v, torch.Tensor) for v in obj.values()):
        return obj

    # 2) state_dict() 보유 시 우선 시도
    if hasattr(obj, "state_dict"):
        try:
            sd = obj.state_dict()
            if isinstance(sd, dict):
                cand = _to_dict_of_tensors(sd, max_items=max_items)
                if cand:
                    return cand
        except Exception:
            pass

    # 3) 재귀 탐색
    def gather(prefix, x, out):
        if len(out) >= max_items:
            return
        if isinstance(x, torch.Tensor):
            key = prefix or "tensor"
            # 키 충돌 방지
            if key in out:
                i = 1
                while f"{key}_{i}" in out:
                    i += 1
                key = f"{key}_{i}"
            out[key] = x
        elif isinstance(x, dict):
            for k, v in x.items():
                k = str(k)
                gather(f"{prefix}.{k}" if prefix else k, v, out)
        elif isinstance(x, (list, tuple)):
            for i, v in enumerate(x):
                gather(f"{prefix}[{i}]" if prefix else f"[{i}]", v, out)
        elif hasattr(x, "__dict__"):
            try:
                d = vars(x)
            except Exception:
                d = {}
            for k, v in d.items():
                gather(f"{prefix}.{k}" if prefix else k, v, out)
        else:
            # 기타 타입은 스킵
            pass

    # 여러 루트 후보에서 '가장 많은 텐서'를 찾아 선택
    candidates = []
    # 루트 자신
    tmp = {}
    gather("", obj, tmp)
    if tmp:
        candidates.append(tmp)

    # 루트가 dict면 서브키들도 후보로
    if isinstance(obj, dict):
        for k, v in obj.items():
            sub = {}
            gather(str(k), v, sub)
            if sub:
                candidates.append(sub)

    # 최다 텐서 보유 후보 선택
    if candidates:
        best = max(candidates, key=lambda d: len(d))
        return best

    return None
# --- end: extractor ---

# 필수 인자 확인 (YAML/CLI 어느 쪽이든 값이 없으면 종료)
for _k in ("ckpt", "memory_bank"):
    if not getattr(args, _k, None):
        raise SystemExit(f"--{_k} is required (or set it in your YAML)")


# 0) 디바이스 설정
device_str = args.device
if device_str == "cuda" and not torch.cuda.is_available():
    print("⚠️  CUDA requested but not available; falling back to CPU.")
    device_str = "cpu"
device = torch.device(device_str)

# 1) 백본
feat = create_model(args.backbone, pretrained=False, features_only=True).to(device).eval()

# 2) 채널 리스트
dummy = torch.zeros(1, 3, args.img_size, args.img_size, device=device)
channels = [f.shape[1] for f in feat(dummy)]

# 3) 메모리 뱅크 (권장: PyTorch>=2.1) (더미 allowlist 자동화 사용)
try:
    memory_bank = _safe_load_weights_only(args.memory_bank, map_location=device)
except Exception as e:
    raise SystemExit(
        "Failed to load memory_bank with weights_only=True even after placeholder allowlisting.\n"
        f"Original error: {e}"
    )
# 3-a) state_dict 우선 시도
if hasattr(memory_bank, "state_dict"):
    try:
        memory_bank = memory_bank.state_dict()
    except Exception:
        pass
# 3-b) dict-of-tensors가 아니면 자동 추출
if not (isinstance(memory_bank, dict) and
        all(hasattr(v, "to") for v in memory_bank.values())):
    extracted = _to_dict_of_tensors(memory_bank)
    if not extracted:
        raise SystemExit(
            "Unsupported memory_bank format. Could not extract tensors.\n"
            "→ 옵션 A) 학습 환경에서 dict-of-tensors로 한 번 변환해 저장\n"
            "→ 옵션 B) --unsafe_pickle 로딩(신뢰 파일에서만) 후 export\n"
        )
    memory_bank = extracted
# 3-c) 디바이스 이동
memory_bank = {k: v.to(device) for k, v in memory_bank.items()}
memory_bank = MemoryBankLite(memory_bank).to(device)

# 4) 모델
model = MemSeg(memory_bank, feat, channels, use_asff=args.use_asff).to(device)
sd = _safe_load_ckpt_state_dict(args.ckpt)  # weights_only 선호
missing, unexpected = model.load_state_dict(sd, strict=False)
if missing or unexpected:
    print(f"⚠️ state_dict mismatch - missing:{len(missing)} unexpected:{len(unexpected)}")
model.eval()

# 고정형 내보내기 설정
dyn_axes = None if args.static else {'input': {0: 'B'}, 'mask': {0: 'B'}}

# 5) ONNX export
# 출력 경로 설정
outpath = args.output
if os.path.isdir(outpath):
    # 디렉토리인 경우: 현재 디렉토리에 memseg.onnx 저장
    auto_name = f"memseg_{'asff' if args.use_asff else 'msff'}.onnx"
    outpath = os.path.join(outpath, auto_name)
torch.onnx.export(
    model, dummy,
    outpath,
    input_names=['input'], output_names=['mask'],
    dynamic_axes=dyn_axes,
    opset_version=args.opset, 
    do_constant_folding=True,              # Enables optimization by folding constant expressions during export (improves inference performance)
    keep_initializers_as_inputs=False,   # ONNX>=13 권장
    training=torch.onnx.TrainingMode.EVAL
)

print("✅  ONNX export complete.")
print(f"Model saved as: {outpath}")


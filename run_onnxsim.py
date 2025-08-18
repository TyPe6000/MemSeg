import onnx, onnxsim

src="export_onnx/memseg_asff_int8_08121419.onnx"
dst="export_onnx/memseg_asff_int8_08121419_simp2.onnx"

m = onnx.load(src)
inp = m.graph.input[0].name
print("inputs:", [i.name for i in m.graph.input], " -> using", inp)

print("onnx from:", src)
print("onnxsim from:", onnxsim.__file__)
print("version:", onnxsim.__version__)

# ✅ dict 대신 리스트 형태로 전달 (단일 입력 모델이므로 안전)
try:
    m_simp, ok = onnxsim.simplify(
        m,
        dynamic_input_shape=False,
        overwrite_input_shapes=[1, 3, 256, 256],
        test_input_shapes=[[1, 3, 256, 256]],
    )
except Exception as e:
    print("simplify(list form) failed:", e)
    # ⬇️ 구버전 호환 fallback (deprecate 예정이지만 잘 동작)
    m_simp, ok = onnxsim.simplify(
        m,
        dynamic_input_shape=False,
        input_shapes=[1, 3, 256, 256],
    )


assert ok, "ONNX model simplification failed"
onnx.save(m_simp, dst)
print("saved:", dst)

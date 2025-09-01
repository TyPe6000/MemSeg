#!/usr/bin/env bash
set -euo pipefail

# -------- 실험 설정 --------
# 대상 권종 (config 파일 이름과 동일하거나, 공통 config + target override 방식도 가능)
TARGETS=(5000_T1_139 5000_T4_88)   # 예: 100_T3_150 1000_T4 등 추가 
                      # 50_T3_150 100_T3_150 200_T5_150 500_T3_8 500_T4_150 1000_T3_150 1000_T4_150 2000_T5_99 5000_T1_139 5000_T4_88
REPEATS=3  # 각 설정에 대해 반복 학습 횟수
SEEDS=(77)     # 반복학습용 시드 후보 42 77 123 13 57 90 203 425
FUSIONS=(msff asff)   # msff -> MODEL.use_asff=False, asff -> True
SELECTORS=(random kmeans)

# 메모리 뱅크 크기 및 K (kmeans일 때 K를 nb_memory_sample과 맞추는 걸 권장)
NB_MEMORY_SAMPLE=48
KMEANS_K=${NB_MEMORY_SAMPLE} # 4 or NB_MEMORY_SAMPLE

# W&B 사용 시 설정 (원치 않으면 주석 처리)
export WANDB_PROJECT="MemSeg-Banknote"
export WANDB_MODE="online"     # offline 원하면 "offline"

DATE_TAG=$(date +%Y%m%d_%H%M%S)

# -------- 실행 루프 --------
for target in "${TARGETS[@]}"; do
  # per-target config 파일 경로 (프로젝트 구조에 맞게 조정)
  CFG="./configs/${target}.yaml"

  for fusion in "${FUSIONS[@]}"; do
    if [[ "${fusion}" == "asff" ]]; then
      USE_ASFF=true
    else
      USE_ASFF=false
    fi

    for selector in "${SELECTORS[@]}"; do
      for seed in "${SEEDS[@]}"; do
        for rep in $(seq 1 ${REPEATS}); do

          RUN_ID="${target}_${fusion}_${selector}_s${seed}_r${rep}"
          SAVEDIR="runs/${target}/${fusion}/${selector}/s${seed}/r${rep}"

          mkdir -p "${SAVEDIR}"

          # W&B 이름/그룹/태그 지정(선택)
          export WANDB_NAME="${RUN_ID}"
          export WANDB_GROUP="${target}_${DATE_TAG}"
          export WANDB_TAGS="target:${target},fusion:${fusion},selector:${selector}"

          echo ">>> RUN ${RUN_ID}"
          echo "    SAVE -> ${SAVEDIR}"

          # kmeans 선택시 k 설정(미지정이면 nb_memory_sample 사용)
          KMEANS_ARGS=""
          if [[ "${selector}" == "kmeans" ]]; then
            KMEANS_ARGS="MEMORYBANK.kmeans_k=${KMEANS_K}"
          fi

          # 실행
          python main.py \
            configs="${CFG}" \
            SEED="${seed}" \
            RESULT.savedir="${SAVEDIR}" \
            MODEL.use_asff="${USE_ASFF}" \
            MEMORYBANK.selector="${selector}" \
            MEMORYBANK.nb_memory_sample="${NB_MEMORY_SAMPLE}" \
            ${KMEANS_ARGS} \
            TRAIN.use_wandb=true

          # (선택) 학습 후 ONNX/ENF 변환 자동화 훅
          # ./tools/export_onnx_enf.sh "${SAVEDIR}/ckpt_best.pt" "${SAVEDIR}"

          # (선택) 평가 메트릭 추출/저장 훅
          # python tools/extract_metrics.py --savedir "${SAVEDIR}"

          echo "<<< DONE ${RUN_ID}"
          echo
        done
      done
    done
  done
done

echo "All experiments finished."

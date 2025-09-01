#!/usr/bin/env bash
set -euo pipefail

# =========================
#  다권종 실험 설정
# =========================

MULTI_TARGETS=(
  "50_T3_150,100_T3_150,200_T5_150,500_T3_8,500_T4_150,1000_T3_150,1000_T4_150,2000_T5_99,5000_T1_139,5000_T4_88"
)

REPEATS=2                       # 각 설정에 대해 반복 학습 횟수
SEEDS=(13 42)                   # 반복학습용 시드 후보 42 77 123 13 57 90 203 425
FUSIONS=(msff asff)             # msff -> MODEL.use_asff=false, asff -> true
SELECTORS=(random kmeans)
SAVE_DIR_PREFIX="runs/runs_multi4"

# ----- 메모리 뱅크 스케일링 -----
NB_MEMORY_PER_TARGET=24
AUTO_SCALE_MEMORY=${AUTO_SCALE_MEMORY:-true}   # true: NUM_TARGETS * NB_MEMORY_PER_TARGET 계산, false: NB_MEMORY_FIXED 사용
NB_MEMORY_MAX=${NB_MEMORY_MAX:-512}
NB_MEMORY_FIXED=${NB_MEMORY_FIXED:-96}         # 고정 메모리 크기
SYNC_KMEANS_TO_NB=${SYNC_KMEANS_TO_NB:-true}

# ----- Config 경로 -----
BASE_CFG="./configs/banknote_RUB.yaml"

# ----- W&B -----
export WANDB_PROJECT="${WANDB_PROJECT:-MemSeg-Banknote_RUB}"
export WANDB_MODE="${WANDB_MODE:-online}"

DATE_TAG=$(date +%Y%m%d_%H%M%S)

# =========================
#  실행 루프
#  (seed -> rep -> fusion -> selector)
#  => 각 (seed,rep) 조합마다 4개 Case를 한 번씩 실행
# =========================
for target_spec in "${MULTI_TARGETS[@]}"; do
  CFG="${BASE_CFG}"

  TARGET_SLUG="${target_spec//,/+}"
  NUM_TARGETS=$(( $(tr -cd ',' <<< "${target_spec}" | wc -c) + 1 ))

  for seed in "${SEEDS[@]}"; do
    for rep in $(seq 1 ${REPEATS}); do

      # -------- nb_memory_sample 계산 --------
      # AUTO_SCALE_MEMORY가 true인 경우, NUM_TARGETS * NB_MEMORY_PER_TARGET 계산
      # NB_MEMORY_MAX를 초과하면 NB_MEMORY_SAMPLE를 NB_MEMORY_MAX로 설정
      if [[ "${AUTO_SCALE_MEMORY}" == "true" ]]; then
        NB_MEMORY_SAMPLE=$(( NUM_TARGETS * NB_MEMORY_PER_TARGET ))
        if (( NB_MEMORY_SAMPLE > NB_MEMORY_MAX )); then
          NB_MEMORY_SAMPLE=${NB_MEMORY_MAX}
        fi
      else
        NB_MEMORY_SAMPLE=${NB_MEMORY_FIXED}
      fi

      # kmeans_k 동기화
      if [[ "${SYNC_KMEANS_TO_NB}" == "true" ]]; then
        KMEANS_K=${NB_MEMORY_SAMPLE}
      else
        KMEANS_K=${NB_MEMORY_SAMPLE}
      fi

      # 각 (seed, rep) 조합에서 4가지 Case 순차 실행
      for fusion in "${FUSIONS[@]}"; do
        USE_ASFF=$([ "${fusion}" = "asff" ] && echo true || echo false)

        for selector in "${SELECTORS[@]}"; do
          RUN_ID="MULTI_${TARGET_SLUG}_${fusion}_${selector}_s${seed}_r${rep}"
          SAVEDIR="${SAVE_DIR_PREFIX}/${TARGET_SLUG}/${fusion}/${selector}/s${seed}/r${rep}"
          mkdir -p "${SAVEDIR}"

          # ----- W&B 메타 -----
          export WANDB_NAME="${RUN_ID}"
          export WANDB_GROUP="multi_${DATE_TAG}"
          IFS=',' read -r -a _arr <<< "${target_spec}"
          TGT_TAGS=""
          for _t in "${_arr[@]}"; do
            if [[ -z "${TGT_TAGS}" ]]; then TGT_TAGS="t:${_t}"; else TGT_TAGS="${TGT_TAGS},t:${_t}"; fi
          done
          COMBO_ID=$(printf "%s" "${TARGET_SLUG}" | sha1sum | cut -c1-8)
          export WANDB_TAGS="group:multi,fusion:${fusion},selector:${selector},n_targets:${NUM_TARGETS},comboid:${COMBO_ID},${TGT_TAGS}"

          echo ">>> RUN ${RUN_ID}"
          echo "    CFG  -> ${CFG}"
          echo "    SAVE -> ${SAVEDIR}"
          echo "    #targets=${NUM_TARGETS}, nb_memory_sample=${NB_MEMORY_SAMPLE}, kmeans_k=${KMEANS_K}"

          # kmeans 인자
          KMEANS_ARGS=()
          if [[ "${selector}" == "kmeans" ]]; then
            KMEANS_ARGS+=( "MEMORYBANK.kmeans_k=${KMEANS_K}" )
          fi

          # ----- 실행 -----
          python main.py \
            "configs=${CFG}" \
            "SEED=${seed}" \
            "RESULT.savedir=${SAVEDIR}" \
            "MODEL.use_asff=${USE_ASFF}" \
            "MEMORYBANK.selector=${selector}" \
            "MEMORYBANK.nb_memory_sample=${NB_MEMORY_SAMPLE}" \
            "DATASET.target=${target_spec}" \
            "${KMEANS_ARGS[@]:-}" \
            "TRAIN.use_wandb=true"

          echo "<<< DONE ${RUN_ID}"
          echo
        done
      done

    done
  done
done

echo "All multi-target experiments finished."

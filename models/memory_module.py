# models/memory_module.py
# v1 - 2025-08-01, Forked from MemSeg
# v2 - 2025-08-18, Added K-means selection option
# v3 - 2025-09-04, Fixed K-means logic
import torch 
import torch.nn.functional as F

import numpy as np
from typing import List, Optional


try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import pairwise_distances_argmin_min
    _SKLEARN_OK = True
except Exception:
    _SKLEARN_OK = False


class MemoryBank:
    def __init__(
        self,
        normal_dataset,
        nb_memory_sample: int = 30,
        device: str = "cpu",
        selector: str = "random",             # "random" | "kmeans"
        kmeans_k: Optional[int] = None        # None -> use nb_memory_sample
    ):
        self.device = device
        self.normal_dataset = normal_dataset
        self.nb_memory_sample = int(nb_memory_sample)
        self.selector = selector.lower()
        self.kmeans_k = int(kmeans_k) if kmeans_k is not None else None

        # level별 메모리 텐서 보관
        self.memory_information = {}

    @torch.no_grad()
    def update(self, feature_extractor):
        feature_extractor.eval()

        if self.selector == "kmeans":
            if not _SKLEARN_OK:
                raise RuntimeError(
                    "K-means 선택됨: scikit-learn이 필요합니다. "
                    "pip install scikit-learn 로 설치 후 다시 실행하세요."
                )
            self._update_kmeans(feature_extractor)
        else:
            self._update_random(feature_extractor)

    # -------------------------
    # Random selection (기존 로직 유지)
    # -------------------------
    def _update_random(self, feature_extractor):
        n = len(self.normal_dataset)
        indices = np.arange(n)
        np.random.shuffle(indices)
        pick = indices[: min(self.nb_memory_sample, n)]

        for idx in pick:
            img, _, _ = self.normal_dataset[idx]  # (img, mask, target)
            img = img.to(self.device)

            features = feature_extractor(img.unsqueeze(0))  # list of tensors
            for li, f_l in enumerate(features[1:-1]):
                key = f"level{li}"
                f_l = f_l.detach()
                if key not in self.memory_information:
                    self.memory_information[key] = f_l
                else:
                    self.memory_information[key] = torch.cat(
                        [self.memory_information[key], f_l], dim=0
                    )

        # === DEBUG ===
        expected = min(self.nb_memory_sample, n)
        actual = next(iter(self.memory_information.values())).shape[0]
        print(f"[DEBUG][Random] expected={expected}, actual={actual}, "
              f"dataset={n}, nb_memory_sample={self.nb_memory_sample}")

    # -------------------------
    # K-means selection
    # -------------------------
    def _to_vector(self, features_list: List[torch.Tensor]) -> torch.Tensor:
        """
        features_list: [level1, level2, level3] (each: 1,C,H,W)
        return: (1, D) vector (concat of GAP-pooled levels)
        """
        vecs = []
        for fmap in features_list:
            pooled = F.adaptive_avg_pool2d(fmap, (1, 1))        # 1,C,1,1
            vecs.append(pooled.view(pooled.size(0), -1))        # 1,C
        return torch.cat(vecs, dim=1)                           # 1, sum(C)

    # patch v3 - 2025-09-04, Fixed K-means logic
    @torch.no_grad()
    def _update_kmeans(self, feature_extractor):
        import math
        from collections import defaultdict

        N = len(self.normal_dataset)
        if N == 0:
            raise RuntimeError("normal_dataset이 비어 있습니다.")

        # 1) 모든 normal 샘플에 대해 전역 벡터 추출
        feat_mat = []
        idx_list = []
        for idx in range(N):
            img, _, _ = self.normal_dataset[idx]
            img = img.to(self.device)
            feats = feature_extractor(img.unsqueeze(0))          # list of tensors
            vec = self._to_vector(feats[1:-1]).cpu().numpy()     # (1, D)
            feat_mat.append(vec)
            idx_list.append(idx)
        X = np.vstack(feat_mat)                                  # (N, D)

        # 2) K-means 실행
        k = self.kmeans_k if self.kmeans_k is not None else self.nb_memory_sample
        k = max(1, min(k, N))                                    # 안전 가드
        kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
        centers = kmeans.cluster_centers_                         # (k, D)
        labels = kmeans.labels_                                   # (N,)

        # 3) 군집별 quota 계산 (nb_memory_sample를 K로 균등 분배)
        total = min(self.nb_memory_sample, N)
        base = total // k
        rem  = total % k
        quotas = [base + (1 if i < rem else 0) for i in range(k)] # 길이 k

        # 4) 군집별로 센터에 가까운 순서로 quota만큼 선택
        #    - 군집에 샘플이 quota보다 적으면 가능한 만큼만 선택
        per_cluster_indices = defaultdict(list)
        for i, lbl in enumerate(labels):
            per_cluster_indices[int(lbl)].append(i)

        selected_global = []
        # (a) 1차 선택: 각 군집에서 quota만큼
        leftovers = 0
        for c in range(k):
            member_idx = per_cluster_indices.get(c, [])
            if len(member_idx) == 0:
                leftovers += quotas[c]
                continue
            # 군집 c의 각 샘플-센터 거리 계산
            Xc = X[member_idx]                                   # (Nc, D)
            dists = np.linalg.norm(Xc - centers[c][None, :], axis=1)
            order = np.argsort(dists)
            take = min(quotas[c], len(member_idx))
            picked = [member_idx[j] for j in order[:take]]
            selected_global.extend(picked)
            if take < quotas[c]:
                leftovers += (quotas[c] - take)

        # (b) 2차 선택: 남은 몫이 있으면, 아직 안 뽑힌 전체 후보 중에서
        #     "자기 군집 센터에 가까운 순"으로 보충
        if leftovers > 0:
            already = set(selected_global)
            cand = [i for i in range(N) if i not in already]
            if len(cand) > 0:
                # 각 후보 i에 대해 자기 군집 c의 센터까지 거리
                cands_X = X[cand]
                cands_lbls = labels[cand]
                dists = np.linalg.norm(cands_X - centers[cands_lbls], axis=1)
                order = np.argsort(dists)
                fill = min(leftovers, len(cand))
                selected_global.extend([cand[j] for j in order[:fill]])

        # 최종 잘라내기(혹시라도 초과했다면)
        if len(selected_global) > total:
            selected_global = selected_global[:total]

        # 5) 선택된 샘플들의 원본 feature map을 level별로 수집해 메모리에 저장
        for sel_i in selected_global:
            orig_idx = idx_list[sel_i]
            img, _, _ = self.normal_dataset[orig_idx]
            img = img.to(self.device)
            feats = feature_extractor(img.unsqueeze(0))
            for li, f_l in enumerate(feats[1:-1]):
                key = f"level{li}"
                f_l = f_l.detach()
                if key not in self.memory_information:
                    self.memory_information[key] = f_l
                else:
                    self.memory_information[key] = torch.cat(
                        [self.memory_information[key], f_l], dim=0
                    )
        # === DEBUG ===
        expected = min(self.nb_memory_sample, N) if self.kmeans_k else min(self.nb_memory_sample, N)
        actual = next(iter(self.memory_information.values())).shape[0]
        print(f"[DEBUG][KMeans] k={k}, dataset={N}, "
              f"expected≈{expected} (k*per_cluster), actual={actual}, "
              f"nb_memory_sample={self.nb_memory_sample}, kmeans_k={self.kmeans_k}")
    # patch v3 end

    # -------------------------
    # Inference path (기존 유지)
    # -------------------------
    def _calc_diff(self, features: List[torch.Tensor]) -> torch.Tensor:
        # batch size X the number of samples saved in memory
        nb = next(iter(self.memory_information.values())).size(0)
        diff_bank = torch.zeros(features[0].size(0), nb).to(self.device)

        for l, level in enumerate(self.memory_information.keys()):
            mem = self.memory_information[level]  # (nb, C,H,W)
            for b_idx, f_b in enumerate(features[l]):  # (C,H,W)
                diff = F.mse_loss(
                    input=torch.repeat_interleave(f_b.unsqueeze(0), repeats=nb, dim=0),
                    target=mem,
                    reduction="none",
                ).mean(dim=[1, 2, 3])
                diff_bank[b_idx] += diff
        return diff_bank

    def select(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        diff_bank = self._calc_diff(features=features)
        # 최소 거리 샘플 선택 후 차이맵 concat (기존 로직 유지)
        argmins = diff_bank.argmin(dim=1)
        for l, level in enumerate(self.memory_information.keys()):
            mem = self.memory_information[level]
            selected = torch.index_select(mem, dim=0, index=argmins)         # (B,C,H,W)
            diff_features = F.mse_loss(selected, features[l], reduction="none")
            features[l] = torch.cat([features[l], diff_features], dim=1)      # 채널 concat
        return features
    
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

    @torch.no_grad()
    def _update_kmeans(self, feature_extractor):
        N = len(self.normal_dataset)
        if N == 0:
            raise RuntimeError("normal_dataset이 비어 있습니다.")

        # 1) 모든 normal 샘플에 대해 전역 벡터 추출
        feat_mat = []
        idx_list = []
        for idx in range(N):
            img, _, _ = self.normal_dataset[idx]
            img = img.to(self.device)

            feats = feature_extractor(img.unsqueeze(0))
            # features[1:-1] 사용하는 것은 기존 설계와 동일
            vec = self._to_vector(feats[1:-1]).cpu().numpy()    # (1,D)
            feat_mat.append(vec)
            idx_list.append(idx)

        X = np.vstack(feat_mat)          # (N, D)

        # 2) K-means 실행
        k = self.kmeans_k if self.kmeans_k is not None else self.nb_memory_sample
        k = max(1, min(k, N))            # 안전 가드

        kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
        centers = kmeans.cluster_centers_
        closest, _ = pairwise_distances_argmin_min(centers, X)

        selected_indices = [idx_list[i] for i in closest]

        # 3) 선택된 샘플들의 원본 feature map을 level별로 수집해 메모리에 저장
        for idx in selected_indices:
            img, _, _ = self.normal_dataset[idx]
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

        # 선택 개수가 nb_memory_sample과 다르면(예: N < k),
        # diff 계산/선택에는 영향이 없지만, 필요시 여기서 보정 로직을 추가할 수 있음.

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
    
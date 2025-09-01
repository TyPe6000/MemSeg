# train.py
# v1 : cloned source from memseg
# v2 : 2025-08-21, adding more metrics json dump
# v3 : 2025-09-01, GPU native AUROC (binary) patch
import time
import json
import os 
import wandb
import logging

import torch
import torch.nn.functional as F
import numpy as np
from typing import List
from sklearn.metrics import roc_auc_score
from metrics import compute_pro, trapezoid

_logger = logging.getLogger('train')

# patch v3: ---- GPU-native AUROC (binary) ----
import torch

def torch_binary_auroc(y_true: torch.Tensor, y_score: torch.Tensor) -> torch.Tensor:
    """
    y_true: (N,) in {0,1}
    y_score: (N,) in [0,1]
    returns: scalar tensor (AUC)
    """
    y_true = y_true.reshape(-1).to(dtype=torch.float32)
    y_score = y_score.reshape(-1).to(dtype=torch.float32)

    P = y_true.sum()
    N = y_true.numel() - P
    # 양/음 샘플이 한쪽으로만 존재하면 정의 불가 → NaN 반환(원하면 0.5 등으로 처리)
    if P == 0 or N == 0:
        return torch.nan

    # 점수 내림차순 정렬
    scores, idx = torch.sort(y_score, descending=True)
    labels = y_true[idx]

    # 누적 TP/FP
    tps = torch.cumsum(labels, dim=0)
    fps = torch.cumsum(1.0 - labels, dim=0)

    tpr = tps / P.clamp_min(1)
    fpr = fps / N.clamp_min(1)

    # 중복 점수 처리(고유 점수 경계에서만 샘플)
    first = torch.ones(1, dtype=torch.bool, device=scores.device)
    changes = torch.cat([first, scores[1:] != scores[:-1]])
    tpr_u = torch.cat([tpr.new_zeros(1), tpr[changes]])
    fpr_u = torch.cat([fpr.new_zeros(1), fpr[changes]])

    # 사다리꼴 적분
    auc = torch.trapz(tpr_u, fpr_u)
    return auc
# patch v3: ---- end ----

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def _safe_json_dump(obj, path: str) -> None:
    """원자적 파일 쓰기 (부분파일 방지)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def training(model, trainloader, validloader, criterion, optimizer, scheduler, num_training_steps: int = 1000, loss_weights: List[float] = [0.6, 0.4], 
             log_interval: int = 1, eval_interval: int = 1, savedir: str = None, use_wandb: bool = False, device: str ='cpu',
             save_metrics_json: bool = True, run_meta: dict = None) -> dict:   

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    l1_losses_m = AverageMeter()
    focal_losses_m = AverageMeter()

    # criterion
    l1_criterion, focal_criterion = criterion
    l1_weight, focal_weight = loss_weights
    
    # set train mode
    model.train()

    # set optimizer
    optimizer.zero_grad()

    # training
    best_score = 0
    step = 0
    train_mode = True
    while train_mode:

        end = time.time()
        for inputs, masks, targets in trainloader:
            # batch
            inputs, masks, targets = inputs.to(device), masks.to(device), targets.to(device)

            data_time_m.update(time.time() - end)

            # predict
            outputs = model(inputs)
            outputs = F.softmax(outputs, dim=1)
            l1_loss = l1_criterion(outputs[:,1,:], masks)
            focal_loss = focal_criterion(outputs, masks)
            loss = (l1_weight * l1_loss) + (focal_weight * focal_loss)

            loss.backward()
            
            # update weight
            optimizer.step()
            optimizer.zero_grad()

            # log loss
            l1_losses_m.update(l1_loss.item())
            focal_losses_m.update(focal_loss.item())
            losses_m.update(loss.item())
            
            batch_time_m.update(time.time() - end)

            # wandb
            if use_wandb:
                wandb.log({
                    'lr':optimizer.param_groups[0]['lr'],
                    'train_focal_loss':focal_losses_m.val,
                    'train_l1_loss':l1_losses_m.val,
                    'train_loss':losses_m.val
                },
                step=step)
            
            if (step+1) % log_interval == 0 or step == 0: 
                _logger.info('TRAIN [{:>4d}/{}] '
                            'Loss: {loss.val:>6.4f} ({loss.avg:>6.4f}) '
                            'L1 Loss: {l1_loss.val:>6.4f} ({l1_loss.avg:>6.4f}) '
                            'Focal Loss: {focal_loss.val:>6.4f} ({focal_loss.avg:>6.4f}) '
                            'LR: {lr:.3e} '
                            'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s) '
                            'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                            step+1, num_training_steps, 
                            loss       = losses_m, 
                            l1_loss    = l1_losses_m,
                            focal_loss = focal_losses_m,
                            lr         = optimizer.param_groups[0]['lr'],
                            batch_time = batch_time_m,
                            rate       = inputs.size(0) / batch_time_m.val,
                            rate_avg   = inputs.size(0) / batch_time_m.avg,
                            data_time  = data_time_m))


            if ((step+1) % eval_interval == 0 and step != 0) or (step+1) == num_training_steps: 
                eval_metrics = evaluate(
                    model        = model, 
                    dataloader   = validloader, 
                    device       = device
                )
                model.train()
                # === [METRICS JSON] step별/최신 저장 ===
                if save_metrics_json and savedir is not None:
                    now = int(time.time())
                    try:
                        metrics_record = {
                            "step": int(step) + 1,     # 보기 좋은 1-based
                            "timestamp": now,
                            "device": str(device),
                            # 누적 평균 손실 스냅샷(이 변수명이 다르면 프로젝트 변수명에 맞춰 바꿔주세요)
                            "train_loss_avg": float(losses_m.avg),
                            "train_l1_loss_avg": float(l1_losses_m.avg),
                            "train_focal_loss_avg": float(focal_losses_m.avg),
                            # 평가 메트릭 그대로 포함 (예: AUROC-image, AUROC-pixel, AUPRO-pixel)
                            **{k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                            for k, v in eval_metrics.items()}
                        }
                    except NameError:
                        # 만약 위 손실 변수명이 다르면 손실 항목 없이라도 저장되도록 방어
                        metrics_record = {
                            "step": int(step) + 1,
                            "timestamp": now,
                            "device": str(device),
                            **{k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                            for k, v in eval_metrics.items()}
                        }

                    if run_meta:
                        metrics_record["meta"] = run_meta

                    # step 스냅샷 파일
                    _safe_json_dump(metrics_record, os.path.join(savedir, f"metrics_step{int(step)+1:06d}.json"))
                    # 최신 스냅샷(덮어쓰기)
                    _safe_json_dump(metrics_record, os.path.join(savedir, "metrics_latest.json"))
                # === [END METRICS JSON] ===

                eval_log = dict([(f'eval_{k}', v) for k, v in eval_metrics.items()])

                # wandb
                if use_wandb:
                    wandb.log(eval_log, step=step)

                # checkpoint
                if best_score < np.mean(list(eval_metrics.values())):
                    # save best score
                    state = {'best_step':step}
                    state.update(eval_log)
                    json.dump(state, open(os.path.join(savedir, 'best_score.json'),'w'), indent='\t')

                    # save best model
                    torch.save(model.state_dict(), os.path.join(savedir, f'best_model.pt'))
                    
                    _logger.info('Best Score {0:.3%} to {1:.3%}'.format(best_score, np.mean(list(eval_metrics.values()))))

                    best_score = np.mean(list(eval_metrics.values()))
                    
                    # --- [METRICS JSON] best 저장 ---
                    if save_metrics_json and savedir is not None:
                        now = int(time.time())
                        # 위에서 만든 metrics_record가 있다면 재사용, 없으면 새로 구성
                        try:
                            best_record = dict(metrics_record)
                        except NameError:
                            best_record = {
                                "step": int(step) + 1,
                                "timestamp": now,
                                "device": str(device),
                                **{k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                                for k, v in eval_metrics.items()}
                            }
                            if run_meta:
                                best_record["meta"] = run_meta

                        _safe_json_dump(best_record, os.path.join(savedir, "metrics_best.json"))
                    # --- [END METRICS JSON] ---

            # scheduler
            if scheduler:
                scheduler.step()

            end = time.time()

            step += 1

            if step == num_training_steps:
                train_mode = False
                break

    # print best score and step
    _logger.info('Best Metric: {0:.3%} (step {1:})'.format(best_score, state['best_step']))

    # save latest model
    torch.save(model.state_dict(), os.path.join(savedir, f'latest_model.pt'))

    # save latest score
    state = {'latest_step':step}
    state.update(eval_log)
    json.dump(state, open(os.path.join(savedir, 'latest_score.json'),'w'), indent='\t')

    

# patch v3: GPU-native AUROC (binary) ----

# def evaluate(model, dataloader, device: str = 'cpu'):
#     # targets and outputs
#     image_targets = []
#     image_masks = []
#     anomaly_score = []
#     anomaly_map = []

#     model.eval()
#     with torch.no_grad():
#         for idx, (inputs, masks, targets) in enumerate(dataloader):
#             inputs, masks, targets = inputs.to(device), masks.to(device), targets.to(device)
            
#             # predict
#             outputs = model(inputs)
#             outputs = F.softmax(outputs, dim=1)
#             anomaly_score_i = torch.topk(torch.flatten(outputs[:,1,:], start_dim=1), 100)[0].mean(dim=1)

#             # stack targets and outputs
#             image_targets.extend(targets.cpu().tolist())
#             image_masks.extend(masks.cpu().numpy())
            
#             anomaly_score.extend(anomaly_score_i.cpu().tolist())
#             anomaly_map.extend(outputs[:,1,:].cpu().numpy())
            
#     # metrics    
#     image_masks = np.array(image_masks)
#     anomaly_map = np.array(anomaly_map)
    
#     auroc_image = roc_auc_score(image_targets, anomaly_score)
#     auroc_pixel = roc_auc_score(image_masks.reshape(-1).astype(int), anomaly_map.reshape(-1))
#     all_fprs, all_pros = compute_pro(
#         anomaly_maps      = anomaly_map,
#         ground_truth_maps = image_masks
#     )
#     aupro = trapezoid(all_fprs, all_pros)
    
#     metrics = {
#         'AUROC-image':auroc_image,
#         'AUROC-pixel':auroc_pixel,
#         'AUPRO-pixel':aupro

#     }

#     _logger.info('TEST: AUROC-image: %.3f%% | AUROC-pixel: %.3f%% | AUPRO-pixel: %.3f%%' % 
#                 (metrics['AUROC-image'], metrics['AUROC-pixel'], metrics['AUPRO-pixel']))


#     return metrics

def evaluate(model, dataloader, device: str = 'cpu'):
    image_targets = []
    image_masks = []
    anomaly_scores = []
    anomaly_maps = []

    model.eval()
    with torch.no_grad():
        for inputs, masks, targets in dataloader:
            inputs  = inputs.to(device)
            masks   = masks.to(device)
            targets = targets.to(device)

            outputs = model(inputs)                 # [B,2,H,W]
            outputs = F.softmax(outputs, dim=1)
            # 이미지 점수(top-k 평균)도 GPU에서 계산
            probs1  = outputs[:, 1, ...]            # [B,H,W]
            flat    = probs1.flatten(start_dim=1)    # [B, H*W]
            k       = min(100, flat.shape[1])
            score_i = torch.topk(flat, k=k, dim=1).values.mean(dim=1)

            image_targets.append(targets.float())    # [B]
            image_masks.append(masks.float())        # [B,H,W]
            anomaly_scores.append(score_i)           # [B]
            anomaly_maps.append(probs1)              # [B,H,W]

    # ---- GPU-native AUROC (image-level) ----
    y_img = torch.cat(image_targets, dim=0).to(device)     # [N]
    s_img = torch.cat(anomaly_scores, dim=0).to(device)    # [N]
    auroc_image = torch_binary_auroc(y_img, s_img).item()

    # ---- GPU-native AUROC (pixel-level) ----
    y_pix = torch.cat(image_masks, dim=0).reshape(-1).to(device)   # [N*H*W]
    s_pix = torch.cat(anomaly_maps, dim=0).reshape(-1).to(device)  # [N*H*W]
    auroc_pixel = torch_binary_auroc(y_pix, s_pix).item()

    # ---- AUPRO는 당분간 CPU 유지 (compute_pro가 numpy 전제) ----
    masks_np = torch.cat(image_masks, dim=0).cpu().numpy()         # [N,H,W]
    maps_np  = torch.cat(anomaly_maps, dim=0).cpu().numpy()        # [N,H,W]
    all_fprs, all_pros = compute_pro(anomaly_maps=maps_np, ground_truth_maps=masks_np)
    aupro = trapezoid(all_fprs, all_pros)

    metrics = {
        'AUROC-image': auroc_image,
        'AUROC-pixel': auroc_pixel,
        'AUPRO-pixel': aupro
    }

    _logger.info('TEST: AUROC-image: %.3f%% | AUROC-pixel: %.3f%% | AUPRO-pixel: %.3f%%' %
                 (metrics['AUROC-image'], metrics['AUROC-pixel'], metrics['AUPRO-pixel']))
    return metrics
# patch v3: ---- end ----
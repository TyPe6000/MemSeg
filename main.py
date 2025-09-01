import wandb
import logging
import os
import torch
import torch.nn as nn
import argparse
from torch.utils.data import ConcatDataset

from omegaconf import OmegaConf
from timm import create_model
from data import create_dataset, create_dataloader
from models import MemSeg, MemoryBank
from focal_loss import FocalLoss
from train import training
from log import setup_default_logging
from utils import torch_seed
from scheduler import CosineAnnealingWarmupRestarts


_logger = logging.getLogger('train')



def run(cfg):

    # setting seed and device
    setup_default_logging()
    torch_seed(cfg.SEED)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    _logger.info('Device: {}'.format(device))

    # # savedir
    # cfg.EXP_NAME = cfg.EXP_NAME + f"-{cfg.DATASET.target}"
    # - patch: 다권종 데이터셋 최적화 - 
    # --- target(s) normalize ---
    def _as_list(x):
        if isinstance(x, (list, tuple)):
            return [str(t).strip() for t in x if str(t).strip()]
        if isinstance(x, str) and ("," in x):
            return [t.strip() for t in x.split(",") if t.strip()]
        return [x]

    targets_list = _as_list(cfg.DATASET.target)

    # savedir (권종 여러 개면 깔끔하게 표기)
    if len(targets_list) == 1:
        suffix = targets_list[0]
    else:
        head = "+".join(targets_list[:3])
        more = f"+{len(targets_list)-3}more" if len(targets_list) > 3 else ""
        suffix = f"multi-{head}{more}"
    cfg.EXP_NAME = cfg.EXP_NAME + f"-{suffix}"
    # - patch: 다권종 데이터셋 최적화 -
    savedir = os.path.join(cfg.RESULT.savedir, cfg.EXP_NAME)
    os.makedirs(savedir, exist_ok=True)

    
    # wandb
    if cfg.TRAIN.use_wandb:
        wandb.init(name=cfg.EXP_NAME, project='MemSeg', config=OmegaConf.to_container(cfg))

    # # build datasets
    # trainset = create_dataset(
    #     datadir                = cfg.DATASET.datadir,
    #     target                 = cfg.DATASET.target, 
    #     is_train               = True,
    #     resize                 = cfg.DATASET.resize,
    #     imagesize              = cfg.DATASET.imagesize,
    #     texture_source_dir     = cfg.DATASET.texture_source_dir,
    #     structure_grid_size    = cfg.DATASET.structure_grid_size,
    #     transparency_range     = cfg.DATASET.transparency_range,
    #     perlin_scale           = cfg.DATASET.perlin_scale,
    #     min_perlin_scale       = cfg.DATASET.min_perlin_scale,
    #     perlin_noise_threshold = cfg.DATASET.perlin_noise_threshold,
    #     use_mask               = cfg.DATASET.use_mask,
    #     bg_threshold           = cfg.DATASET.bg_threshold,
    #     bg_reverse             = cfg.DATASET.bg_reverse
    # )
    # print(f"Trainset size: {len(trainset)}")
    # print(f"Trainset sample files: {getattr(trainset, 'files', 'N/A')}")

    # memoryset = create_dataset(
    #     datadir   = cfg.DATASET.datadir,
    #     target    = cfg.DATASET.target, 
    #     is_train  = True,
    #     to_memory = True,
    #     resize    = cfg.DATASET.resize,
    #     imagesize = cfg.DATASET.imagesize,
    # )

    # testset = create_dataset(
    #     datadir   = cfg.DATASET.datadir,
    #     target    = cfg.DATASET.target, 
    #     is_train  = False,
    #     resize    = cfg.DATASET.resize,
    #     imagesize = cfg.DATASET.imagesize,
    # )
    # - patch: 다권종 데이터셋 최적화 -
    # build datasets (단일/다중 타겟 공용)
    def _build_set(is_train: bool, to_memory: bool = False):
        ds_list = []
        for t in targets_list:
            ds = create_dataset(
                datadir                = cfg.DATASET.datadir,
                target                 = t,
                is_train               = is_train,
                to_memory              = to_memory,
                resize                 = cfg.DATASET.resize,
                imagesize              = cfg.DATASET.imagesize,
                texture_source_dir     = getattr(cfg.DATASET, "texture_source_dir", None),
                structure_grid_size    = cfg.DATASET.structure_grid_size,
                transparency_range     = cfg.DATASET.transparency_range,
                perlin_scale           = cfg.DATASET.perlin_scale,
                min_perlin_scale       = cfg.DATASET.min_perlin_scale,
                perlin_noise_threshold = cfg.DATASET.perlin_noise_threshold,
                use_mask               = cfg.DATASET.use_mask,
                bg_threshold           = cfg.DATASET.bg_threshold,
                bg_reverse             = cfg.DATASET.bg_reverse
            )
            ds_list.append(ds)
        return ds_list[0] if len(ds_list) == 1 else ConcatDataset(ds_list)

    # datasets
    trainset  = _build_set(is_train=True,  to_memory=False)
    memoryset = _build_set(is_train=True,  to_memory=True)
    testset   = _build_set(is_train=False, to_memory=False)

    print(f"Trainset size:  {len(trainset)}")
    print(f"Memoryset size: {len(memoryset)}")
    print(f"Testset size:   {len(testset)}")
    # - patch: 다권종 데이터셋 최적화 -

    # build dataloader
    trainloader = create_dataloader(
        dataset     = trainset,
        train       = True,
        batch_size  = cfg.DATALOADER.batch_size,
        num_workers = cfg.DATALOADER.num_workers
    )
    
    testloader = create_dataloader(
        dataset     = testset,
        train       = False,
        batch_size  = cfg.DATALOADER.batch_size,
        num_workers = cfg.DATALOADER.num_workers
    )


    # build feature extractor
    feature_extractor = create_model(
        cfg.MODEL.feature_extractor_name, 
        pretrained    = True, 
        features_only = True
    ).to(device)
    ## freeze weight of layer1,2,3
    for l in ['layer1','layer2','layer3']:
        for p in feature_extractor[l].parameters():
            p.requires_grad = False

    # patch
    # # build memory bank
    # memory_bank = MemoryBank(
    #     normal_dataset   = memoryset,
    #     nb_memory_sample = cfg.MEMORYBANK.nb_memory_sample,
    #     device           = device
    # )
    # ## update normal samples and save
    # memory_bank.update(feature_extractor=feature_extractor)
    # torch.save(memory_bank, os.path.join(savedir, f'memory_bank.pt'))
    # _logger.info('Update {} normal samples in memory bank'.format(cfg.MEMORYBANK.nb_memory_sample))

    # build memory bank
    memory_bank = MemoryBank(
        normal_dataset   = memoryset,
        nb_memory_sample = cfg.MEMORYBANK.nb_memory_sample,
        device           = device,
        selector         = getattr(cfg.MEMORYBANK, 'selector', 'random'),
        kmeans_k         = getattr(cfg.MEMORYBANK, 'kmeans_k', None),
    )
    ## update normal samples and save
    memory_bank.update(feature_extractor=feature_extractor)
    torch.save(memory_bank, os.path.join(savedir, f'memory_bank.pt'))
    _logger.info('Update {} normal samples in memory bank (selector: {})'
                 .format(cfg.MEMORYBANK.nb_memory_sample, getattr(cfg.MEMORYBANK, 'selector', 'random')))

    # build MemSeg
    use_asff = False
    if hasattr(cfg.MODEL, 'use_asff'):
        use_asff = cfg.MODEL.use_asff

    # feature_extractor의 출력 채널 리스트 추출
    with torch.no_grad():
        dummy = torch.zeros(1, 3, cfg.DATASET.imagesize, cfg.DATASET.imagesize).to(device)
        features = feature_extractor(dummy)
        feature_channels = [f.shape[1] for f in features]

    model = MemSeg(
        memory_bank       = memory_bank,
        feature_extractor = feature_extractor,
        feature_channels  = feature_channels,
        use_asff          = use_asff
    ).to(device)

    # Set training
    l1_criterion = nn.L1Loss()
    f_criterion = FocalLoss(
        gamma = cfg.TRAIN.focal_gamma, 
        alpha = cfg.TRAIN.focal_alpha
    )

    optimizer = torch.optim.AdamW(
        params       = filter(lambda p: p.requires_grad, model.parameters()), 
        lr           = cfg.OPTIMIZER.lr, 
        weight_decay = cfg.OPTIMIZER.weight_decay
    )

    if cfg['SCHEDULER']['use_scheduler']:
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer, 
            first_cycle_steps = cfg.TRAIN.num_training_steps,
            max_lr = cfg.OPTIMIZER.lr,
            min_lr = cfg.SCHEDULER.min_lr,
            warmup_steps   = int(cfg.TRAIN.num_training_steps * cfg.SCHEDULER.warmup_ratio)
        )
    else:
        scheduler = None

    # Fitting model
    training(
        model              = model, 
        num_training_steps = cfg.TRAIN.num_training_steps, 
        trainloader        = trainloader, 
        validloader        = testloader, 
        criterion          = [l1_criterion, f_criterion], 
        loss_weights       = [cfg.TRAIN.l1_weight, cfg.TRAIN.focal_weight],
        optimizer          = optimizer,
        scheduler          = scheduler,
        log_interval       = cfg.LOG.log_interval,
        eval_interval      = cfg.LOG.eval_interval,
        savedir            = savedir,
        device             = device,
        use_wandb          = cfg.TRAIN.use_wandb,
        run_meta           = {                              # - patch: 다권종 데이터셋 최적화 -
            "targets": targets_list,
            "use_asff": bool(use_asff),
            "memory_selector": getattr(cfg.MEMORYBANK, 'selector', 'random'),
            "kmeans_k": getattr(cfg.MEMORYBANK, 'kmeans_k', None),
        }
    )




if __name__=='__main__':
    import sys
    from omegaconf import DictConfig
    args = OmegaConf.from_cli()
    # load default config
    cfg = OmegaConf.load(args.configs)
    del args['configs']

    # add use_asff argument if not present
    if 'use_asff' not in args:
        args['use_asff'] = False
    if 'MODEL' not in cfg:
        cfg.MODEL = {}
    cfg.MODEL['use_asff'] = args['use_asff']

    # merge config with new keys
    cfg = OmegaConf.merge(cfg, args)

    # target cfg
    # target_cfg = OmegaConf.load(cfg.DATASET.anomaly_mask_info)

    # 가드 추가
    if 'target' not in cfg.DATASET or cfg.DATASET.target is None:
        raise ValueError(
            "DATASET.target 이 설정되어 있지 않습니다. "
            "예: DATASET.target=50_T3_150"
        )

    # available = list(target_cfg.keys()) if hasattr(target_cfg, 'keys') else []
    # if cfg.DATASET.target not in available:
    #     raise KeyError(
    #         f"'{cfg.DATASET.target}' 키를 {cfg.DATASET.anomaly_mask_info}에서 찾지 못했습니다. "
    #         f"사용 가능: {available}"
    #     )

    # cfg.DATASET = OmegaConf.merge(cfg.DATASET, target_cfg[cfg.DATASET.target])

    # anomaly_mask_info 병합: 단일 타겟만 엄격 병합, 멀티는 사용자 DATASET 공통 설정 사용
    targets_list = cfg.DATASET.target if isinstance(cfg.DATASET.target, (list, tuple)) else (
        [t.strip() for t in str(cfg.DATASET.target).split(",")] if (isinstance(cfg.DATASET.target, str) and "," in cfg.DATASET.target) else [cfg.DATASET.target]
    )
    if len(targets_list) == 1:
        target_cfg = OmegaConf.load(cfg.DATASET.anomaly_mask_info)
        available = list(target_cfg.keys()) if hasattr(target_cfg, 'keys') else []
        if targets_list[0] not in available:
            raise KeyError(
                f"'{targets_list[0]}' 키를 {cfg.DATASET.anomaly_mask_info}에서 찾지 못했습니다. "
                f"사용 가능: {available}"
            )
        cfg.DATASET = OmegaConf.merge(cfg.DATASET, target_cfg[targets_list[0]])
    else:
        # 멀티-타겟: 권종별 세부 마스크 파라미터가 상이할 수 있어 별도 병합 생략
        # (DATASET의 공통 파라미터로 동작; 필요 시 per-target 병합 규칙을 추후 확장)
        pass

    print(OmegaConf.to_yaml(cfg))

    run(cfg)

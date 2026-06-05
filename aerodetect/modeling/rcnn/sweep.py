from aerodetect.modeling.rcnn.rcnn_detector import RcnnDetector


RUNS = [
    # 1) 1e-3, no scheduler (simple, strong baseline)
    {
        "run_name": "r50_lr1e-3_none_e12",
        "lr": 1e-3,
        "epochs": 12,
        "batchsize": 8,
        "img_size": 640,
        "augs": "light",
        "weighted_sampling": True,
        "optimizer_name": "sgd",
        "scheduler_name": "none",
    },

    # 2) 1e-3, warmup + cosine
    {
        "run_name": "r50_lr1e-3_warmcos_e12",
        "lr": 1e-3,
        "epochs": 12,
        "batchsize": 8,
        "img_size": 640,
        "augs": "light",
        "weighted_sampling": True,
        "optimizer_name": "sgd",
        "scheduler_name": "warmup_cosine",
        "warmup_epochs": 2,
    },

    # 3) 3e-4, cosine (lower LR variant)
    {
        "run_name": "r50_lr3e-4_cosine_e12",
        "lr": 3e-4,
        "epochs": 12,
        "batchsize": 8,
        "img_size": 640,
        "augs": "light",
        "weighted_sampling": True,
        "optimizer_name": "sgd",
        "scheduler_name": "cosine",
    },
]


if __name__ == "__main__":
    for cfg in RUNS:
        try:
            detector = RcnnDetector(
                model_name="fasterrcnn_resnet50_fpn",
                dataset_name="military",
                num_workers=4,
                trainable_backbone_layers=3,
                pretrained=True,
                class_metrics=True,
                use_amp=True,
                sweep_name="rcnn_resnet50_budget_sweep_20260605",
                **cfg,
            )
            detector.train()
        except Exception as e:
            print(f"Run failed: {cfg['run_name']} -> {e}")
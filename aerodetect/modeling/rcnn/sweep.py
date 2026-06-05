from aerodetect.modeling.rcnn.rcnn_detector import RcnnDetector

RUNS = [
    {
        "run_name": "rcnn_baseline_plain",
        "lr": 0.001,
        "epochs": 8,          # short smoke test
        "batchsize": 8,
        "img_size": 640,
        "augs": False,
        "weighted_sampling": False,
        "optimizer_name": "sgd",
        "scheduler_name": "none",
    },
    {
        "run_name": "rcnn_balance_sampler",
        "lr": 0.001,
        "epochs": 8,
        "batchsize": 8,
        "img_size": 640,
        "augs": False,
        "weighted_sampling": True,
        "optimizer_name": "sgd",
        "scheduler_name": "none",
    },
    {
        "run_name": "rcnn_aug_light",
        "lr": 0.001,
        "epochs": 8,
        "batchsize": 8,
        "img_size": 640,
        "augs": "light",
        "weighted_sampling": False,
        "optimizer_name": "sgd",
        "scheduler_name": "none",
    },
    {
        "run_name": "rcnn_balance_aug",
        "lr": 0.001,
        "epochs": 8,
        "batchsize": 8,
        "img_size": 640,
        "augs": "light",
        "weighted_sampling": True,
        "optimizer_name": "sgd",
        "scheduler_name": "none",
    },
    {
        "run_name": "rcnn_balance_aug_cosine",
        "lr": 0.001,
        "epochs": 8,
        "batchsize": 4,
        "img_size": 640,
        "augs": "light",
        "weighted_sampling": True,
        "optimizer_name": "sgd",
        "scheduler_name": "warmup_cosine",
        "warmup_epochs": 1,
    },
]


if __name__ == "__main__":
    for cfg in RUNS:
        try:
            detector = RcnnDetector(
                model_name="fasterrcnn_mobilenet_v3_large_fpn",
                dataset_name="military",
                num_workers=4,
                trainable_backbone_layers=3,
                pretrained=True,
                class_metrics=True,
                use_amp=True,
                sweep_name="rcnn_tune_sweep_05062026",
                **cfg,
            )
            detector.train()
        except Exception as e:
            print(f"Run failed: {cfg['run_name']} -> {e}")
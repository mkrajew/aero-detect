from aerodetect.modeling.rcnn.rcnn_detector import RcnnDetector


RUNS = [
    # # 1) Continue best MobileNet run from its best checkpoint to 50 total epochs
    # {
    #     "run_name": "mbv3_bestcfg_long_resume_e50",
    #     "lr": 0.001,
    #     "epochs": 50,  # total epochs (0..49); train() will start at start_epoch
    #     "batchsize": 10,
    #     "img_size": 640,
    #     "augs": False,
    #     "weighted_sampling": True,
    #     "optimizer_name": "sgd",
    #     "scheduler_name": "none",
    #     "resume_from": "aerodetect/detection-yolo/fasterrcnn_mobilenet_v3_large_fpn-best-model:v177",
    # },

    # 2) New short ResNet run with same best config (no resume)
    {
        "run_name": "r50_bestcfg_match_mbv3_e10",
        "lr": 0.001,
        "epochs": 10,
        "batchsize": 8,
        "img_size": 640,
        "augs": False,
        "weighted_sampling": True,
        "optimizer_name": "sgd",
        "scheduler_name": "none",
        "resume_from": None,
    },
]


if __name__ == "__main__":
    for cfg in RUNS:
        try:
            model_name = (
                "fasterrcnn_mobilenet_v3_large_fpn"
                if cfg["run_name"].startswith("mbv3")
                else "fasterrcnn_resnet50_fpn"
            )

            # pop resume_from so it doesn’t go into RcnnDetector kwargs accidentally,
            # or keep it if your RcnnDetector accepts it explicitly
            resume_from = cfg.pop("resume_from", None)

            detector = RcnnDetector(
                model_name=model_name,
                dataset_name="military",
                num_workers=5 if "mbv3" in cfg["run_name"] else 4,
                trainable_backbone_layers=3,
                pretrained=True,
                class_metrics=True,
                use_amp=True,
                sweep_name="rcnn_bestcfg_resume_and_resnet_match_20260606",
                resume_from=resume_from,
                **cfg,
            )
            detector.train()
        except Exception as e:
            print(f"Run failed: {cfg['run_name']} -> {e}")
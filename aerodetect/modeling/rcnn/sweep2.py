from aerodetect.modeling.rcnn.rcnn_detector import RcnnDetector


RUNS = [
        {
        "run_name": "skyfusion_resnet_bestcfg_lr005_800",
        "lr": 0.003,
        "epochs": 50,  
        "batchsize": 32,
        "img_size": 800,
        "augs": "light",
        "weighted_sampling": True,
        # "downscale_anchor": True,
        "optimizer_name": "sgd",
        "scheduler_name": "none"    },
]


if __name__ == "__main__":
    for cfg in RUNS:
        try:

            detector = RcnnDetector(
                model_name="fasterrcnn_resnet50_fpn",
                dataset_name="skyfusion",
                num_workers=2,
                trainable_backbone_layers=3,
                pretrained=True,
                class_metrics=True,
                use_amp=True,
                sweep_name="sky_fusion_sweep",
                **cfg,
            )
            print("Anchor sizes:", detector.model.rpn.anchor_generator.sizes)
            print("Anchor aspect ratios:", detector.model.rpn.anchor_generator.aspect_ratios)
            print("Anchors per location:", detector.model.rpn.anchor_generator.num_anchors_per_location())
            detector.train()
        except Exception as e:
            print(f"Run failed: {cfg['run_name']} -> {e}")
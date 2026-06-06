from aerodetect.modeling.rcnn.rcnn_detector import RcnnDetector


RUNS = [
        {
        "run_name": "resnet_skyfusion_best_cfg",
        "lr": 0.0025,
        "epochs": 80,  
        "batchsize": 48,
        "img_size": 800,
        "augs": "light",
        "weighted_sampling": True,
        "optimizer_name": "sgd",
        "scheduler_name": "none" 
        },
        # {
        # "run_name": "mbv3_skyfusion_bestcfg_800_32_anchor",
        # "lr": 0.003,
        # "epochs": 50,  
        # "batchsize": 32,
        # "img_size": 800,
        # "augs": "light",
        # "weighted_sampling": True,
        # "downscale_anchor": True,
        # "optimizer_name": "sgd",
        # "scheduler_name": "none"    },
]


if __name__ == "__main__":
    for cfg in RUNS:
        try:
            
            model_name = (
                "fasterrcnn_mobilenet_v3_large_fpn"
                if cfg["run_name"].startswith("mbv3")
                else "fasterrcnn_resnet50_fpn"
            )

            detector = RcnnDetector(
                model_name=model_name,
                dataset_name="skyfusion",
                num_workers=5,
                trainable_backbone_layers=3,
                pretrained=True,
                class_metrics=True,
                use_amp=True,
                sweep_name="sky_fusion_sweep",
                **cfg,
            )
    
            detector.train()
        except Exception as e:
            print(f"Run failed: {cfg['run_name']} -> {e}")
from aerodetect.modeling.rcnn.rcnn_detector import RcnnDetector


RUNS = [
        #{
        # "run_name": "resnet_skyfusion_continue_best_cfg",
        # "lr": 0.003,
        # "epochs": 50,  
        # "batchsize": 32,
        # "img_size": 800,
        # "augs": "light",
        # "weighted_sampling": True,
        # "optimizer_name": "sgd",
        # "scheduler_name": "none",
        #  "resume_from": "aerodetect/detection-yolo/fasterrcnn_resnet50_fpn-best-model:v78",  
        # },
        {
            "run_name": "rs50_military_best_cfg",
            "lr": 0.003,          # more aggressive than 0.001
            "epochs": 80,          # strict cap
            "batchsize": 48,      # keep this if GPU is tight
            "img_size": 800,
            "augs": "light",      # turn on light augs
            "weighted_sampling": True,
            "optimizer_name": "sgd",
            "scheduler_name": "none",
            "resume_from": "aerodetect/detection-yolo/fasterrcnn_resnet50_fpn-best-model:v52",
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

            detector = RcnnDetector(
                model_name=model_name,
                dataset_name="skyfusion",
                num_workers=4,
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
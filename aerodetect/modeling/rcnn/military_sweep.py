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
            "lr": 0.0015,          
            "epochs": 80,         
            "batchsize": 48,      
            "img_size": 640,
            "augs": "light",      
            "weighted_sampling": True,
            "optimizer_name": "sgd",
            "scheduler_name": "none",
            "resume_from": "aerodetect/detection-yolo/fasterrcnn_resnet50_fpn-best-model:v67"  
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
                sweep_name="military_run",
                **cfg,
            )
    
            detector.train()
        except Exception as e:
            print(f"Run failed: {cfg['run_name']} -> {e}")

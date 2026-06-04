from pathlib import Path
from torchvision.io.image import decode_image
from torchvision.models.detection import fasterrcnn_resnet50_fpn, fasterrcnn_resnet50_fpn_v2, fasterrcnn_mobilenet_v3_large_fpn
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from collections import defaultdict
from torchvision.utils import draw_bounding_boxes
from military_dataset import MilitaryDataset
from datetime import datetime
import wandb
from aerodetect.config import RCNN_CHECKPOINTS_DIR



MODEL_REGISTRY = {
    "fasterrcnn_mobilenet_v3_large_fpn": fasterrcnn_mobilenet_v3_large_fpn,
    "fasterrcnn_resnet50_fpn_v2": fasterrcnn_resnet50_fpn_v2,
    "fasterrcnn_resnet50_fpn": fasterrcnn_resnet50_fpn,
}


DATASET_REGISTRY = {
    "military" :  MilitaryDataset,
    "skyfusion" : None
}


def collate_fn(batch):
    return tuple(zip(*batch))



class RcnnDetector:
    def __init__(self, model_name, dataset_name,  lr,epochs, batchsize,num_workers, augs,  trainable_backbone_layers, pretrained, weighted_sampling, warmup_epochs = 0 , momentum=0.9,
        weight_decay=0.0005):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self._model_ref = MODEL_REGISTRY[model_name]
        self._dataset_ref = DATASET_REGISTRY[dataset_name]
        self.lr =  lr
        self.batchsize = batchsize
        self.augs = augs
        self.trainable_backbone_layers = trainable_backbone_layers
        self.epochs = epochs
        self.num_workers = num_workers
        self.pretrained =  pretrained
        self.weighted_sampling = weighted_sampling
        self.device = torch.device( 
               "cuda"
                if torch.cuda.is_available()
                else "cpu"
                )
        
        self.build_model()
        self.warump_epochs = warmup_epochs
        self.momentum = momentum
        self.weight_decay = weight_decay



    def build_training_data(self):
        self.train_dataset = self.build_dataset("train")
        self.val_dataset = self.build_dataset("val")

        dataset_weights = self.train_dataset.build_sampling_weight_map()
        dataset_weights = torch.tensor(dataset_weights, dtype=torch.double)

        train_sampler = WeightedRandomSampler(
            weights=dataset_weights,
            num_samples=len(dataset_weights),
            replacement=True,
        )

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batchsize,
            sampler=train_sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )

        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batchsize,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )

        return train_loader, val_loader




    def build_test_data(self):
         self.test_dataset = self.build_dataset('test')
         return self.build_data_loader(self.test_dataset, False)





    def build_model(self):
        num_classes = self._dataset_ref.get_class_number()
        model = None
        if self.pretrained:
            model = self._model_ref(
                weights="DEFAULT",
                trainable_backbone_layers=self.trainable_backbone_layers,
            )
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        else:
            model = self._model_ref(
                num_classes=num_classes,
                trainable_backbone_layers=self.trainable_backbone_layers,
            )

        model.to(self.device)
        self.model = model
       





    def build_dataset(self, split):
            return self._dataset_ref(split = split)
    
    def build_data_loader(self,dataset, shuffle = True):
        return DataLoader(
            dataset,
            batch_size=self.batchsize,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )

    def train_one_epoch(self, data_loader, optimizer, epoch_id, run):
        running_losses = defaultdict(float)
        epoch_losses = defaultdict(float)


        self.model.train()


        use_amp = self.device.type == "cuda"
        scaler = getattr(self, "scaler", None)
        if scaler is None:
            self.scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
            scaler = self.scaler


        for i, (images, targets) in enumerate(tqdm(data_loader)):
            optimizer.zero_grad(set_to_none=True)


            images = [img.to(self.device) for img in images]
            targets = [
                {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in t.items()
                }
                for t in targets
            ]


            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                loss_dict = self.model(images, targets)
                loss = sum(loss_dict.values())


            for k, v in loss_dict.items():
                running_losses[k] += v.detach().item()
                epoch_losses[k] += v.detach().item()


            running_losses["loss"] += loss.detach().item()
            epoch_losses["loss"] += loss.detach().item()


            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()


            if (i + 1) % 1000 == 0:
                avg_losses = {k: v / 1000 for k, v in running_losses.items()}


                loss_str = " | ".join(
                    f"{k}: {v:.4f}"
                    for k, v in avg_losses.items()
                )
                print(f"[batch {i+1}] | {loss_str}")


                run.log(
                    {
                        "train/batch_step": self._global_step,
                        **{f"train/batch/{k}": v for k, v in avg_losses.items()},
                    }
                )


                running_losses.clear()
                self._global_step += 1


        num_batches = len(data_loader)



        epoch_avg_losses = {
            k: v / num_batches
            for k, v in epoch_losses.items()
        }


        loss_str = " | ".join(
            f"{k}: {v:.4f}"
            for k, v in epoch_avg_losses.items()
        )
        print(f"Epoch {epoch_id + 1} summary | {loss_str}")


        run.log(
            {
                "epoch": epoch_id + 1,
                **{f"train/epoch/{k}": v for k, v in epoch_avg_losses.items()},
            }
        )


        return epoch_avg_losses["loss"]




    def eval(self,data_loader, metric, run, epoch):
        self.model.eval()
        metric.reset()


        with torch.no_grad():
            for i, data in enumerate(tqdm(data_loader)):
                images, targets = data
                images = list(image.to(self.device) for image in images)
                targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
                
                outputs =  self.model(images)
                outputs = [
                    {k: v.cpu() for k, v in out.items()}
                    for out in outputs
                ]


                targets = [
                    {k: v.cpu() for k, v in tgt.items()}
                    for tgt in targets
                ]


                metric.update(outputs, targets)



                if i in [1, 2, 4, 5]:

                    img = (images[0].cpu() * 255).to(torch.uint8)


                    boxes = outputs[0]["boxes"].cpu()
                    labels = outputs[0]["labels"].cpu()
                    scores = outputs[0]["scores"].cpu()


                    keep = scores > 0.5


                    box_labels = [
                        f"{self.val_dataset.idx_to_class[l.item()]} {s:.2f}"
                        for l, s in zip(labels[keep], scores[keep])
                    ]


                    img_with_boxes = draw_bounding_boxes(
                        img,
                        boxes[keep],
                        labels=box_labels,
                        width=2,
                    )


                    
                    try:    
                        run.log(
                            {
                                "epoch": epoch + 1,
                                f"val/epoch/predictions{i}": wandb.Image(
                                    to_pil_image(img_with_boxes),
                                    caption=f"epoch={epoch + 1}, batch_index={i}"
                                ),
                            }
                        )

                    except FileNotFoundError as e:
                        print(f"[WARN] Skipping wandb image log at epoch={epoch}, i={i}: {e}")

        results = metric.compute()
        
        print(results)

        return results






    def train(self):
        best_map = 0.0
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        runname = f"train_{self.model_name}_pretrained_{self.dataset_name}_lr{self.lr}_no_augs_{timestamp}"

        run = wandb.init(
            project="detection-yolo",
            name=runname,
            config={
                "model_name": self.model_name,
                "dataset_name": self.dataset_name,
                "lr": self.lr,
                "epochs": self.epochs,
                "batchsize": self.batchsize,
                "augs": self.augs,
                "trainable_backbone_layers": self.trainable_backbone_layers,
                "weighted_sampling": self.weighted_sampling,
                "pretrained": self.pretrained,
                "warmup": self.warump_epochs,
                "momentum": self.momentum,
                "weight_decay": self.weight_decay,
                "device": str(self.device),
            }
        )

        run.define_metric("epoch", hidden=True)
        run.define_metric("train/batch_step", hidden=True)

        run.define_metric("train/batch/*", step_metric="train/batch_step")
        run.define_metric("train/epoch/*", step_metric="epoch")
        run.define_metric("val/epoch/*", step_metric="epoch")


        params = [p for p in self.model.parameters() if p.requires_grad]
     
        

        optimizer = torch.optim.SGD(
        params,
        lr=self.lr,
        momentum=0.9,
        weight_decay=0.0005
        )



        lr_scheduler = None

        if self.warump_epochs > 0:

       
            scheduler_warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1e-3,      # 0.00001 -> 0.01 over 1 epoch
                end_factor=1.0,
                total_iters=self.warmup_epochs,
            )

            scheduler_main = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.epochs - self.warmup_epochs,
            )

            lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[scheduler_warmup, scheduler_main],
                milestones=[self.warmup_epochs],
            )
        else:
             lr_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=3,
                gamma=0.1
                )    

        train_loader,val_loader = self.build_training_data()
        eval_metric = MeanAveragePrecision()


        self._global_step = 1 


        for epoch in range(self.epochs):
            print(f"\n=== Epoch {epoch + 1}/{self.epochs} ===")
            self.train_one_epoch(data_loader=train_loader, optimizer= optimizer, epoch_id= epoch, run=run)
            lr_scheduler.step()
            results = self.eval(data_loader=val_loader, metric =  eval_metric, epoch= epoch, run=run)
            
            val_metrics = {
                "epoch": epoch + 1,
                "val/epoch/mAP": results["map"].item(),
                "val/epoch/mAP@50": results["map_50"].item(),
                "val/epoch/mAR@100": results["mar_100"].item(),
            }

            if "map_75" in results:
                val_metrics["val/epoch/mAP@75"] = results["map_75"].item()
            if "mar_1" in results:
                val_metrics["val/epoch/mAR@1"] = results["mar_1"].item()
            if "mar_10" in results:
                val_metrics["val/epoch/mAR@10"] = results["mar_10"].item()

            run.log(val_metrics)


            current_map = results["map"].item()


            if current_map > best_map:
                best_map = current_map
                checkpoint_path = RCNN_CHECKPOINTS_DIR / f"{runname}.pt"
                torch.save(
                    self.model.state_dict(),
                    checkpoint_path
                )

                best_artifact = wandb.Artifact(
                    name=f"{self.model_name}-best-model",
                    type="model",
                    metadata={
                        "model_name": self.model_name,
                        "dataset_name": self.dataset_name,
                        "best_map": best_map,
                        "epoch": epoch + 1,
                        "timestamp": timestamp,
                    },
                )
                best_artifact.add_file(str(checkpoint_path))
                run.log_artifact(best_artifact)

                run.summary["best_map"] = best_map
                run.summary["best_epoch"] = epoch + 1
                run.summary["best_model_path"] = str(checkpoint_path)


        wandb.finish()






if __name__ == "__main__":
    RcnnDetectorInsRun1 = RcnnDetector(model_name = 'fasterrcnn_resnet50_fpn', 
                                dataset_name = 'military', 
                                  lr=0.005,
                                  epochs=50, 
                                  batchsize =2,
                                  num_workers=1, 
                                  augs=[],  
                                  trainable_backbone_layers=3, pretrained=True, weighted_sampling = True)
    RcnnDetectorInsRun1.train()



    
    
from torchvision.io.image import decode_image
from torchvision.models.detection import fasterrcnn_resnet50_fpn, fasterrcnn_resnet50_fpn_v2, fasterrcnn_mobilenet_v3_large_fpn
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict
from torchvision.utils import draw_bounding_boxes
from torch.utils.tensorboard import SummaryWriter
from military_dataset import MilitaryDataset
from datetime import datetime


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
    def __init__(self, model_name, dataset_name,  lr,epochs, batchsize, augs,  trainable_backbone_layers):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self._model_ref = MODEL_REGISTRY[model_name]
        self._dataset_ref = DATASET_REGISTRY[dataset_name]
        self.lr =  lr
        self.batchsize = batchsize
        self.augs = augs
        self.trainable_backbone_layers = trainable_backbone_layers
        self.epochs = epochs
       
        self.device = torch.device( 
               "cuda"
                if torch.cuda.is_available()
                else "cpu"
                )
        
        self.build_model()
    


    def build_training_data(self):
         self.train_dataset = self.build_dataset('train')
         self.val_dataset = self.build_dataset('val')
         return self.build_data_loader(self.train_dataset), self.build_data_loader(self.val_dataset)



    def build_test_data(self):
         self.test_dataset = self.build_dataset('test')
         return self.build_data_loader(self.test_dataset, False)




    def build_model(self):
        #Build model
        num_classes = self._dataset_ref.get_class_number()
        model = self._model_ref(num_classes = num_classes,trainable_backbone_layers = self.trainable_backbone_layers)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        model.to(self.device)
        self.model = model
       




    def build_dataset(self, split):
            return self._dataset_ref(split = split)
    
    def build_data_loader(self,dataset, shuffle = True):
        return DataLoader(
            dataset,
            batch_size=2,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=True,
            collate_fn=collate_fn
        )



    def train_one_epoch(self, data_loader, optimizer, epoch_id, tb_writer):

        running_losses = defaultdict(float)
        epoch_losses = defaultdict(float)


        self.model.train()

        for i, (images, targets) in enumerate(tqdm(data_loader)):

            optimizer.zero_grad()

            images = [img.to(self.device) for img in images]
            targets = [
                {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in t.items()
                }
                for t in targets
            ]

            loss_dict = self.model(images, targets)
            loss = sum(loss_dict.values())

            # accumulate interval losses
            for k, v in loss_dict.items():
                running_losses[k] += v.item()
                epoch_losses[k] += v.item()

            running_losses["loss"] += loss.item()
            epoch_losses["loss"] += loss.item()

            loss.backward()
            optimizer.step()

            # report every 1000 batches
            if (i + 1) % 1000 == 0:

                avg_losses = {
                    k: v / 1000
                    for k, v in running_losses.items()
                }

                loss_str = " | ".join(
                    f"{k}: {v:.4f}"
                    for k, v in avg_losses.items()
                )

                print(f"[batch {i+1}] | {loss_str}")

               

                tb_writer.add_scalars(f"train/batch/combo", avg_losses, global_step= self._global_step)
                
                for k,v in avg_losses.items():
                    tb_writer.add_scalar(f"train/batch/{k}", v, global_step= self._global_step)
                

                running_losses.clear()
                self._global_step+=1

        # epoch summary
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

        tb_writer.add_scalars(f"train/epoch/combo", epoch_avg_losses, global_step= epoch_id + 1)
        for k,v in epoch_avg_losses.items():
            tb_writer.add_scalar(f"train/epoch/{k}", v, global_step= epoch_id + 1)
        


        return epoch_avg_losses["loss"]



    def eval(self,data_loader, metric, tb_writer, epoch):
        visualized = False
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


                if not visualized:

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

                    tb_writer.add_image(
                        "val/epoch/predictions",
                        img_with_boxes,
                        epoch,
                    )

                    visualized = True


        results = metric.compute()
        

        print(results)

        return results





    def train(self):

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        tb_writer = SummaryWriter(
              f"runs/{self.model_name}_{timestamp}"
                )
         

        #Construct an optimizer
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(
        params,
        lr=self.lr,
        momentum=0.9,
        weight_decay=0.0005
        )

        #Build learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=3,
        gamma=0.1
        )
        

        train_loader,val_loader = self.build_training_data()
        eval_metric =  MeanAveragePrecision()

        self._global_step = 1 

        for epoch in range(self.epochs):
            print(f"\n=== Epoch {epoch + 1}/{self.epochs} ===")
            self.train_one_epoch(data_loader=train_loader, optimizer= optimizer, epoch_id= epoch, tb_writer=tb_writer)
            lr_scheduler.step()
            results = self.eval(data_loader=val_loader, metric =  eval_metric, epoch= epoch, tb_writer=tb_writer)
            
        
            tb_writer.add_scalar(
                        "val/epoch/mAP",
                        results["map"].item(),
                        epoch,
                    )

            tb_writer.add_scalar(
                "val/epoch/mAP@50",
                results["map_50"].item(),
                epoch,
            )

            tb_writer.add_scalar(
                "val/epoch/mAR@100",
                results["mar_100"].item(),
                epoch,
            )

            tb_writer.add_scalars(
                "val-epoch-combo",
                {
                    "mAP": results["map"].item(),
                    "mAP@50": results["map_50"].item(),
                    "mAP@75": results["map_75"].item(),
                },
                epoch,
            )


            current_map = results["map"].item()

            if current_map > best_map:
                best_map = current_map
                torch.save(
                    self.model.state_dict(),
                    "best_model.pt"
                )
            





if __name__ == "__main__":
    RcnnDetector = RcnnDetector(model_name = 'fasterrcnn_resnet50_fpn', dataset_name = 'military',  lr=0.003,epochs=3, batchsize =22, augs=[],  trainable_backbone_layers=3)

    RcnnDetector.train()
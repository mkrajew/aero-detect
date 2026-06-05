from pathlib import Path
import torch
from torch.utils.data import Subset, DataLoader
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from military_dataset import MilitaryDataset


def collate_fn(batch):
    return tuple(zip(*batch))


def save_image(tensor_uint8, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    to_pil_image(tensor_uint8).save(path)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    img_size = 800
    sample_idx = 1
    num_steps = 300
    lr = 0.001
    score_thresh = 0.05

    dataset = MilitaryDataset(split="train", img_size=img_size, augment=False)
    subset = Subset(dataset, [sample_idx])

    loader = DataLoader(
        subset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    images, targets = next(iter(loader))

    images = [img.to(device) for img in images]
    targets = [{k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

    # force toy binary task: object vs background
    for t in targets:
        t["labels"] = torch.ones_like(t["labels"])

    targets = [
        {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()}
        for t in targets
    ]

    print("\n=== INPUT CHECK ===")
    print("image shape:", images[0].shape)
    print("image dtype:", images[0].dtype)
    print("boxes shape:", targets[0]["boxes"].shape)
    print("boxes:", targets[0]["boxes"])
    print("labels:", targets[0]["labels"])
    print("num boxes:", len(targets[0]["boxes"]))

    if len(targets[0]["boxes"]) == 0:
        raise ValueError("Selected image has no boxes. Pick another sample_idx.")

    boxes = targets[0]["boxes"]
    h, w = images[0].shape[-2:]
    if not ((boxes[:, 0] < boxes[:, 2]).all() and (boxes[:, 1] < boxes[:, 3]).all()):
        raise ValueError("Invalid boxes: xmin/xmax or ymin/ymax are broken.")
    if not ((boxes[:, 0] >= 0).all() and (boxes[:, 2] <= w).all() and (boxes[:, 1] >= 0).all() and (boxes[:, 3] <= h).all()):
        raise ValueError("Boxes go out of image bounds after transforms.")

    model = fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    model.to(device)

    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        momentum=0.9,
        weight_decay=0.0005,
    )

    out_dir = Path("debug_outputs")
    out_dir.mkdir(exist_ok=True)

    gt_img = (images[0].detach().cpu() * 255).to(torch.uint8)
    gt_boxes = targets[0]["boxes"].detach().cpu()
    gt_labels = [f"gt_obj_{i}" for i in range(len(gt_boxes))]
    gt_vis = draw_bounding_boxes(gt_img, gt_boxes, labels=gt_labels, colors="green", width=3)
    save_image(gt_vis, out_dir / "gt.png")

    print("\n=== TRAINING ===")
    for step in range(1, num_steps + 1):
        model.train()

        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step <= 20 or step % 25 == 0:
            loss_items = {k: round(v.item(), 4) for k, v in loss_dict.items()}
            print(f"step {step:03d} | total={loss.item():.4f} | {loss_items}")

    print("\n=== EVAL ===")
    model.eval()
    metric = MeanAveragePrecision()

    with torch.no_grad():
        outputs = model(images)

    outputs_cpu = [{k: v.detach().cpu() for k, v in out.items()} for out in outputs]
    targets_cpu = [{k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

    metric.update(outputs_cpu, targets_cpu)
    results = metric.compute()

    print("\n=== METRICS ===")
    print(results)

    pred_boxes = outputs_cpu[0]["boxes"]
    pred_scores = outputs_cpu[0]["scores"]
    pred_labels = outputs_cpu[0]["labels"]

    print("\n=== PREDICTIONS TOP 20 ===")
    for i in range(min(20, len(pred_scores))):
        print(
            f"{i:02d} | score={pred_scores[i].item():.4f} "
            f"| label={pred_labels[i].item()} "
            f"| box={pred_boxes[i].tolist()}"
        )

    keep = pred_scores > score_thresh
    pred_vis_labels = [
        f"obj {s:.2f}" for s in pred_scores[keep]
    ]

    pred_vis = draw_bounding_boxes(
        gt_img,
        pred_boxes[keep],
        labels=pred_vis_labels,
        colors="red",
        width=3,
    )
    save_image(pred_vis, out_dir / "pred_only.png")

    combined = draw_bounding_boxes(
        gt_img,
        gt_boxes,
        labels=gt_labels,
        colors="green",
        width=3,
    )
    combined = draw_bounding_boxes(
        combined,
        pred_boxes[keep],
        labels=pred_vis_labels,
        colors="red",
        width=2,
    )
    save_image(combined, out_dir / "gt_vs_pred.png")

    print("\nSaved:")
    print(out_dir / "gt.png")
    print(out_dir / "pred_only.png")
    print(out_dir / "gt_vs_pred.png")


if __name__ == "__main__":
    main()
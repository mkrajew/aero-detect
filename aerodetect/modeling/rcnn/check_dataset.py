import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from aerodetect.modeling.rcnn.military_dataset import MilitaryDataset
from aerodetect.modeling.rcnn.skyfusion_dataset import SkyFusionDataset


def save_visualization(
    dataset,
    idx,
    out_dir,
):
    image, target = dataset[idx]

    image = image.permute(
        1,
        2,
        0,
    ).cpu().numpy()

    fig, ax = plt.subplots(
        figsize=(10, 10)
    )

    ax.imshow(image)

    boxes = target["boxes"]
    labels = target["labels"]

    for box, label in zip(
        boxes,
        labels,
    ):
        x1, y1, x2, y2 = box.tolist()

        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            linewidth=2,
        )

        ax.add_patch(rect)

        class_name = dataset.idx_to_class[
            label.item()
        ]

        ax.text(
            x1,
            y1,
            class_name,
        )

    ax.set_axis_off()

    output_file = (
        out_dir
        / f"sample_{idx:06d}.jpg"
    )

    plt.savefig(
        output_file,
        bbox_inches="tight",
        pad_inches=0,
    )

    plt.close()

    print(
        f"Saved {output_file}"
    )


def print_stats(dataset):
    total_boxes = 0

    class_counts = {}

    for i in range(len(dataset)):
        _, target = dataset[i]

        for label in target[
            "labels"
        ].tolist():
            total_boxes += 1

            name = dataset.idx_to_class[
                label
            ]

            class_counts[name] = (
                class_counts.get(name, 0)
                + 1
            )

    print()
    print(
        f"Images: {len(dataset)}"
    )
    print(
        f"Objects: {total_boxes}"
    )

    print("\nClass counts:")

    for cls, count in sorted(
        class_counts.items(),
        key=lambda x: x[1],
        reverse=True,
    ):
        print(
            f"{cls:<20} {count}"
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        default="train",
    )

    parser.add_argument(
        "--idx",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--augment",
        action="store_true",
    )

    parser.add_argument(
        "--save",
        type=int,
        default=1,
        help="number of images to save",
    )

    parser.add_argument(
        "--stats",
        action="store_true",
    )

    parser.add_argument(
        "--out",
        default="debug_dataset",
    )

    args = parser.parse_args()

    dataset = MilitaryDataset(
        split=args.split,
        augment=args.augment,
    )

    print(
        f"Dataset size: {len(dataset)}"
    )
    print(
        f"Classes: {dataset.get_class_number()}"
    )

    if args.stats:
        print_stats(dataset)
        return

    out_dir = Path(args.out)
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.idx is not None:
        indices = [args.idx]
    else:
        count = min(
            args.save,
            len(dataset),
        )

        indices = random.sample(
            range(len(dataset)),
            count,
        )

    for idx in indices:
        save_visualization(
            dataset,
            idx,
            out_dir,
        )


if __name__ == "__main__":
    main()
import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 with a YOLO-format dataset.")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to the YOLO-format data.yaml / dataset.yaml file",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="yolov8n.pt",
        help="Pretrained model or path to weights (default: yolov8n.pt)",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs (default: 100)")
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Train/inference image size; must match chip size (default: 640)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help='Device, e.g. "cuda:0", "cpu" (default: cuda:0)',
    )
    parser.add_argument(
        "--degrees",
        type=float,
        default=180.0,
        help="Rotation augmentation +/- degrees (default: 180)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/detect",
        help="Ultralytics project dir for run outputs (default: runs/detect)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="windmill_run",
        help=(
            "Run name under project/ (default: windmill_run). "
            "If the folder exists, Ultralytics creates windmill_run2, etc. "
            "(exist_ok=False)."
        ),
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Optional path to an image for inference after training",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="If set, export the trained model to ONNX format",
    )
    args = parser.parse_args()

    model = YOLO(args.weights)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        device=args.device,
        degrees=args.degrees,
        project=args.project,
        name=args.name,
        exist_ok=False,
    )

    # Ultralytics may bump the run name (windmill_run → windmill_run3, …).
    save_dir = Path(getattr(results, "save_dir", None) or model.trainer.save_dir)
    best_pt = (save_dir / "weights" / "best.pt").resolve()
    last_pt = (save_dir / "weights" / "last.pt").resolve()
    print(f"BEST_WEIGHTS={best_pt}")
    if last_pt.is_file():
        print(f"LAST_WEIGHTS={last_pt}")

    model.val()

    if args.image:
        results = model(args.image)
        results[0].show()

    if args.export:
        export_path = model.export(format="onnx")
        print(f"Model exported to: {export_path}")


if __name__ == "__main__":
    main()

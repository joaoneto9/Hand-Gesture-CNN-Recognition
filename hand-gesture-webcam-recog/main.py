import os
from pathlib import Path
import time

import cv2
import torch
from torchvision import models, transforms

NEURAL_NETWORK_MAP = {
    "1": "baseline",
    "2": "transfer-learning",
    "3": "fine-tuning",
}

INFERENCE_INTERVAL_SECONDS = 1.15
GUESS_CONFIDENCE_THRESHOLD = 0.60
SNAPSHOT_SIZE = 180


def setup_display() -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

    if not os.environ.get("DISPLAY"):
        print("Variável DISPLAY não configurada; rode a partir da sessão gráfica (ex.: GNOME).")
        return

    if os.environ.get("XAUTHORITY"):
        return

    for entry in Path("/proc").iterdir():
        cmdline = entry / "cmdline"
        if not cmdline.is_file():
            continue
        try:
            args = cmdline.read_bytes().split(b"\0")
        except OSError:
            continue
        if not args:
            continue
        name = os.path.basename(args[0].decode(errors="ignore"))
        if name not in ("Xorg", "Xwayland", "Xvfb"):
            continue
        for arg in args:
            if arg.startswith(b"-auth"):
                index = args.index(arg)
                if index + 1 < len(args) and args[index + 1]:
                    os.environ["XAUTHORITY"] = args[index + 1].decode()
                    print(f"XAUTHORITY detectado: {os.environ['XAUTHORITY']}")
                    return


def init_app() -> str:
    input_text = """
    Escolha uma das redes para usar a web-cam:

    1) baseline
    2) transfer learning
    3) fine-tuning

    ---
    """

    print(input_text)
    choice = input("Escolha: ")

    if choice not in NEURAL_NETWORK_MAP:
        print("Escolha uma Opção válida.")
        return init_app()

    return NEURAL_NETWORK_MAP[choice]


def get_model_path(network_name: str) -> Path:
    directory = Path(f"neural-networks/{network_name}")
    return next(directory.glob("*.pth"))


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(pth_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    checkpoint = torch.load(pth_path, map_location=device, weights_only=False)

    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(checkpoint["class_names"]))

    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    return model.to(device), checkpoint


def build_transform(image_size: int, mean: list[float], std: list[float]) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            transforms.Resize((image_size, image_size)),
        ]
    )


def preprocess(frame, transform: transforms.Compose, device: torch.device) -> torch.Tensor:
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = transform(frame_rgb).unsqueeze(0)
    return tensor.to(device)


@torch.inference_mode()
def infer(
    model: torch.nn.Module,
    frame,
    device: torch.device,
    checkpoint: dict,
) -> tuple[str, float]:
    transform = build_transform(
        checkpoint["image_size"],
        checkpoint["mean"],
        checkpoint["std"],
    )

    tensor = preprocess(frame, transform, device)

    probabilities = torch.softmax(model(tensor), dim=1)
    confidence, index = torch.max(probabilities, dim=1)

    class_names = checkpoint["class_names"]
    return class_names[index.item()], confidence.item()


def run_webcam(model: torch.nn.Module, device: torch.device, checkpoint: dict) -> None:
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    capture.set(cv2.CAP_PROP_FPS, 60)

    if not capture.isOpened():
        print("Não foi possível abrir a webcam.")
        return

    label_text = "Aguardando..."
    last_snapshot = None
    last_inference_time = 0.0

    while True:
        ok, frame = capture.read()

        if not ok:
            print("Falha ao capturar frame da webcam.")
            break

        now = time.monotonic()

        if now - last_inference_time >= INFERENCE_INTERVAL_SECONDS:
            class_name, confidence = infer(model, frame, device, checkpoint)
            last_snapshot = frame.copy()

            if confidence >= GUESS_CONFIDENCE_THRESHOLD:
                label_text = f"{class_name} {confidence:.0%}"
            else:
                label_text = ""

            last_inference_time = now

        overlay = frame

        if last_snapshot is not None:
            snapshot_h, snapshot_w = last_snapshot.shape[:2]
            scale = SNAPSHOT_SIZE / max(snapshot_h, snapshot_w)
            thumb_w = int(snapshot_w * scale)
            thumb_h = int(snapshot_h * scale)

            thumbnail = cv2.resize(last_snapshot, (thumb_w, thumb_h))

            x = overlay.shape[1] - thumb_w - 12
            y = overlay.shape[0] - thumb_h - 40
            overlay[y:y + thumb_h, x:x + thumb_w] = thumbnail

            bar_y = y + thumb_h + 4
            overlay[bar_y:bar_y + 28, x:x + thumb_w] = (0, 0, 0)
            cv2.putText(
                overlay,
                label_text,
                (x + 6, bar_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow("Hand Gesture Recognition", overlay)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    capture.release()
    cv2.destroyAllWindows()


def main() -> None:
    setup_display()
    network_name = init_app()

    device = get_device()
    model, checkpoint = load_model(get_model_path(network_name), device)

    print(f"Usando rede '{network_name}' no dispositivo: {device}")
    print("Pressione 'q' para sair da webcam.")

    run_webcam(model, device, checkpoint)


if __name__ == "__main__":
    main()
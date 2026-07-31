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

INFERENCE_INTERVAL_SECONDS = 1.0


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

    if not capture.isOpened():
        print("Não foi possível abrir a webcam.")
        return

    label_text = "Aguardando..."
    last_inference_time = 0.0

    while True:
        ok, frame = capture.read()

        if not ok:
            print("Falha ao capturar frame da webcam.")
            break

        now = time.monotonic()

        if now - last_inference_time >= INFERENCE_INTERVAL_SECONDS:
            class_name, confidence = infer(model, frame, device, checkpoint)
            label_text = f"{class_name} {confidence:.0%}"
            last_inference_time = now

        overlay = cv2.putText(
            frame,
            label_text,
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3,
            cv2.LINE_AA,
        )

        cv2.imshow("Hand Gesture Recognition", overlay)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    capture.release()
    cv2.destroyAllWindows()


def main() -> None:
    network_name = init_app()

    device = get_device()
    model, checkpoint = load_model(get_model_path(network_name), device)

    print(f"Usando rede '{network_name}' no dispositivo: {device}")
    print("Pressione 'q' para sair da webcam.")

    run_webcam(model, device, checkpoint)


if __name__ == "__main__":
    main()

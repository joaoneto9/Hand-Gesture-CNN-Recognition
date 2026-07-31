from pathlib import Path
from typing import final


neural_network_map = {
    "1": "baseline",
    "2": "transfer-learning",
    "3": "fine-tuning"
}

def init_app() -> str:
    input_text = """
    Escolha uma das redes para usar a web-cam:

    1) baseline
    2) transfer learning
    3) fine-tuning

    --- 
    """

    neural_network_map = {
        "1": "baseline",
        "2": "transfer-learning",
        "3": "fine-tuning"
    }

    print(input_text)
    choice = input("Escolha: ")

    if not neural_network_map.get(choice):
        print("Escolha uma Opção válida.")
        init_app()

    return neural_network_map[choice]


final_path: str = init_app()

path = Path(f"neural-networks/{final_path}")
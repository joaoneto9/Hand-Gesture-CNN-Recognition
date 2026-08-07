# Hand-Gesture-CNN-Recognition

Ferramenta de reconhecimento de gestos de mão em tempo real usando Redes Neurais Convolucionais (CNN). A aplicação captura frames da webcam via OpenCV, processa cada frame com um modelo ResNet18 e exibe o gesto detectado junto com o nível de confiança da predição.

## Estrutura

```
.
├── README.md
├── hand-gesture-webcam-recog/          # Aplicação principal
│   ├── main.py                         # Fluxo da aplicação (webcam + inferência)
│   ├── Dockerfile                      # Definição da imagem da aplicação
│   ├── docker-compose.yml              # Orquestração do serviço (webcam, display, GPU)
│   ├── pyproject.toml                  # Dependências e metadados do projeto
│   ├── uv.lock                        # Versões travadas das dependências
│   ├── .python-version                # Versão do Python utilizada
│   ├── .dockerignore                   # Arquivos ignorados no build da imagem
│   └── neural-networks/                # Checkpoints dos modelos treinados
│       ├── baseline/                   # modelo_baseline.pth
│       ├── transfer-learning/          # modelo_transfer_learning.pth
│       └── fine-tuning/                # modelo_finetuning.pth
└── neural-networks/                    # Diretório reservado na raiz (vazio)
```

## Funcionamento

O fluxo da aplicação (`main.py`) é:

1. **Seleção da rede**: ao iniciar, o usuário escolhe qual modelo usar entre `1` (baseline), `2` (transfer learning) e `3` (fine-tuning). Essa escolha é mapeada pelo dicionário `NEURAL_NETWORK_MAP` (`main.py:9`).

2. **Localização do checkpoint**: `get_model_path()` procura o arquivo `.pth` dentro do diretório correspondente da pasta `neural-networks` (`main.py:73`).

3. **Carregamento do modelo**: `load_model()` instancia uma ResNet18 com a camada final substituída por uma `Linear` com o número de classes do checkpoint (`main.py:82`). O modelo é colocado em modo de inferência (`eval`).

4. **Captura e inferência**: a webcam é aberta com OpenCV (`cv2.VideoCapture(0)`). Os frames são pré-processados (conversão RGB + normalização + resize) e, a cada `INFERENCE_INTERVAL_SECONDS` (~1.15s), a inferência é executada (`main.py:156`).

5. **Exibição de resultado**: o gesto é exibido apenas quando a confiança da predição é maior ou igual ao limiar `GUESS_CONFIDENCE_THRESHOLD` (0.60) (`main.py:160`). Pressione `q` para sair.

## Como Utilizar

Esta ferramenta roda em um contêiner Docker. Pré-requisitos:

- Docker e Docker Compose instalados;
- Sessão gráfica ativa (ex.: GNOME) com o dispositivo de vídeo (`/dev/video0`) disponível;
- Variáveis de ambiente `DISPLAY` e `XAUTHORITY` configuradas (usadas pelo `docker-compose.yml`).

### Rodando a ferramenta

A partir do diretório `hand-gesture-webcam-recog`, execute:

```bash
docker compose run hand-gesture
```

O prompt interativo será aberto. Digite o número da rede desejada (`1`, `2` ou `3`) e confirme. Uma janela da webcam será exibida com o reconhecimento em tempo real; pressione `q` para fechar.

### Rebuild após alterações

Caso realize qualquer alteração em arquivos do código (ex.: `main.py`, `Dockerfile`, `pyproject.toml`), é necessário reconstruir a imagem para aplicar as mudanças:

```bash
docker compose build
```

Depois de buildar, basta rodar novamente:

```bash
docker compose run hand-gesture
```
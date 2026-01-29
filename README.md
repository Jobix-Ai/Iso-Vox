# Iso-Vox

A real-time speech-to-text system with speaker verification, smart turn detection, and multiple ASR backends. We are open-sourcing our Jobix STT system. Contributions are welcome!

For an overview of the system architecture, see [architecture.md](architecture.md).

Demo with background noise and background voice
https://github.com/user-attachments/assets/45196e2e-b965-4593-afdf-a26727abd382

## Prerequisites

- Python 3.11+
- Docker with GPU support (nvidia-docker)
- CUDA-compatible GPU

## Services Overview

| Service              | Port  | Description                           |
|----------------------|-------|---------------------------------------|
| Gateway              | 8089  | Main WebSocket entry point            |
| ASR Batching Worker  | 8090  | Batches audio for efficient processing|
| Smart Turn Worker    | 8091  | Detects conversation turn boundaries  |
| Speaker Verification | 8092  | Verifies speaker identity             |
| GPU Workers          | 8093+ | ASR inference (Parakeet, Moonshine)   |

## Local Development

### 1. Setup Environment

### Installation

Clone this repo then install dependencies. Note, install git-lfs first to download some model checkpoints from github

```bash
sudo apt install -y python3-dev build-essential git-lfs

git clone https://github.com/Jobix-Ai/Iso-Vox.git
cd Iso-Vox

git lfs install
git lfs pull
```

```bash
pip install -r requirements.txt

cd src/moonshine-onnx && pip3 install . && cd ../..

pip uninstall -y onnxruntime

pip install -U onnxruntime-gpu
```

### 3. Start Services
Copy `env.example` to `.env` and configure as needed:

```bash
cp env.example .env
```

Start the services in different terminals

```bash
python src/asr_parakeet_worker_app.py
python src/asr_moonshine_worker_app.py
python src/asr_batching_worker_app.py
python src/speaker_verification_app.py
python src/smart_turn_worker_app.py
```

Start the main gateway (runs on port 8089):

```bash
python src/gateway_app.py
```

### Test STT from client
Open the `examples/client_demo.html` with a web browser then click on `Start Chatting`. Or simulate how the STT handle the call with 
```bash
python examples/simulate_call.py examples/test.wav
```
## Docker Deployment

### 1. Setup Environment

Copy `env.example` to `.env` and configure as needed:

```bash
cp env.example .env
```

### 2. Build the Docker Image

```bash
docker-compose build
```

### 3. Start Services

```bash
docker-compose up -d
```

View logs:

```bash
docker-compose logs -f
```

### Alternative: Run Without Docker Compose

```bash
# Build the image
docker build -t jobix-pipeline-stt:latest .

# Run with .env file
docker run -d \
  --name jobix-pipeline-stt \
  --gpus all \
  --restart unless-stopped \
  --env-file .env \
  -p 8089:8089 \
  -p 8090:8090 \
  -p 8091:8091 \
  -p 8092:8092 \
  -p 8093:8093 \
  -p 8094:8094 \
  -v ./logs:/app/logs \
  -v ./debug_audio:/app/debug_audio \
  jobix-pipeline-stt:latest

# Check logs
docker logs -f jobix-pipeline-stt
```

## Limitations

While Iso-Vox provides real-time ASR with speaker verification for isolating a target speaker, the system has known limitations that affect accuracy in certain scenarios. We document these transparently to help users understand when the system may underperform. Contributions and ideas for improvement are welcome!

| Limitation | Description |
|------------|-------------|
| **Overlapping Speech** | When the main speaker starts in the middle of background speech (TV, other person), the speaker embedding gets contaminated, leading to verification failures |
| **Enrollment Quality** | Our verification logic heavily depends on quality of enrolled speeches. If the enrolled speeches have low quality or locked to wrong speaker, the STT system will fail. |
| **Single Speaker Assumption** | System assumes single primary speaker; no multi-speaker diarization |
| **Enhancement Artifacts** | Speech enhancement may introduce artifacts that affect downstream ASR accuracy |
| **Stateless** | Current logic uses the same thresholds for all users, and ASR models also work independently without having access to conversation context |


### Roadmap
- [X] **Target Speaker Tracking**: The STT be able to transcribe speech of user speaker, ignore other voices.
- [ ] **Dynamic Threshold Calibration**: Adapt per-speaker thresholds based on intra-exemplar similarity, speaking style, acoustic characteristics.
- [ ] **Real-Time Source Separation**: Integrate a lightweight source separation e.g. [TIGER](https://github.com/JusperLee/TIGER/tree/main) or [MossFormer2](https://github.com/alibabasglab/MossFormer2).
- [ ] **Context Awareness**: Make the speaker verifier or ASR aware of conversation context

## Acknowledgements

This STT system uses the following projects:

- [NVIDIA TitaNet](https://huggingface.co/nvidia/speakerverification_en_titanet_large) - Speaker verification
- [NVIDIA Parakeet](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) - ASR model
- [Smart Turn](https://github.com/pipecat-ai/smart-turn) - Turn detection
- [Moonshine](https://github.com/moonshine-ai/moonshine) - Lightweight ASR
- [MPSENet](https://github.com/JacobLinCool/MPSENet) - Speech enhancement
- [GTCRN](https://github.com/Xiaobin-Rong/gtcrn) - Noise reduction

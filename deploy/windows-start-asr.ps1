$Distro = "archlinux"
$Command = "cd /home/fragt/services/asr-server && ASR_ADAPTER=qwen ASR_QWEN_BATCH_SIZE=1 ASR_IDLE_UNLOAD_SECONDS=180 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/uvicorn asr_server.main:app --host 0.0.0.0 --port 18080"

wsl.exe -d $Distro -- bash -lc $Command

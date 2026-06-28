$Distro = "archlinux"
$Command = "cd /home/fragt/services/asr-server && ASR_ADAPTER=qwen uv run uvicorn asr_server.main:app --host 0.0.0.0 --port 18080"

wsl.exe -d $Distro -- bash -lc $Command


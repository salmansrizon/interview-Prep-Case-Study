# Single-click deployment (Windows PowerShell 5.1+ / PowerShell 7).
#   .\start.ps1          CPU (default)
#   .\start.ps1 --gpu    + docker-compose.gpu.yml (NVIDIA GPU for ollama)
# Blocked by execution policy?  powershell -ExecutionPolicy Bypass -File .\start.ps1
# Idempotent: re-running keeps unchanged containers, never re-downloads the model, never duplicates the KB.
Set-Location -LiteralPath $PSScriptRoot

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Host "docker not found: install Docker Desktop (WSL2 backend)."; exit 1
}
docker info *> $null
if ($LASTEXITCODE -ne 0) { Write-Host "Docker daemon not reachable: start Docker Desktop and wait until it says 'running'."; exit 1 }
docker compose version *> $null
if ($LASTEXITCODE -ne 0) { Write-Host "'docker compose' (Compose v2 plugin) not found: update Docker Desktop."; exit 1 }

if (-not (Test-Path .env)) { Copy-Item .env.example .env; Write-Host "created .env from .env.example" }

# no -f flags by default, so COMPOSE_FILE in .env (GPU machines) still applies
$files = @()
if ($args -contains '--gpu' -or $args -contains '-gpu') { $files = @('-f', 'docker-compose.yml', '-f', 'docker-compose.gpu.yml') }

Write-Host "starting the stack (first run: image build + ~4.9 GB model pull, can take 10-20 minutes)..."
docker compose @files up -d --build
if ($LASTEXITCODE -ne 0) {
    docker compose @files ps -a
    Write-Host "docker compose up failed. Look at: docker compose logs ollama-init   (or chroma / api)"; exit 1
}

$timeout = 900
$m = Select-String -Path .env -Pattern '^READY_TIMEOUT_S=(\d+)' | Select-Object -Last 1
if ($m) { $timeout = [int]$m.Matches[0].Groups[1].Value }
Write-Host "waiting for API /ready (Ollama has the model + knowledge base indexed), up to ${timeout}s..."
$deadline = (Get-Date).AddSeconds($timeout)
while ($true) {
    docker compose @files exec -T api python -c "import urllib.request as u; u.urlopen('http://localhost:8000/ready', timeout=5)" *> $null
    if ($LASTEXITCODE -eq 0) { break }
    if ((Get-Date) -gt $deadline) {
        docker compose @files ps -a
        Write-Host "API not ready after ${timeout}s. Look at: docker compose logs api   and: curl.exe http://127.0.0.1:8000/ready"; exit 1
    }
    Start-Sleep -Seconds 5
}

$ui = (docker compose @files port ui 8501) | Select-Object -First 1
if (-not $ui) { $ui = '127.0.0.1:8501' }
Write-Host "Ready: http://$ui"

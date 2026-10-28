# Stop the stack: removes containers + network, KEEPS the named volumes (model + vectors), so the next
# .\start.ps1 is fast and offline-safe.
#   docker compose down -v   would ALSO delete ollama_models (~4.9 GB re-download) and chroma_data
#   (re-ingested on next start). Only do that for a deliberate reset.
Set-Location -LiteralPath $PSScriptRoot
docker compose down
exit $LASTEXITCODE

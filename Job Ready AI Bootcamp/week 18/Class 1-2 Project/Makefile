# Thin wrapper (Linux/macOS; Windows users run start.ps1 / stop.ps1 instead).
.PHONY: up gpu down ps logs eval config test

up:
	./start.sh

gpu:
	./start.sh --gpu

down:
	./stop.sh

ps:
	docker compose ps -a

logs:
	docker compose logs -f --tail=100

# exit code 2 = a safety metric failed (unsafe_answers or missed_escalations > 0)
eval:
	docker compose exec api python cli.py eval

# static check, no daemon needed
config:
	docker compose --env-file .env.example -f docker-compose.yml -f docker-compose.gpu.yml config --quiet && echo "compose OK"

test:
	pytest -q

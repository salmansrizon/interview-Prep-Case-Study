"""
Static checks of the deployment files: compose (base + GPU override), Dockerfiles, .dockerignore, .env.example,
single-click scripts and the copied app. No Docker daemon, no network, no image build.

HIGHLIGHTS: এই repo-র machine-এ Docker daemon বা GPU নেই (ADR 0008), তাই stack চালিয়ে দেখা যায় না।
বদলে যে ভুলগুলো `docker compose up`-কে ভাঙে বা নীরবে খারাপ করে, সেগুলো file পড়েই ধরা হয়:
`latest` tag, healthcheck ছাড়া `service_healthy`, container-এর ভেতরে `localhost`, undeclared volume,
Chroma client-server version mismatch, image-এ `.env`, code COPY-এর পরে model bake, root user।
`docker compose config` (CLI আছে, daemon লাগে না) থাকলে শেষ test সেটাও চালায়; না থাকলে skip।
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from fnmatch import fnmatch
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app"
BASE = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
GPU = yaml.safe_load((ROOT / "docker-compose.gpu.yml").read_text(encoding="utf-8"))
SERVICES = BASE["services"]
LONG_RUNNING = [n for n, s in SERVICES.items() if s.get("restart") != "no"]
ENV = dict(line.split("=", 1) for line in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
           if line.strip() and not line.startswith("#"))
_VAR = re.compile(r"\$\{(\w+)(?::([-?])([^}]*))?\}")


def interp(s: str) -> str:
    """Compose-style ${VAR}, ${VAR:-default}, ${VAR:?error} with .env.example as the .env (missing -> '')."""
    return _VAR.sub(lambda m: ENV.get(m[1]) or (m[3] if m[2] == "-" else ""), str(s))


def instructions(dockerfile: str) -> list[tuple[str, str, str]]:
    """(INSTRUCTION, args, raw lines) per Dockerfile instruction, continuation lines joined."""
    out, buf, raw = [], "", ""
    for line in (ROOT / dockerfile).read_text(encoding="utf-8").splitlines():
        if not buf and (not line.strip() or line.lstrip().startswith("#")):
            continue
        raw += line + "\n"
        buf += line.rstrip("\\").strip() + " "
        if not line.endswith("\\"):
            word, _, rest = buf.strip().partition(" ")
            out.append((word.upper(), rest.strip(), raw))
            buf, raw = "", ""
    return out


# ── compose: shape, images, health, order ────────────────────────────
def test_stack_is_four_services_plus_init_with_a_fixed_project_name():
    assert BASE["name"] == "healthcare-assistant"           # volume names survive a folder rename
    assert set(SERVICES) == {"ui", "api", "chroma", "ollama", "ollama-init"}
    assert LONG_RUNNING == ["ollama", "chroma", "api", "ui"] and SERVICES["ollama-init"]["restart"] == "no"


def test_pulled_images_are_pinned_and_fail_loudly_without_env():
    for name, svc in SERVICES.items():
        if "build" in svc:
            continue
        assert ":?" in svc["image"], f"{name}: tag must be ${{VAR:?}}, never a silent default"
        tag = interp(svc["image"]).rsplit(":", 1)[1]
        assert re.fullmatch(r"\d+\.\d+\.\d+", tag), f"{name}: unpinned tag {tag!r}"
    assert SERVICES["ollama"]["image"] == SERVICES["ollama-init"]["image"]


def test_chroma_server_tag_matches_the_pinned_chromadb_client():
    pin = re.search(r"^chromadb==(\S+)", (APP / "requirements-api.txt").read_text(encoding="utf-8"), re.M)[1]
    assert interp(SERVICES["chroma"]["image"]) == f"chromadb/chroma:{pin}"


def test_every_long_running_service_has_a_healthcheck_the_image_can_run():
    tests = {n: " ".join(SERVICES[n]["healthcheck"]["test"]) for n in LONG_RUNNING}
    assert all(t.startswith("CMD") for t in tests.values())
    assert not any(tool in t for t in tests.values() for tool in ("curl", "wget"))   # not in these images
    assert tests["ollama"] == "CMD ollama list"
    assert "bash" in tests["chroma"] and "/api/v2/heartbeat" in tests["chroma"]
    assert "localhost:8000/health" in tests["api"] and "localhost:8501/_stcore/health" in tests["ui"]
    assert SERVICES["api"]["healthcheck"]["start_period"] == "120s"                  # model load + auto-ingest


def test_depends_on_waits_for_health_or_completion():
    got = {(n, dep): c["condition"] for n, s in SERVICES.items() for dep, c in s.get("depends_on", {}).items()}
    assert got == {("ollama-init", "ollama"): "service_healthy",
                   ("api", "chroma"): "service_healthy",
                   ("api", "ollama"): "service_healthy",
                   ("api", "ollama-init"): "service_completed_successfully",
                   ("ui", "api"): "service_healthy"}
    for (_, dep), cond in got.items():
        assert ("healthcheck" in SERVICES[dep]) if cond == "service_healthy" else SERVICES[dep]["restart"] == "no"


def test_init_pulls_only_when_the_model_is_missing():
    cmd = interp(" ".join(SERVICES["ollama-init"]["entrypoint"]))
    assert "ollama show llama3.1:8b" in cmd and "|| ollama pull llama3.1:8b" in cmd   # offline-safe re-run
    assert SERVICES["ollama-init"]["environment"]["OLLAMA_HOST"] == "http://ollama:11434"


# ── compose: network, ports, state, config ───────────────────────────
def test_inter_service_env_uses_service_names_never_localhost():
    for name, svc in SERVICES.items():
        for key, value in (svc.get("environment") or {}).items():
            assert not re.search(r"localhost|127\.0\.0\.1|host\.docker\.internal", interp(value)), (name, key)
    api = SERVICES["api"]["environment"]
    assert (api["HC_OLLAMA_HOST"], api["HC_CHROMA_HOST"], api["HC_CHROMA_PORT"]) == ("http://ollama:11434", "chroma", "8000")
    assert api["HC_BACKEND"] == "real" and api["HC_AUTO_INGEST"] == "true" and api["HC_KB_DIR"] == "/app/data/kb"
    assert SERVICES["ui"]["environment"]["HC_API_URL"] == "http://api:8000"


def test_only_ui_and_api_are_published_and_only_on_loopback():
    published = {n: [interp(p) for p in s["ports"]] for n, s in SERVICES.items() if s.get("ports")}
    assert published == {"api": ["127.0.0.1:8000:8000"], "ui": ["127.0.0.1:8501:8501"]}   # no auth on ollama/chroma


def test_state_lives_in_declared_named_volumes():
    used = {v.split(":")[0] for s in SERVICES.values() for v in s.get("volumes", [])}
    assert used == set(BASE["volumes"]) == {"ollama_models", "chroma_data"}
    assert SERVICES["ollama"]["volumes"] == ["ollama_models:/root/.ollama"]
    assert SERVICES["chroma"]["volumes"] == ["chroma_data:/data"]


def test_env_example_covers_every_variable_and_matches_the_app_defaults():
    from src.settings import Settings
    for m in _VAR.finditer(yaml.safe_dump(BASE)):                                     # values only, not comments
        assert m[1] in ENV, f"{m[1]} missing from .env.example"
        if m[2] == "-":
            assert ENV[m[1]] == m[3], f"{m[1]}: compose default {m[3]!r} != .env.example {ENV[m[1]]!r}"
    for key, value in ENV.items():
        if key.startswith("HC_"):
            default = Settings.model_fields[key[3:].lower()].default
            assert type(default)(value) == default, key
    assert ENV["HC_MODEL"] == "llama3.1:8b"                                            # ADR 0006
    api = SERVICES["api"]
    assert api["build"]["args"]["EMBEDDING_MODEL"] == api["environment"]["HC_EMBEDDING_MODEL"]   # baked == used


def test_gpu_override_only_touches_ollama():
    assert set(GPU) == {"services"} and set(GPU["services"]) == {"ollama"}
    assert set(GPU["services"]["ollama"]) == {"deploy"}
    assert GPU["services"]["ollama"]["deploy"]["resources"]["reservations"]["devices"] == [
        {"driver": "nvidia", "count": "all", "capabilities": ["gpu"]}]
    assert "deploy" not in SERVICES["ollama"]                                          # CPU is the default


# ── images ───────────────────────────────────────────────────────────
def test_dockerignore_keeps_secrets_out_and_the_knowledge_base_in():
    pats = [p.strip() for p in (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
            if p.strip() and not p.startswith("#")]
    assert ".env" in pats and "app/tests/" in pats
    for keep in ("app/data/kb/fever.md", "app/data/eval/questions.jsonl", "app/src/api/main.py", "app/ui.py"):
        assert not any(fnmatch(keep, p.rstrip("/")) or keep.startswith(p.rstrip("/") + "/") for p in pats), keep


@pytest.mark.parametrize("dockerfile", ["Dockerfile.api", "Dockerfile.ui"])
def test_dockerfile_non_root_healthcheck_exec_cmd_and_valid_copies(dockerfile):
    ins = instructions(dockerfile)
    users = [a for i, a, _ in ins if i == "USER"]
    assert users and users[-1] not in ("root", "0")
    assert any(i == "HEALTHCHECK" for i, _, _ in ins)
    assert isinstance(json.loads([a for i, a, _ in ins if i == "CMD"][-1]), list)       # exec form: PID 1 gets SIGTERM
    for i, args, raw in ins:
        assert not re.search(r"\s#", raw), f"inline comment becomes an argument: {raw}"
        if i == "COPY":
            *sources, _dest = [a for a in args.split() if not a.startswith("--")]
            assert all((ROOT / s).exists() for s in sources), args                    # build context paths


def test_api_image_bakes_the_model_before_the_code_on_cpu_torch():
    ins = instructions("Dockerfile.api")
    at = lambda pred: next(n for n, (i, a, _) in enumerate(ins) if pred(i, a))    # noqa: E731
    user = at(lambda i, a: i == "USER")
    bake = at(lambda i, a: i == "RUN" and "SentenceTransformer" in a)
    offline = at(lambda i, a: i == "ENV" and "HF_HUB_OFFLINE=1" in a)
    code = at(lambda i, a: i == "COPY" and "app/src" in a)
    assert user < bake < offline < code
    pip = next(a for i, a, _ in ins if i == "RUN" and "torch" in a)
    assert "download.pytorch.org/whl/cpu" in pip and "torch==2.14.1" in pip


def test_ui_image_installs_only_streamlit_and_httpx():
    reqs = [r.split("==")[0] for r in (APP / "requirements-ui.txt").read_text(encoding="utf-8").splitlines()
            if r and not r.startswith("#")]
    assert reqs == ["streamlit", "httpx"]
    assert not any("torch" in a or "requirements-api" in a for _, a, _ in instructions("Dockerfile.ui"))
    assert "streamlit" not in (APP / "requirements-api.txt").read_text(encoding="utf-8")


# ── single click ─────────────────────────────────────────────────────
@pytest.mark.parametrize("script", ["start.sh", "start.ps1"])
def test_start_script_checks_docker_creates_env_builds_and_waits_for_ready(script):
    t = (ROOT / script).read_text(encoding="utf-8")
    for needle in ("docker info", ".env.example", "up -d --build", "/ready", "READY_TIMEOUT_S",
                   "--gpu", "docker-compose.gpu.yml", "port ui 8501"):
        assert needle in t, f"{script}: missing {needle!r}"


@pytest.mark.parametrize("script", ["stop.sh", "stop.ps1"])
def test_stop_script_keeps_the_volumes(script):
    code = [ln for ln in (ROOT / script).read_text(encoding="utf-8").splitlines() if not ln.lstrip().startswith("#")]
    assert any("docker compose down" in ln for ln in code) and not any("-v" in ln for ln in code)


def test_shell_scripts_have_lf_line_endings():
    for f in ("start.sh", "stop.sh", "Makefile"):
        assert b"\r\n" not in (ROOT / f).read_bytes(), f


# ── self-contained copy (ADR 0002) ───────────────────────────────────
def test_copied_app_files_exist_and_nothing_points_outside_the_project():
    for rel in ("src/api/main.py", "src/assistant.py", "src/settings.py", "src/backend.py", "cli.py", "ui.py",
                ".streamlit/config.toml", "data/eval/questions.jsonl", "samples/make_handbook.py"):
        assert (APP / rel).is_file(), rel
    assert len(list((APP / "data" / "kb").glob("*.md"))) == 30
    for f in ("docker-compose.yml", "docker-compose.gpu.yml", "Dockerfile.api", "Dockerfile.ui", "start.sh",
              "start.ps1", "stop.sh", "stop.ps1", "Makefile"):
        assert not re.search(r"\.\./|week 17[\\/]", (ROOT / f).read_text(encoding="utf-8"), re.I), f


# ── optional: the real Compose parser (CLI only, no daemon) ──────────
def _compose_cli() -> bool:
    if shutil.which("docker") is None:
        return False
    return subprocess.run(["docker", "compose", "version"], capture_output=True).returncode == 0


@pytest.mark.skipif(not _compose_cli(), reason="docker compose CLI not installed")
@pytest.mark.parametrize("files", [["docker-compose.yml"], ["docker-compose.yml", "docker-compose.gpu.yml"]])
def test_docker_compose_config_accepts_the_files(files):
    flags = [x for f in files for x in ("-f", f)]
    r = subprocess.run(["docker", "compose", "--env-file", ".env.example", *flags, "config", "--quiet"],
                       cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr

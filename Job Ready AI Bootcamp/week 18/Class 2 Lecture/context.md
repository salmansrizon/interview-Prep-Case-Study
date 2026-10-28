# সহজ ভাষায় Lecture Overview

এই lecture-এর main topic হলো **1. Topic: GPU Passthrough আর Cloud Deployment — Driver থেকে HTTPS পর্যন্ত**। Technical term English-এ রাখা হয়েছে, আর explanation Bangla-তে দেওয়া হয়েছে—যাতে concept বোঝা, code পড়া এবং interview-তে explain করা তিনটিই সহজ হয়।

## কেন এই Topic দরকার?

Class 35-এ পুরো Healthcare Assistant এক command-এ উঠেছে — কিন্তু CPU-তে। `llama3.1:8b`-এর decode memory-bandwidth-bound (Week 15), আর GPU-র memory bandwidth সাধারণ CPU RAM-এর চেয়ে বহুগুণ বেশি — তাই একই উত্তর GPU-তে অনেক দ্রুত আসে। সমস্যা হলো container-এ GPU "এমনি" যায় না: host-এর driver, NVIDIA Container Toolkit, Compose-এ device request, driver-এর version — যেকোনো একটা ভুল হলে Ollama **কোনো error ছাড়াই** CPU-তে চলে। GPU-র টাকা দিচ্ছেন, কাজ হচ্ছে CPU-তে, কেউ জানে না।

দ্বিতীয় অর্ধেক: laptop-এর stack অন্যদের ব্যবহার করতে দিতে হলে internet-এ একটা server লাগে। তখন একদম নতুন প্রশ্ন — কে ঢুকতে পারবে (এটা health data!), HTTPS কোথা থেকে, secret কোথায় থাকবে, VM reboot হলে stack ফিরবে কিনা, disk log-এ ভরে গেলে কী হবে, নতুন version ভাঙলে কীভাবে ফেরাবেন, আর মাস শেষে bill কত। এই class-এ শিখব GPU কীভাবে container-এ পৌঁছায় আর VM-এ একটা AI stack চালাতে কী কী বদলায়।

## শেখার সহজ Workflow

GPU-র চার স্তর → host driver → Container Toolkit → `--gpus` / Compose device request → GPU override file → driver/CUDA compatibility → VRAM sizing (weights + KV cache) → GPU সত্যিই চলছে কিনা প্রমাণ → CPU fallback, Mac, WSL2 → VM sizing → VM থেকে `docker compose up` → firewall → HTTPS reverse proxy + login → secret → backup → monitoring → tag আর rollback → খরচ → কখন Kubernetes।

প্রতিটি section পড়ার সময় তিনটি প্রশ্ন করুন: **এটি কোন problem solve করে? কীভাবে কাজ করে? Alternative-এর তুলনায় কখন better?**

> **Practical mindset:** "GPU কাজ করছে" বলার আগে দুটো প্রমাণ: container-এর **ভেতরে** `nvidia-smi` চলে, আর `ollama ps`-এ `100% GPU`। প্রমাণ না থাকলে ধরে নিন CPU-তে চলছে।

> **Verification-এর সৎ কথা (ADR 0008, ADR 0010):** এই lecture যে machine-এ লেখা হয়েছে সেখানে NVIDIA GPU, Docker daemon বা cloud account ছিল না। GPU override file শুধু static ভাবে যাচাই করা, চালানো না। VRAM-এর সংখ্যা formula থেকে **estimate**, মাপা না। Cloud অংশ পড়ার material — কোনো deploy script নেই, কোনো আসল দাম নেই। যেখানে command-এর output দেখানো, সেগুলো illustrative।

---

## 1. Topic: GPU Passthrough আর Cloud Deployment

এই ক্লাসে আমরা শিখব:

* **GPU কীভাবে container-এ পৌঁছায়** — host driver, NVIDIA Container Toolkit, `--gpus all`, Compose-এর `deploy.resources.reservations.devices`, আলাদা GPU override file।
* **Compatibility** — driver বনাম image-এর CUDA, Ollama-র ন্যূনতম driver আর compute capability।
* **VRAM sizing** — `llama3.1:8b` Q4 / Q8 / FP16, KV cache আর context, partial offload।
* **CPU fallback, Apple Silicon, Windows/WSL2।**
* **Cloud VM** — sizing, VM থেকে `docker compose up` পর্যন্ত ধাপ, firewall, HTTPS reverse proxy, login, secret, volume backup, monitoring, image tag আর rollback, খরচের হিসাব, Bangladesh-এর প্রেক্ষাপট, আর কখন Compose ছেড়ে managed service বা Kubernetes।

**Project Goal (Shared Project `../Class 1-2 Project/`):** Class 35-এর ৪-service stack-এর সাথে একটা optional `docker-compose.gpu.yml` — NVIDIA GPU থাকলে শুধু `ollama` service-কে GPU দেয়, না থাকলে base file যেমন আছে CPU-তে চলে। Cloud deployment এই lecture-এ theory হিসেবে (ADR 0010): কোনো deploy script নেই, কোনো cloud account ধরে নেওয়া হয়নি।

---

## 2. Why It Is Related

* **Class 35** stack-কে container-এ ভরেছে: service name, healthcheck, init container, volume, single-click। এই ক্লাস একই stack-কে GPU দেয় আর একটা VM-এ তোলার পথ দেখায় — compose file প্রায় একই থাকে, চারপাশ বদলায়।
* **Week 15 Class 1** দেখিয়েছিল latency-র গঠন: prefill (prompt পড়া) আর decode (token লেখা, memory-bandwidth-bound)। GPU দুটোকেই দ্রুত করে, আর quantization কম byte পড়ায় বলে decode আরও দ্রুত।
* **Week 15 Class 2**-এর `/metrics`, burn-rate alert আর GPU-seconds cost proxy — VM-এ এগুলো আসল কাজে লাগে, আর GPU-seconds এখন আসল টাকা।

---

## 3. How It Works

### 3.1 GPU-র চার স্তর

```
Ollama (app)                     <- container
CUDA user-space library          <- container image-এর ভেতরে (Ollama image নিজেই আনে)
NVIDIA Container Toolkit         <- host: container start-এর সময় GPU device আর driver library ঢোকায়
NVIDIA driver (kernel module)    <- host: driver শুধু এখানে, কখনো image-এ না
GPU hardware
```

* **Driver সবসময় host-এ।** Container host-এর kernel ভাগ করে; kernel module image-এ থাকতে পারে না। Image-এ থাকে শুধু CUDA-র user-space library।
* **Container Toolkit সেতু।** Container চালুর সময় সে GPU device file আর host driver-এর library (যেমন `libcuda`, `nvidia-smi`) container-এ mount করে। Toolkit না থাকলে Docker জানেই না GPU কীভাবে দিতে হয়।
* **চাইতে হয়।** Toolkit থাকলেও container GPU পায় শুধু যখন run-এর সময় device request থাকে (`--gpus` বা Compose-এর `devices`)।
* `ollama/ollama` image নিজেই GPU-র জন্য প্রস্তুত — তার Dockerfile-এ `NVIDIA_VISIBLE_DEVICES=all` আর `NVIDIA_DRIVER_CAPABILITIES=compute,utility` বসানো। আমাদের শুধু request দিতে হয়।

### 3.2 Host Setup (Linux VM বা GPU PC) — পড়ার জন্য, script না

1. **Driver:** distribution-এর package manager দিয়ে NVIDIA driver। Host-এ `nvidia-smi` চললে তবেই পরের ধাপ।
2. **Toolkit:** NVIDIA-র apt repository যোগ করে `nvidia-container-toolkit` install (NVIDIA-র install guide-এর exact command দেখুন)।
3. **Docker-কে জানানো:** `sudo nvidia-ctk runtime configure --runtime=docker` (এটা `/etc/docker/daemon.json` বদলায়), তারপর `sudo systemctl restart docker`।
4. **যাচাই:** `docker run --rm --gpus all ubuntu nvidia-smi` — container-এর ভেতর থেকে GPU-র তালিকা আসা উচিত।

NVIDIA-র docs-এর একটা সতর্কবার্তা: systemd cgroup driver ব্যবহার করা system-এ `systemctl daemon-reload` চালালে চলমান container GPU-র access হারাতে পারে — "গতকাল চলছিল, আজ CPU" রহস্যের একটা কারণ। Container restart করলে ফেরে।

### 3.3 GPU চাওয়া: `docker run`, Compose আর Override File

```bash
docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 ollama/ollama   # Ollama docs-এর উদাহরণ
```

Compose-এ একই জিনিস device reservation দিয়ে:

```yaml
# docker-compose.gpu.yml — শুধু override; base file-এর সাথে merge হয়
services:
  ollama:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all            # বা একটা সংখ্যা; অথবা device_ids: ["0"]
              capabilities: [gpu]   # বাধ্যতামূলক — না দিলে deploy-এর সময় error
```

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

* `count` আর `device_ids` একসাথে দেওয়া যায় না; `count` না দিলে বা `all` দিলে host-এর সব GPU। নতুন Compose spec-এ ছোট রূপ `gpus: all`-ও আছে; course long syntax রাখে কারণ পুরোনো Compose-এও চলে আর বেশি স্পষ্ট।
* **কেন আলাদা file?** NVIDIA runtime ছাড়া machine-এ (laptop, Mac, CPU VM) device request সরাসরি ব্যর্থ হয়: `could not select device driver "nvidia" with capabilities: [[gpu]]`। Base file portable থাকে; GPU machine-এ শুধু override যোগ।
* **প্রতিবার `-f -f` মনে রাখা ঝামেলা** — GPU VM-এর `.env`-এ `COMPOSE_FILE=docker-compose.yml:docker-compose.gpu.yml` দিলে সাধারণ `docker compose up -d`-ই দুটো পড়ে (Windows-এ separator `;`)। Single-click script তখনো এক command।
* **শুধু `ollama` GPU পায়।** API-র embedding (MiniLM) CPU-তে কয়েক ms; API-কে GPU দিলে CUDA torch লাগবে — image কয়েক GB ভারী, লাভ প্রায় শূন্য (Class 35-এর size budget)।
* **Merge-এর একটা ফাঁদ:** override file-এ `ports`-এর মতো list **যোগ হয়**, বদলায় না। Base-এর একটা port সরাতে Compose-এর `!reset` tag (`ports: !reset []`) লাগে।

AMD GPU-র জন্য Ollama আলাদা `rocm` tag দেয়, device দিতে হয় `--device /dev/kfd --device /dev/dri` দিয়ে — NVIDIA toolkit সেখানে লাগে না।

### 3.4 Compatibility: Driver, CUDA আর Compute Capability

* **`nvidia-smi`-র "CUDA Version" installed CUDA না** — এটা driver সর্বোচ্চ কোন CUDA version চালাতে পারে। Image-এর ভেতরের CUDA runtime এর সমান বা নিচে হতে হবে। নতুন image + পুরোনো driver = container চলে, কিন্তু GPU init fail → CPU।
* **Ollama-র শর্ত (docs, October 2026):** NVIDIA compute capability 5.0+ আর driver 550+; compute capability 5.0-6.2-এর card-এ driver 570+। এর চেয়ে পুরোনো GPU (যেমন কিছু Kepler card) Ollama ব্যবহার করে না — নীরবে CPU।
* নিজের card দেখুন: `nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv`; card-এর compute capability NVIDIA-র CUDA GPUs তালিকায়।
* **Kernel update-এর ফাঁদ:** VM-এর kernel update হলে driver module নতুন kernel-এর জন্য build না হলে host-এ `nvidia-smi` বলে "couldn't communicate with the NVIDIA driver"। Unattended upgrade চালু থাকলে reboot-এর পর প্রথমে এটা দেখুন।

### 3.5 VRAM Sizing: Weights + KV Cache + Overhead

```
VRAM ≈ params × bits_per_weight / 8          (weights)
     + 2 × layers × kv_heads × head_dim × bytes × context   (KV cache)
     + overhead                              (CUDA context + compute buffer)
```

`llama3.1:8b`-এর public config: ~8.03B parameter, ৩২ layer, ৮টা KV head (grouped-query attention — ৩২টা query head ৮টা KV head ভাগ করে নেয়, তাই KV cache চার গুণ ছোট), head dim ১২৮। FP16 KV cache-এ প্রতি token = 2 × 32 × 8 × 128 × 2 byte = **১২৮ KiB**।

Notebook-এর estimator-এর ফল (overhead ০.৮ GB ধরে — একটা অনুমান; সব **approximate estimate**, মাপা না):

| Config | Weights | KV @৪k | KV @৮k | KV @৩২k | মোট @৮k (≈) |
|---|---|---|---|---|---|
| `llama3.1:8b` Q4_K_M (Ollama-র default tag) | ≈৪.৯ GB | ০.৫ GB | ১.১ GB | ৪.৩ GB | **≈৬.৭ GB** |
| `llama3.1:8b` Q8_0 | ≈৮.৫ GB | ০.৫ | ১.১ | ৪.৩ | ≈১০.৪ GB |
| `llama3.1:8b` FP16 | ≈১৬.১ GB | ০.৫ | ১.১ | ৪.৩ | ≈১৭.৯ GB |
| `llama3.1:8b` Q4_K_M, KV q8_0 | ≈৪.৯ GB | — | — | ২.১ GB | @৩২k ≈৭.৮ GB |
| `llama3.2:3b` Q4_K_M | ≈১.৯ GB | — | ০.৯ | — | ≈৩.৭ GB |

* Q4_K_M weight-এর estimate (≈৪.৯ GB) Ollama library-তে `llama3.1:8b`-এর download size-এর সাথে মেলে — formula ঠিক পথে।
* **লম্বা context-এর দাম model-এর বাইরে:** ৩২k context-এর KV (৪.৩ GB) প্রায় পুরো Q4 model-এর সমান। RAG-এ top-k × chunk যত বড়, `num_ctx` তত বড়।
* **KV cache quantization:** Ollama-র `OLLAMA_KV_CACHE_TYPE=q8_0` (flash attention চালু থাকলে) KV অর্ধেক করে — লম্বা context-এ দরকারি knob।
* **Context স্পষ্ট দিন।** Ollama-র default context version-ভেদে বদলেছে; `ollama ps`-এর `CONTEXT` column দেখুন, আর app-এ `num_ctx` (বা server-এ `OLLAMA_CONTEXT_LENGTH`) নিজে বসান — নইলে VRAM-এর হিসাব মেলে না।
* **না আঁটলে partial offload:** Ollama কিছু layer CPU-তে রাখে; `ollama ps` দেখায় যেমন `27%/73% CPU/GPU`। চলে, কিন্তু speed CPU-র দিকে নামে। VRAM-এর ~৯০% পর্যন্ত plan করুন।
* **Quantization আর speed:** decode-এ প্রতি token-এ সব weight একবার পড়তে হয় (Week 15) — কম byte মানে দ্রুত। একই GPU-তে Q4 সাধারণত Q8-এর চেয়ে দ্রুত decode করে, সামান্য quality-র বিনিময়ে।

**Model/hardware choice table:**

| Model | Use Case | Advantage | Limitation |
|---|---|---|---|
| `llama3.1:8b` Q4_K_M (chosen, ADR 0006; Ollama default tag) | ৮-১৬ GB GPU, CPU fallback, course-এর সব project | ≈৪.৯ GB weight; সবচেয়ে সাধারণ GPU-তে আঁটে; দ্রুত decode | quantization-এ সামান্য quality ক্ষতি; ৮ GB GPU-তে লম্বা context আঁটে না |
| `llama3.1:8b` Q8_0 | ১২-১৬ GB GPU, quality একটু বেশি চাইলে | FP16-এর খুব কাছের quality | ≈৮.৫ GB; একই GPU-তে decode ধীর |
| `llama3.1:8b` FP16 | ২৪ GB GPU, evaluation-এর reference | পূর্ণ precision | ≈১৬ GB weight; ১৬ GB-এর T4-এ আঁটে না; সবচেয়ে ধীর decode |
| `llama3.2:3b` Q4_K_M | CPU-only VM, ৪-৬ GB GPU, demo | ≈২ GB, দ্রুত, সস্তা VM-এ চলে | medical প্রশ্নে grounding আর instruction following দুর্বল |
| Hosted API | GPU নেই, অনিয়মিত traffic | VRAM শূন্য, scale provider-এর | health data তৃতীয় পক্ষের কাছে; per-token খরচ; Sovereign AI framing ভাঙে |

### 3.6 GPU সত্যিই চলছে কিনা: প্রমাণ

| ধাপ | Command | কী দেখবেন |
|---|---|---|
| Host GPU দেখে | `nvidia-smi` | GPU-র নাম, driver version |
| Container GPU দেখে | `docker compose exec ollama nvidia-smi` | একই GPU; "command not found" = device request নেই |
| Model GPU-তে বসেছে | `docker compose exec ollama ollama ps` | `PROCESSOR`: `100% GPU` |
| Speed | `docker compose exec ollama ollama run llama3.1:8b --verbose` | `eval rate` (tok/s) — CPU-র সাথে তুলনা করুন |
| Startup | `docker compose logs ollama` | GPU detect হওয়ার লাইন, না হলে CPU-তে fallback-এর লাইন |

সবচেয়ে সাধারণ কারণ "GPU VM-এ CPU-তে চলছে"-র: `docker compose up -d` চালানো হয়েছে override ছাড়া। Notebook-এর section 2 এই output-গুলো পড়ে diagnosis দেয়।

### 3.7 CPU Fallback, Apple Silicon আর Windows/WSL2

* **CPU fallback:** override ছাড়া base file CPU-তে চলে — একই stack, একই command, শুধু ধীর। RAM-এ model (Q4 ≈৫-৬ GB) আর সব service-এর জায়গা লাগে। খুব দুর্বল machine-এ `llama3.2:3b` (ADR 0006-এর fallback), quality-র দাম মেনে।
* **Apple Silicon (Mac):** Ollama-র FAQ স্পষ্ট — Docker Desktop-এ macOS-এ GPU passthrough নেই। Ollama container তাই CPU-তে চলে। GPU (Metal) চাইলে host-এ native Ollama app চালান, আর API-কে `HC_OLLAMA_HOST=http://host.docker.internal:11434` দিন (Week 15-এর pattern)। Trade-off: model host-এ আলাদা pull করতে হয়, আর "single click"-এর একটা অংশ হাতে।
* **Windows:** Docker Desktop-এর WSL2 backend NVIDIA GPU সমর্থন করে। Windows-এ সাধারণ NVIDIA driver install করুন (সেটাই WSL2-কে GPU দেয়); NVIDIA-র CUDA on WSL guide বলে **WSL-এর ভেতরে Linux driver install করবেন না**। তারপর `--gpus all` আর Compose override Linux-এর মতোই।

### 3.8 Cloud-এ কী বদলায়

| বিষয় | Laptop | Cloud VM |
|---|---|---|
| কে পৌঁছায় | শুধু আপনি | সারা internet — bot প্রথম ঘণ্টাতেই port scan করে |
| HTTPS | লাগে না | বাধ্যতামূলক (password, health data) |
| Login | নেই | বাধ্যতামূলক |
| Secret | `.env` laptop-এ | server-এর `.env`, কড়া permission |
| Reboot | আপনি চালু করেন | stack নিজে থেকে ফিরতে হবে |
| Disk | অনেক জায়গা | ছোট; log আর পুরোনো image ভরে ফেলে |
| Data হারানো | "আবার বানাই" | backup আর restore drill |
| কেউ খেয়াল রাখে | আপনি সামনে বসে | monitoring আর alert |
| Update | `git pull` | tag দিয়ে release, rollback-এর পথ |
| খরচ | বিদ্যুৎ | ঘণ্টা ধরে bill, অলস থাকলেও |

**VM sizing (শুরুর budget, মাপা না):**

* **CPU-only VM:** RAM ≥ ১৬ GB (Q4 8B ≈৫-৬ GB + API-র torch + Chroma + OS); বেশি vCPU = দ্রুত CPU inference; disk ≥ ৫০ GB (image কয়েক GB, model ৪.৯ GB, volume, পুরোনো tag, log)।
* **GPU VM:** VRAM উপরের table থেকে (Q4 @৮k ≈৬.৭ GB → ১৬ GB-এর card আরামে); RAM ≥ ১৬ GB; disk ≥ ১০০ GB (CUDA-সহ Ollama image বড়, driver, backup-এর staging)।
* **GPU-র শ্রেণি:** cloud-এ সাধারণ inference card — T4 (১৬ GB, পুরোনো, সস্তার দিকে), L4 বা A10G (২৪ GB, নতুন), A100 (৪০/৮০ GB — একটা ৮B model-এর জন্য অপচয়)। নিজের PC-তে RTX 3060 (১২ GB) বা 4090 (২৪ GB)। এক `llama3.1:8b` Q4-এর জন্য ১৬ GB-ই যথেষ্ট।

### 3.9 VM থেকে `docker compose up` — ধাপে ধাপে (পড়ার জন্য, script না)

1. **VM তৈরি:** Ubuntu LTS, user-দের কাছের region, SSH key দিয়ে login, disk উপরের budget-এ।
2. **Harden:** sudo-সহ আলাদা user; SSH-এ password login বন্ধ (`PasswordAuthentication no`); security update স্বয়ংক্রিয় (`unattended-upgrades`)।
3. **Firewall:** provider-এর firewall/security group-এ শুধু 22 (সম্ভব হলে শুধু আপনার IP থেকে), 80, 443। **ফাঁদ:** Docker-এর published port Linux-এ ufw-কে পাশ কাটায় — Docker নিজের iptables rule বসায়, তাই `ufw deny 8000` দিলেও `ports: "8000:8000"` internet-এ খোলা থাকে। সমাধান: internal service-এর `ports` বাদ দিন, debug port `127.0.0.1:`-এ bind করুন, আর provider-এর firewall (VM-এর বাইরে) ব্যবহার করুন।
4. **Docker Engine:** Docker-এর নিজের apt repository থেকে (docs-এর Ubuntu install page); Compose plugin সাথে আসে। User-কে `docker` group-এ দিলে sudo লাগে না — কিন্তু Docker docs সতর্ক করে, `docker` group-এ থাকা মানে root-এর সমান ক্ষমতা।
5. **GPU VM হলে:** driver → `nvidia-smi` → Container Toolkit → `nvidia-ctk runtime configure` → Docker restart → test container (৩.২)।
6. **Code আর config:** `git clone` (বা registry থেকে image pull); server-এ `cp .env.example .env`, edit, `chmod 600 .env`। `.env` কখনো git-এ না।
7. **Production override:** একটা `caddy` service যোগ (80/443), `ui`-এর `8501` publish বন্ধ (`ports: !reset []`) — বাইরে থেকে শুধু Caddy।
8. **Single click, server-এও:** `docker compose up -d` (`.env`-এ `COMPOSE_FILE` দিয়ে GPU আর prod override)।
9. **DNS:** domain-এর A record VM-এর IP-তে; Caddy প্রথম request-এ certificate নেয়।
10. **যাচাই:** `docker compose ps -a`, browser-এ `https://`, `ollama ps`-এ `100% GPU`, একটা আসল প্রশ্ন, একটা Red-Flag Escalation প্রশ্ন (fixed উত্তর আসে কিনা), একটা knowledge base-এর বাইরের প্রশ্ন (Abstention)।
11. **Reboot test:** `restart: unless-stopped` + Docker boot-এ চালু (`systemctl enable docker`) → `sudo reboot` করে দেখুন সত্যিই ফেরে কিনা। না দেখা পর্যন্ত বিশ্বাস না।
12. **Backup, log rotation, monitoring** (নিচে)।

### 3.10 HTTPS আর Login: Reverse Proxy

User কখনো সরাসরি Streamlit-এ যায় না; সামনে একটা reverse proxy — TLS শেষ করে, login চায়, তারপর `ui:8501`-এ পাঠায়। Caddy-র automatic HTTPS-এর কারণে config তিন লাইনের:

```caddyfile
assistant.example.com {
    basic_auth {
        demo <bcrypt-hash-from: caddy hash-password>
    }
    reverse_proxy ui:8501
}
```

```yaml
  caddy:
    image: caddy:${CADDY_TAG:?pin CADDY_TAG}
    ports: ["80:80", "443:443"]
    volumes:
      - ./Caddyfile:/etc/caddy/Caddyfile:ro
      - caddy_data:/data          # certificate এখানে — volume না থাকলে প্রতি recreate-এ নতুন certificate
      - caddy_config:/config
    depends_on:
      ui: {condition: service_healthy}
    restart: unless-stopped
```

* Caddy certificate নেয় আর নিজেই renew করে — শর্ত: domain-এর DNS এই VM-এ, আর port 80/443 internet থেকে পৌঁছায়।
* `/data` volume বাধ্যতামূলক: certificate হারালে বারবার নতুন করে চাইতে হয়, আর certificate authority-র rate limit-এ আটকে যাওয়া যায়।
* Streamlit WebSocket ব্যবহার করে; Caddy-র `reverse_proxy` WebSocket upgrade নিজেই সামলায় — আলাদা config লাগে না।
* **Login বাধ্যতামূলক** — এটা health assistant। `basic_auth` demo-র ন্যূনতম; আসল user-দের জন্য identity-aware proxy বা SSO। আর মনে রাখুন log-এ রোগীর প্রশ্ন জমে — retention ঠিক করুন।
* বিকল্প: nginx + certbot (বেশি নিয়ন্ত্রণ, বেশি config), Traefik (container label দিয়ে config), বা tunnel service (কোনো port খোলা লাগে না — নিজের PC থেকে serve করার সময় কাজের)।

### 3.11 Secret

* Server-এর `.env`-এ, owner deploy user, `chmod 600`; `.gitignore` আর `.dockerignore`-এ। Image-এ কখনো না (Class 35: `docker history`)।
* Env variable `docker inspect`-এ দেখা যায় — server-এ যার Docker access আছে সে দেখে। কড়া হতে চাইলে Compose `secrets:` — secret file container-এ `/run/secrets/<name>`-এ mount হয়, env-এ থাকে না (app-কে file থেকে পড়তে হয়)।
* বড় setup-এ provider-এর secret manager। Secret ফাঁস হলে মুছে দেওয়া যথেষ্ট না — **rotate** করুন, git history-তে থেকে যায়।

### 3.12 Volume আর Backup

প্রথমে ভাগ করুন — কোন data **আবার বানানো যায়**, কোনটা **যায় না**:

| Volume | হারালে | Backup |
|---|---|---|
| `ollama_models` | আবার download (৪.৯ GB) | ঐচ্ছিক — শুধু সময় বাঁচায় |
| `chroma_data` | `HC_AUTO_INGEST` knowledge base থেকে আবার বানায় | ঐচ্ছিক, যতক্ষণ KB-র source repo-তে আছে |
| `caddy_data` | certificate আবার নেওয়া (rate limit-এর ঝুঁকি) | হালকা backup |
| ভবিষ্যতের user upload, chat history, feedback | **ফেরানো যায় না** | বাধ্যতামূলক, VM-এর বাইরে |

```bash
docker compose stop chroma          # consistent copy: chroma-র ভেতরে SQLite
docker run --rm -v healthcare-assistant_chroma_data:/v:ro -v "$PWD/backups":/b alpine \
  tar czf /b/chroma-$(date +%F).tgz -C /v .
docker compose start chroma
```

* Backup VM-এর **বাইরে** (object storage বা অন্য machine) — VM-এর disk মরলে একই disk-এর backup-ও যায়। বিকল্প: provider-এর disk snapshot।
* **Restore drill:** খালি volume-এ backup ফিরিয়ে stack চালান — একবারও restore না করা backup একটা অনুমান।
* **Upgrade-এর আগে backup:** নতুন Chroma version disk format একমুখী migrate করতে পারে; image tag ফেরানো সহজ, data ফেরানো backup ছাড়া অসম্ভব।

### 3.13 Monitoring: Week 15-এর সাথে জোড়া

* **Metrics:** API যদি Week 15-এর মতো `/metrics` দেয়, compose-এ একটা `prometheus` service (port publish না, internal) `api:8000/metrics` scrape করে; Week 15-এর burn-rate alert rule একই থাকে। GPU-র জন্য NVIDIA-র DCGM exporter (utilization, VRAM, temperature), host-এর জন্য node exporter।
* **বাইরে থেকে uptime check:** VM নিজের মৃত্যু নিজে report করতে পারে না — অন্য জায়গা থেকে `https://.../_stcore/health` বা API-র health endpoint probe করুন।
* **Log rotation:** Docker-এর default `json-file` log driver নিজে rotate করে না — মাসের পর মাস চললে disk ভরে stack থেমে যায়। প্রতিটা service-এ:

```yaml
    logging:
      driver: json-file
      options: {max-size: "10m", max-file: "3"}
```

* **Disk:** পুরোনো image জমে — `docker image prune` (rollback-এর জন্য দরকারি tag রেখে), আর disk-usage alert।

### 3.14 Release, Image Tag আর Rollback

* Build laptop বা CI-তে, tag version আর git SHA দিয়ে (যেমন `hc-api:1.4.0`), registry-তে push; server শুধু pull করে। Server-এ build করাও চলে (সহজ), কিন্তু ছোট VM-এ torch install ধীর, আর প্রতিটা server-এর image একটু আলাদা হতে পারে।
* `.env`-এ `API_TAG=1.4.0`। Release = tag বদলানো + `docker compose pull && docker compose up -d`। **Rollback = আগের tag + `up -d`।** তাই আগের ২-৩টা tag disk-এ/registry-তে রাখুন — `latest` দিয়ে rollback অসম্ভব, কারণ "আগেরটা" কোনটা কেউ জানে না।
* **Downtime সৎভাবে:** এক VM-এ `up -d` container recreate করে — API-র `start_period` (model load + ingest) পর্যন্ত সেবা বন্ধ। Zero-downtime-এর জন্য দুটো replica + load balancer লাগে — এক VM-এর Compose-এর সীমার বাইরে।
* Code rollback সহজ, data rollback backup ছাড়া হয় না (৩.১২)।

### 3.15 খরচের হিসাব

এই lecture-এ কোনো আসল দাম নেই — দাম provider, region, মাস আর exchange rate-ভেদে বদলায়, আর বানানো সংখ্যা ভুল সিদ্ধান্তে নেয়। Notebook-এর calculator নকল দাম (`FAKE_RATES`) দিয়ে পদ্ধতি দেখায়; আজকের আসল দাম provider-এর pricing page থেকে বসিয়ে নিজে চালান।

* **Bill-এর অংশ:** চালু থাকার ঘণ্টা × ঘণ্টার দাম (GPU VM-এ সবচেয়ে বড়) + disk (VM বন্ধ থাকলেও) + egress + public IP + snapshot/backup storage + domain — আর আপনার সময়।
* **অলস ঘণ্টাই আসল খরচ:** দিনে কয়েকশো প্রশ্নে একটা GPU হয়তো ১%-ও ব্যস্ত না; ২৪/৭ চালালে টাকার বেশিরভাগ অলসতার। Office hours-এ চালু রাখলে (দিনে ১০ ঘণ্টা × ২২ দিন) compute bill প্রায় তিন ভাগের এক ভাগ।
* **Right-size:** Q4 8B ১৬ GB-এর card-এ আরামে — ৮০ GB-এর card ভাড়া মানে বেশিরভাগ VRAM অলস।
* **Spot/preemptible VM:** অনেক সস্তা, কিন্তু provider যেকোনো সময় নিয়ে নিতে পারে — volume আলাদা persistent disk-এ রাখলে আর stack নিজে উঠতে পারলে (single-click!) demo-র জন্য চলে।
* **Hosted API-র সাথে breakeven:** কম traffic-এ per-token API সস্তা, traffic বাড়লে নিজের GPU। RAG-এ input token বেশি (retrieved passage) — top-k কমালে API bill-ও কমে। কিন্তু Healthcare Assistant-এ hosted API মানে রোগীর প্রশ্ন তৃতীয় পক্ষের কাছে — সিদ্ধান্ত শুধু টাকার না।
* **Capacity:** Week 15-এর Little's law দিয়ে এক slot কত প্রশ্ন টানে সেটা হিসাব করুন; ৬০% utilization-এর বেশি plan করবেন না।

**Bangladesh-এর প্রেক্ষাপট (high level):**

* **Hyperscaler** (AWS, Google Cloud, Azure): GPU VM কাছের region-এ (যেমন Mumbai বা Singapore) পাওয়া যায়; বিল USD-তে — international card বা dual-currency card-এর সীমা বাস্তব বাধা। Data দেশের বাইরে যায় — health data হলে প্রতিষ্ঠানের policy আর দেশের data-protection নিয়ম আগে দেখুন।
* **Local VPS / data center:** BDT-তে বিল, local payment, data দেশে, latency কম; কিন্তু GPU instance বিরল — বেশিরভাগ CPU VPS। Spec, uptime, support আর backup/snapshot সুবিধা নিজে যাচাই করুন।
* **Student-এর বাস্তব পথ:** local CPU VPS-এ `llama3.2:3b` দিয়ে demo; অথবা নিজের GPU PC + একটা tunnel (কোনো port খোলা লাগে না) — তখন লোডশেডিং-এর জন্য UPS আর ISP বিভ্রাট মাথায় রাখুন, আর reboot-এ stack ফেরার test আরও জরুরি।

### 3.16 কখন Compose ছেড়ে অন্য কিছু

| Option | কখন | সুবিধা | সীমা |
|---|---|---|---|
| **এক VM-এ Compose** (course) | এক team, এক GPU, কয়েক মিনিট downtime মেনে নেওয়া যায় | laptop-এর মতোই, সস্তা, বোঝা সহজ | single point of failure; zero-downtime নেই; scale হাতে |
| Managed container service (GPU-সহ, যেমন Cloud Run, ECS, Azure Container Apps) | traffic ওঠানামা করে, ops কম চান | managed TLS আর load balancer; কিছুতে scale-to-zero | cold start-এ ৫ GB model load; vendor lock-in; GPU সব region-এ নেই |
| Kubernetes + NVIDIA GPU Operator | অনেক service, অনেক node, অনেক team | autoscaling, rolling update, liveness probe-এ self-healing | শেখা আর চালানোর খরচ অনেক; এক GPU-র project-এ overkill |
| Managed LLM endpoint / hosted API | model চালাতেই চান না | GPU ops শূন্য | data বাইরে, per-token খরচ |

সরে যাওয়ার সংকেত: একাধিক replica (high availability) লাগছে, একাধিক GPU node, দিনে কয়েকবার deploy downtime ছাড়া, বা কয়েকজনের বেশি মানুষ একই system চালাচ্ছে। এর আগে Kubernetes মানে সমস্যার চেয়ে বড় সমাধান।

---

## 4. Details & Valid Points

### 4.1 GPU Error → কারণ → Fix

| দেখছেন | সম্ভাব্য কারণ | Fix |
|---|---|---|
| `could not select device driver "nvidia" with capabilities: [[gpu]]` | Toolkit নেই/configure হয়নি, বা CPU machine-এ GPU override | toolkit install + `nvidia-ctk runtime configure` + Docker restart; CPU machine-এ override বাদ |
| Host-এ `nvidia-smi` চলে, `ollama ps`-এ `100% CPU` | override ছাড়া `up` | `-f docker-compose.gpu.yml` বা `COMPOSE_FILE` |
| Container-এ `nvidia-smi: command not found` | device request নেই | override দিয়ে recreate |
| `48%/52% CPU/GPU` | VRAM-এ আঁটে না | `num_ctx` কমান, KV q8_0, ছোট quantization, বড় GPU |
| গতকাল GPU ছিল, আজ নেই | `systemctl daemon-reload` (systemd cgroup), বা kernel update-এ driver module নেই | container restart; host-এ `nvidia-smi` দেখে driver ঠিক করুন |
| GPU দেখে কিন্তু Ollama ব্যবহার করে না | driver ৫৫০-এর নিচে বা compute capability < ৫.০ | host driver upgrade; পুরোনো card হলে CPU মেনে নিন |
| Mac-এ container ধীর | Docker Desktop-এ macOS GPU passthrough নেই | native Ollama + `host.docker.internal` |
| WSL2-এ GPU পায় না | WSL-এর ভেতরে Linux driver install করা | শুধু Windows driver রাখুন |

### 4.2 Public VM-এর Security Basics

* Ollama আর Chroma-র কোনো auth নেই — কখনো publish না; API-ও শুধু internal network-এ, বাইরে শুধু Caddy।
* Docker-এর published port ufw-কে পাশ কাটায় — provider-এর firewall আর `127.0.0.1` bind।
* `docker` group = root-এর সমান; কাকে দেবেন ভেবে দিন।
* Container non-root (Class 35), image tag pinned, OS update স্বয়ংক্রিয়।
* Health data: HTTPS, login, log retention, backup encrypted আর access-সীমিত।

### 4.3 12-Factor-এর আরও তিনটা Factor

* **Build, release, run** — image একবার build হয়; তার সাথে একটা নির্দিষ্ট config (`.env`) জুড়লে একটা release; তারপর run। প্রতিটা release-এর আলাদা tag থাকলে তবেই rollback।
* **Logs** — app log stdout-এ লেখে, জমানো/rotate করা platform-এর কাজ (Docker logging driver)।
* **Disposability** — container যেকোনো সময় মুছে আবার ওঠে; data volume-এ, startup দ্রুত, shutdown পরিচ্ছন্ন (Class 35-এর exec-form `CMD`)।

---

## 5. Related Analogy: নতুন শাখায় রেস্তোরাঁ খোলা

**GPU passthrough = রান্নাঘরে গ্যাস লাইন।**

* **Host driver** = বাড়ির মূল গ্যাস সংযোগ। শুধু বাড়িওয়ালা বসাতে পারে, প্রতিটা চুলায় আলাদা না (driver image-এ থাকে না)।
* **Container Toolkit** = গ্যাস মিস্ত্রি, যে মূল লাইন থেকে প্রতিটা চুলায় পাইপ টানে।
* **Image-এর CUDA** = চুলার burner — মূল লাইনের চাপের সাথে মিলতে হয় (image-এর CUDA ≤ driver-এর CUDA)।
* **`--gpus` / Compose device** = "এই চুলায় গ্যাস দাও" অনুরোধ। না চাইলে চুলা কাঠে চলে (CPU) — রান্না হয়, শুধু অনেক ধীরে, আর কেউ অভিযোগ করে না। এটাই নীরব failure।
* **VRAM** = হাঁড়ির আকার। বড় রান্না (FP16, লম্বা context) ছোট হাঁড়িতে আঁটে না, অর্ধেক পাশের চুলায় (partial offload)।

**Cloud VM = শহরের মার্কেটে দোকান ভাড়া।**

* **Firewall** = শাটার — শুধু সামনের দরজা (80/443) খোলা, পেছনের গুদামের দরজা (Ollama, Chroma) সবসময় বন্ধ।
* **Reverse proxy + HTTPS + login** = দরজার গার্ড, যে পরিচয় দেখে তবেই ঢুকতে দেয়।
* **Secret** = ক্যাশবাক্সের চাবি — মালিকের পকেটে, সাইনবোর্ডে (image) না।
* **Backup** = হিসাবের খাতার কপি অন্য বাড়িতে — দোকানে আগুন লাগলে দোকানের ভেতরের কপিও পোড়ে।
* **Monitoring** = CCTV আর পাশের দোকানদার, যে ফোন করে জানায় "আপনার শাটার বন্ধ কেন?"
* **Tag আর rollback** = নতুন মেনু খারাপ হলে গতকালের ছাপানো মেনু ফেরত — "সর্বশেষ মেনু" লেখা থাকলে গতকালেরটা কোনটা কেউ জানে না।
* **খরচ** = ভাড়া মাসে মাসে, খদ্দের না এলেও; দরকারের চেয়ে বড় দোকান মানে খালি জায়গার ভাড়া।
* **Kubernetes** = দেশজুড়ে franchise system — একটা দোকানের জন্য বানালে system চালাতেই দিন যায়।

---

## 6. Common Pitfalls

| Pitfall | কী হয় | Fix |
|---|---|---|
| GPU VM-এ override ছাড়া `up` | নীরবে CPU, GPU-র টাকা অপচয় | `COMPOSE_FILE`; `ollama ps`-এ `100% GPU` যাচাই |
| Driver image-এ ভরার চেষ্টা | কাজ করে না; image ভারী | driver host-এ, toolkit সেতু |
| "`nvidia-smi`-র CUDA Version = installed CUDA" | ভুল version-এর image বাছাই | ওটা driver-এর সর্বোচ্চ সমর্থিত CUDA |
| Base compose-এ GPU reservation | CPU machine-এ `up` ব্যর্থ | আলাদা `docker-compose.gpu.yml` |
| `capabilities` বাদ | deploy-এর সময় error | `capabilities: [gpu]` |
| `count` আর `device_ids` একসাথে | error | একটা বেছে নিন |
| লম্বা context-এর VRAM না গোনা | partial offload, ধীর | KV = ১২৮ KiB/token (8B, f16); `num_ctx` স্পষ্ট |
| Mac-এ container-এ GPU আশা | CPU-তে চলে | native Ollama + `host.docker.internal` |
| WSL-এ Linux driver | GPU পায় না | শুধু Windows driver |
| `ufw deny` দিয়ে container port বন্ধ ভাবা | Docker ufw-কে পাশ কাটায়, port খোলা | `ports` বাদ / `127.0.0.1` / provider firewall |
| Ollama বা API public | auth ছাড়া LLM আর data | শুধু reverse proxy publish |
| HTTPS আছে, login নেই | health assistant সবার জন্য খোলা | `basic_auth` ন্যূনতম, SSO আসল user-এ |
| Caddy-র `/data` volume নেই | প্রতি recreate-এ নতুন certificate, rate limit | `caddy_data` volume |
| `restart` নেই বা Docker boot-এ বন্ধ | reboot-এর পর stack মৃত | `unless-stopped` + `systemctl enable docker` + reboot test |
| Default `json-file` log | disk ভরে stack থামে | `max-size`, `max-file` |
| `latest` দিয়ে deploy | rollback অসম্ভব | version tag, আগের ২-৩টা রাখা |
| Backup একই VM-এ, restore কখনো test হয়নি | disk মরলে সব শেষ | VM-এর বাইরে, restore drill |
| Upgrade-এর আগে backup না | data format একমুখী migrate | upgrade-এর আগে backup |
| ২৪/৭ বড় GPU VM, প্রায় কোনো traffic না | bill-এর বেশিরভাগ অলসতা | right-size, schedule, breakeven হিসাব |
| `down -v` production-এ | model, vector, certificate শেষ | কখনো না; volume-এর নাম ধরে কাজ |

---

## 7. Interview Q&A

1. **Docker container কীভাবে GPU ব্যবহার করে?** Driver (kernel module) host-এ থাকে, image-এ না। Image-এ CUDA-র user-space library। NVIDIA Container Toolkit container start-এর সময় GPU device আর driver library container-এ mount করে — কিন্তু শুধু তখন, যখন run-এ device request থাকে (`--gpus all` বা Compose-এর `deploy.resources.reservations.devices` + `capabilities: [gpu]`)।
2. **Host-এ `nvidia-smi` চলে, কিন্তু container-এ Ollama ধীর — কীভাবে debug করবেন?** ক্রমে: container-এর ভেতরে `nvidia-smi` (না চললে device request বা toolkit); `ollama ps`-এর `PROCESSOR` (`100% CPU` = GPU ব্যবহার হচ্ছে না, split = VRAM কম); `docker compose logs ollama`-এ GPU detection; driver version Ollama-র ন্যূনতম (৫৫০+) কিনা, card-এর compute capability ৫.০+ কিনা। সবচেয়ে সাধারণ উত্তর: override file ছাড়া `up`।
3. **`llama3.1:8b`-এর কত VRAM লাগবে?** Weights = params × bits ÷ 8: Q4_K_M ≈৪.৯ GB, Q8 ≈৮.৫, FP16 ≈১৬। KV cache = 2 × layer × KV head × head dim × byte × context — 8B-এ ১২৮ KiB/token (f16), ৮k context-এ ≈১.১ GB, ৩২k-এ ≈৪.৩ GB। সাথে কয়েকশো MB থেকে এক GB overhead। তাই Q4 @৮k ≈৬-৭ GB: ৮ GB-এ টায়টায়, ১২-১৬ GB-এ আরামে। VRAM-এর ৯০% পর্যন্ত plan।
4. **GPU config আলাদা override file-এ কেন?** NVIDIA runtime ছাড়া machine-এ device request-ই error দেয়; base file portable থাকে (laptop, Mac, CPU VM), GPU machine-এ `-f docker-compose.gpu.yml` বা `COMPOSE_FILE`। শুধু Ollama GPU পায় — API-র embedding CPU-তে যথেষ্ট, GPU দিলে image-এ CUDA torch-এর কয়েক GB।
5. **CUDA version-এর compatibility কীভাবে কাজ করে?** `nvidia-smi`-র "CUDA Version" হলো driver সর্বোচ্চ কোনটা চালায়; image-এর CUDA runtime তার সমান বা নিচে হতে হবে। তাই host driver আপডেট রাখুন; নতুন image + পুরোনো driver = GPU init fail, নীরবে CPU।
6. **এই stack একটা cloud VM-এ কীভাবে deploy করবেন?** Ubuntu LTS VM, SSH key-only, provider firewall-এ শুধু 22/80/443; Docker-এর apt repo থেকে Engine + Compose; GPU হলে driver → toolkit → `nvidia-ctk` → test; repo clone, server-এ `.env` (`chmod 600`); Caddy override (HTTPS + login), `ui`-এর port বন্ধ; `docker compose up -d`; DNS; যাচাই (`ps`, `ollama ps`, আসল প্রশ্ন, red-flag প্রশ্ন); reboot test; backup, log rotation, external uptime check।
7. **Port খোলা রাখার নিরাপদ নিয়ম কী? ufw কি যথেষ্ট?** বাইরে শুধু reverse proxy (80/443); Ollama, Chroma, API শুধু internal network-এ, debug port `127.0.0.1`-এ। ufw যথেষ্ট না — Docker-এর published port নিজের iptables rule দিয়ে ufw-কে পাশ কাটায়; provider-এর firewall আর publish না করাই আসল সুরক্ষা।
8. **Server-এ secret কোথায়?** Server-এর `.env`-এ, deploy user-এর, `chmod 600`, git আর image-এর বাইরে। Env `docker inspect`-এ দেখা যায়, তাই কড়া হলে Compose `secrets:` (file mount, `/run/secrets`) বা provider-এর secret manager। ফাঁস হলে rotate।
9. **কোন volume backup করবেন, কীভাবে?** আগে ভাগ: model আর vector আবার বানানো যায় (download, auto-ingest), user data যায় না। Consistent copy-র জন্য service থামিয়ে volume tar, VM-এর বাইরে রাখা, বা disk snapshot। Restore drill না হলে backup অপ্রমাণিত। Upgrade-এর আগে সবসময় backup।
10. **নতুন version ভাঙলে rollback কীভাবে?** Image version/SHA দিয়ে tag, registry-তে; `.env`-এ tag; rollback = আগের tag + `docker compose up -d`। `latest`-এ rollback অসম্ভব। Code ফেরানো সহজ, data format migrate হয়ে থাকলে backup ছাড়া ফেরানো যায় না। এক VM-এ recreate-এ কয়েক মিনিট downtime — zero-downtime লাগলে দুটো replica + load balancer।
11. **GPU VM, CPU VM না hosted API — কীভাবে ঠিক করবেন?** তিনটা প্রশ্ন: traffic (Little's law দিয়ে capacity, utilization), খরচ (ঘণ্টা × দাম + disk + egress বনাম token × দাম; breakeven বের করুন; অলস ঘণ্টা আসল খরচ), আর data (health data তৃতীয় পক্ষে যেতে পারে কিনা)। কম traffic-এ API বা schedule-করা GPU VM, স্থির বেশি traffic-এ নিজের GPU; data বাইরে যেতে না পারলে API বাদ।
12. **কখন Compose ছেড়ে Kubernetes বা managed service?** যখন high availability (একাধিক replica), একাধিক GPU node, downtime ছাড়া ঘনঘন deploy, autoscaling বা অনেক team লাগে। তার আগে এক VM-এ Compose সহজ, সস্তা আর বোঝা যায় — Kubernetes-এর operational খরচ এক GPU-র project-এ লাভের চেয়ে বেশি।

---

## Achievement: Single-Click Deployment (Week 18 Shared Project)

**`../Class 1-2 Project/`** — Week 17-এর Healthcare Assistant-এর self-contained কপি (ADR 0002), ৪-service Compose stack: `ui` (Streamlit, 8501, `HC_API_URL=http://api:8000`), `api` (FastAPI, 8000, `HC_OLLAMA_HOST=http://ollama:11434`, `HC_CHROMA_HOST=chroma`, `HC_AUTO_INGEST=true`), `chroma` (persistent volume), `ollama` (official `ollama/ollama` image, model volume) + one-shot `ollama-init` (`llama3.1:8b` pull)। Healthcheck, `depends_on: condition: service_healthy`, named volume, `.env.example`, আর এই ক্লাসের অংশ — **optional `docker-compose.gpu.yml`** (NVIDIA `deploy.resources.reservations.devices`, শুধু `ollama`-র জন্য)। Single click = একটা local command: `make up` / `start.sh` / `start.ps1` (ভেতরে `docker compose up -d`)। Cloud deployment এই lecture-এ theory (ADR 0010) — project-এ কোনো deploy script নেই, cloud account ধরে নেওয়া হয়নি। Stack আর GPU override statically verified (ADR 0008); GPU passthrough এই repo-তে test হয়নি — প্রথম আসল test NVIDIA machine-এ। এই ক্লাসের companion notebook **`gpu_cloud_deploy_lecture.ipynb`** সম্পূর্ণ self-contained — GPU, Docker বা cloud account ছাড়াই চলে।

---

## 🧠 Brain Teasers & Exercises (নিজে চেষ্টা করুন)

1. **VRAM Math:** `llama3.1:8b` Q4_K_M, context ১৬k। f16 KV-তে ৮ GB GPU-তে আঁটে? KV q8_0-তে? Notebook-এর estimator দিয়ে যাচাই করুন, তারপর আসল GPU থাকলে `ollama ps`-এর `SIZE` আর `PROCESSOR`-এর সাথে মেলান।
2. **GPU Detective:** GPU VM-এ `ollama ps` বলছে `100% CPU`। চারটা সম্ভাব্য কারণ যে ক্রমে check করবেন সেই ক্রমে লিখুন, প্রতিটার জন্য একটা command।
3. **Override Merge:** GPU override-এ `ollama`-কে `ports: ["127.0.0.1:11434:11434"]` দিন, আর base-এ `ui`-এর `8501` আছে। `docker compose -f ... -f ... config` দেখে বলুন final `ports` কী। Production override-এ `ui`-এর port কীভাবে সরাবেন?
4. **ufw Trap:** একটা Linux VM-এ `ufw deny 8000` দিয়ে `ports: ["8000:8000"]`-এর container চালান, অন্য machine থেকে `curl` করুন। কী হয়, কেন? দুটো fix লিখুন।
5. **Cost Drill:** Notebook-এর `FAKE_RATES`-এ দুটো provider-এর আজকের আসল দাম বসান (তারিখ লিখে রাখুন)। ২৪/৭ বনাম office hours-এর পার্থক্য কত? দিনে কত প্রশ্নে hosted API আর GPU VM-এর খরচ সমান? Healthcare Assistant-এর জন্য কোনটা বেছে নেবেন — কেন খরচই একমাত্র মাপকাঠি না?
6. **Reboot আর Restore Drill:** একটা VM-এ stack চালিয়ে `sudo reboot` — কী কী ফেরে? তারপর `chroma_data`-র backup নিন, volume মুছুন, restore করুন — কত সময় লাগল? `HC_AUTO_INGEST` থাকলে এই backup কি আদৌ দরকার? কখন দরকার হবে?

---

## References

* Docker — GPU support in Compose: https://docs.docker.com/compose/how-tos/gpu-support/ · GPU access with `docker run`: https://docs.docker.com/engine/containers/gpu/ · GPU support in Docker Desktop (Windows/WSL2): https://docs.docker.com/desktop/features/gpu/ · Compose `gpus` and `deploy` reference: https://docs.docker.com/reference/compose-file/services/ · https://docs.docker.com/reference/compose-file/deploy/
* Docker — Merge multiple Compose files: https://docs.docker.com/compose/how-tos/multiple-compose-files/merge/ · Merge rules and `!reset`: https://docs.docker.com/reference/compose-file/merge/ · Pre-defined env vars (`COMPOSE_FILE`): https://docs.docker.com/compose/how-tos/environment-variables/envvars/ · Secrets in Compose: https://docs.docker.com/compose/how-tos/use-secrets/
* Docker — Install Engine on Ubuntu: https://docs.docker.com/engine/install/ubuntu/ · Linux post-install (`docker` group): https://docs.docker.com/engine/install/linux-postinstall/ · Packet filtering and firewalls (ufw): https://docs.docker.com/engine/network/packet-filtering-firewalls/ · Port publishing: https://docs.docker.com/engine/network/port-publishing/ · JSON File logging driver: https://docs.docker.com/engine/logging/drivers/json-file/ · Volumes (backup/restore): https://docs.docker.com/engine/storage/volumes/
* NVIDIA Container Toolkit — Install guide: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html · Sample workload: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/sample-workload.html · CUDA compatibility: https://docs.nvidia.com/deploy/cuda-compatibility/ · CUDA on WSL: https://docs.nvidia.com/cuda/wsl-user-guide/index.html · CUDA GPUs (compute capability): https://developer.nvidia.com/cuda-gpus · DCGM exporter: https://github.com/NVIDIA/dcgm-exporter · GPU Operator: https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/index.html
* Ollama — Docker: https://docs.ollama.com/docker · GPU support: https://docs.ollama.com/gpu · FAQ (`ollama ps`, KV cache type, context, macOS Docker GPU): https://docs.ollama.com/faq · Context length: https://docs.ollama.com/context-length · `llama3.1` library page (tags and sizes): https://ollama.com/library/llama3.1 · Image Dockerfile: https://github.com/ollama/ollama/blob/main/Dockerfile
* Llama 3.1 8B model card and config: https://huggingface.co/meta-llama/Llama-3.1-8B
* Caddy — Automatic HTTPS: https://caddyserver.com/docs/automatic-https · Reverse proxy quick-start: https://caddyserver.com/docs/quick-starts/reverse-proxy · `basic_auth`: https://caddyserver.com/docs/caddyfile/directives/basic_auth · Official image: https://hub.docker.com/_/caddy
* The Twelve-Factor App — Config: https://12factor.net/config · Build, release, run: https://12factor.net/build-release-run · Logs: https://12factor.net/logs · Disposability: https://12factor.net/disposability
* Google SRE Book — Monitoring Distributed Systems: https://sre.google/sre-book/monitoring-distributed-systems/

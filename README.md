# Video → Voice-Only 20-Minute Video (Telegram Bot) + 150-Picture ZIP (Web)

Two front ends, one processing core. A **Telegram bot** that turns any video
into a **20-minute voice-only video**:

```
uploaded video
   │
   ├─ 1. 300 representative photos       two photos from each of the 150 equal
   │                                     segments of the video (not consecutive
   │                                     frames), black/flat/corrupt/duplicate
   │                                     photos rejected
   ├─ 2. 20:00 slideshow                 150 pairs (photo 1+2, ..., 299+300),
   │                                     both photos side by side for exactly
   │                                     8,000 ms → 150 × 8 s = 1200 s; each
   │                                     half is face-centred, never stretched
   ├─ 3. vocal separation                 a real source-separation model
   │                                     (Demucs / Hybrid Transformer) isolates
   │                                     the voice; music and noise are dropped
   ├─ 4. overlay                          your PNG/JPG/WebP logo stays visible
   │                                     in one of five positions - or covers
   │                                     the whole screen (black = transparent)
   └─ 5. final render                     H.264 + AAC MP4, 20:00 exactly
```

The whole conversation happens in **one status message** that is continuously
edited - download bar, video information, overlay questions, then the five
progress steps - and the finished video arrives as an attachment.

The **web interface** uses a picture-only path for the browser: upload the
video on a page, watch the three steps, download a ZIP with the same 150
side-by-side pictures. There is no vocal separation, overlay or MP4 on this
path. No Telegram is involved, so its 20 MB / 50 MB limits never apply - only
`MAX_VIDEO_SIZE`.

---

## Contents

- [Quick start (Docker, recommended)](#quick-start-docker-recommended)
- [Quick start (manual install)](#quick-start-manual-install)
- [Web interface (browser upload)](#web-interface-browser-upload)
- [Deploy the web picture tool (Render)](#deploy-the-web-picture-tool-render)
- [Deploy on a VPS](#deploy-on-a-vps)
- [Talking to the bot](#talking-to-the-bot)
- [Configuration](#configuration)
- [How it works](#how-it-works)
- [Verifying the pipeline](#verifying-the-pipeline)
- [Troubleshooting](#troubleshooting)
- [Project layout](#project-layout)

---

## Quick start (Docker, recommended)

Docker Compose starts **three** containers:

| service             | purpose                                                                 |
| ------------------- | ----------------------------------------------------------------------- |
| `telegram-bot-api`  | a local Telegram Bot API server: lifts the limits from 20 MB / 50 MB to 2000 MB |
| `bot`               | the Telegram bot (FFmpeg + Demucs + pre-downloaded model weights)       |
| `web`               | the browser interface on port 80 - same image, same pipeline            |

```bash
cp .env.example .env      # fill in TELEGRAM_BOT_TOKEN, TELEGRAM_API_ID, TELEGRAM_API_HASH, WEB_PASSWORD
docker compose up -d --build
docker compose logs -f bot
```

The first build takes a few minutes (PyTorch + model weights). Afterwards a
`docker compose up -d` starts in seconds. The host directories `./temp`,
`./output`, `./models` and `./logs` are mounted into the containers, so you can
watch the working files and pick up the results directly. Open
`http://localhost/` for the web interface.

With a GPU (much faster vocal separation):

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

> **Why the local Bot API server?** The official API caps bot *downloads* at
> 20 MB and *uploads* at 50 MB. A typical source video and a 20-minute output
> are both larger. The local server has no such limit. If you only ever test
> with short clips, delete the `telegram-bot-api` service and the two
> `TELEGRAM_API_*_URL` variables from `docker-compose.yml` and the bot will use
> `api.telegram.org` instead.

## Quick start (manual install)

Requirements:

- **Python 3.11+**
- **FFmpeg** (`ffmpeg` **and** `ffprobe` on the `PATH`)
  - Windows: `winget install Gyan.FFmpeg` (or `choco install ffmpeg`)
  - Debian/Ubuntu: `sudo apt install ffmpeg`
  - macOS: `brew install ffmpeg`

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu   # or .../cu126 for CUDA
python -m pip install -r requirements.txt
python scripts/download_model.py        # pre-fetch the separation weights (~80 MB)
cp .env.example .env                    # fill in TELEGRAM_BOT_TOKEN
python -m bot.telegram_bot
```

Vocal separation is the slowest step. With the default auto worker count
(½ the CPU cores, capped at 8) and the default crossfade `DEMUCS_OVERLAP=0.1`,
the htdemucs model runs at roughly **1.45x realtime on a modern 4-core laptop
CPU**, so a 20-minute video spends about **14 minutes** in this step. A
16-vCPU server runs 8 workers and brings separation down to ~5 minutes, a
CUDA GPU to 1-3 minutes. All other stages together finish in under
2 minutes.

Only the **first 20 minutes of audio** are ever separated - audio beyond that
is trimmed out before the model runs, so a 40-minute source pays separation
for 20 minutes, not 40 (the discarded part could not have appeared in the
20:00 result anyway).

> Laptops cap the CPU on battery / "power saver" modes - separation can run
> 2-3x slower there. Plug in and pick "Best performance" for the numbers above.

## Web interface (browser upload)

A picture-only front end - no Telegram account or bot token needed:

```bash
python -m web.app        # then open http://localhost:8080
```

1. Pick the video and press **Upload and make the ZIP**. Upload progress and
   the three processing steps are shown live; **Cancel** stops a queued or
   running job.
2. The tool selects 300 photos from the video, builds the 150 side-by-side
   pictures (photo 1 + photo 2 = picture 1, …, photo 299 + photo 300 =
   picture 150), and packs them into one ZIP.
3. When it is done, **Download the ZIP** saves `150-pictures-<job>.zip` with
   `picture_001.jpg` … `picture_150.jpg` at 1280x720.

There is no overlay step, no vocal separation and no MP4 on this path.  The
server deletes the result right after a successful download - and anything
never downloaded after `WEB_JOB_TTL_SECONDS` (default 1 hour).

Because the video never passes through Telegram, only `MAX_VIDEO_SIZE`
applies here (default 2000 MB).  Set `WEB_PASSWORD` in `.env` before the
interface is reachable from the internet: the whole page, downloads included,
then sits behind HTTP Basic auth. With Docker the page is served on port 80
(change `WEB_PUBLISH_PORT` in `.env` if that port is taken).

## Deploy the web picture tool (Render)

The browser tool needs FFmpeg, several minutes per job and uploads larger than
4.5 MB, so it cannot run on Vercel's serverless functions. Use a long-lived
container instead - Render is the shortest path:

1. Push this folder to a GitHub repository.
2. Open `https://render.com/deploy?repo=<your-repository-url>`. Render reads
   `render.yaml`, builds `Dockerfile.web` and asks you for `WEB_PASSWORD`.
3. Choose **Free** (0.1 CPU, 512 MB - works but a job can take a long time) or
   **Starter** (about $7/month, much faster) and deploy. The health check is
   `/api/health`; the page is served on the service URL.

`Dockerfile.web` contains only FFmpeg and the web dependencies - it does not
install Demucs or PyTorch, so the image stays small. `render.yaml` also sets
the temporary directories to `/tmp`, which is writable inside the container.
Any other Docker host works the same way:

```bash
docker build -f Dockerfile.web -t pictures-150 .
docker run -p 8080:8080 -e WEB_PASSWORD=change-me pictures-150
```

> Vercel limitation, in one sentence: its functions reject request bodies over
> 4.5 MB, do not include FFmpeg, and stop after 60 seconds, so the 150-picture
> job cannot complete there.

## Deploy on a VPS

The bot is built to run on a persistent, always-on server: a job takes minutes
and keeps working files on disk, and long polling means the bot must stay
online to receive messages. Docker Compose is the intended way to deploy.

**Server sizing**

| resource | minimum                   | comfortable |
| -------- | ------------------------- | ----------- |
| CPU      | 2 vCPU                    | 4+ vCPU     |
| RAM      | 8 GB (a 20-min track peaks ≈5-6 GB with `DEMUCS_JOBS=0`) | 16 GB |
| Disk     | 20 GB SSD                 | 40 GB       |
| OS       | Ubuntu 22.04+ (any Docker host) | same  |

A CUDA GPU makes vocal separation 10-30x faster - use `docker-compose.gpu.yml`
on a GPU server with the NVIDIA Container Toolkit installed.

**Step by step**

1. From inside the project directory, create a deployment bundle in the parent
   folder (excludes `.env`, caches, model weights and runtime data; writing it
   outside the project keeps the archive from including itself):

   ```bash
   tar --exclude=./.env --exclude=./temp --exclude=./output --exclude=./logs \
       --exclude=./models/hub --exclude=./video-bot-deploy.tar.gz \
       --exclude='*/__pycache__' --exclude='*.pyc' \
       -czf ../video-bot-deploy.tar.gz .
   scp ../video-bot-deploy.tar.gz you@your-server:~
   ```

2. On the server, unpack and run the bootstrap script - it installs Docker (if
   missing), asks for the three Telegram credentials plus a web password (leave
   it empty if the page should be open to anyone) and starts everything:

   ```bash
   mkdir -p video-bot && tar -xzf video-bot-deploy.tar.gz -C video-bot && cd video-bot
   bash scripts/vps_setup.sh
   ```

   Prefer to type the commands yourself? The script performs exactly three
   steps: install Docker (`curl -fsSL https://get.docker.com | sh`, then
   `sudo usermod -aG docker $USER` and re-login), create `.env`
   (`cp .env.example .env && nano .env` - token, API id, API hash,
   `WEB_PASSWORD`, plus any settings you changed locally such as
   `MAX_VIDEO_SIZE`), and start the stack
   (`docker compose up -d --build` - the first build takes ~5-15 minutes; watch
   it with `docker compose logs -f bot`).

3. A healthy start reports `connected to Telegram as @<your bot>` and
   `Application started`, with `API mode local Bot API server` and the
   effective download/upload limits from your `.env` (up to 2000 MB - the
   official API's 20 MB / 50 MB caps do not apply here). The web interface is
   up on `http://<server>/` at the same time (the `video-web` container shows
   as healthy).

4. Send the bot a video on Telegram, or use the web interface - results and
   logs land in `./output` and `./logs` on the host. To update later, upload a
   fresh bundle and run `docker compose up -d --build` again.

The bot only makes outbound connections (long polling). The web interface
needs **one inbound port** - the one it is published on (80 by default,
`WEB_PUBLISH_PORT` in `.env` otherwise): open it in the cloud firewall
(Hetzner: *Firewalls*, or the ufw rules of your image). Everything else stays
closed; the bundled Bot API server is bound to `127.0.0.1` on purpose - keep
it that way, and use an SSH tunnel if you ever need to reach it directly.
Set `WEB_PASSWORD` before exposing the page: without it the interface (and
every download) is open to whoever finds the address.

## Talking to the bot

| command    | what it does                                                       |
| ---------- | ------------------------------------------------------------------ |
| `/start`   | greeting + "please upload your video"                              |
| `/help`    | the same instructions, including the current size limit             |
| `/status`  | job id, current step, elapsed time - even while queued             |
| `/cancel`  | cancels a running job or discards a pending one                    |
| `/skip`    | skips the overlay step                                             |

Typical flow:

1. Send a video (or a video file as a document).
2. The bot downloads it with a progress bar, then shows
   **duration / resolution / FPS / audio track** and asks for an overlay.
3. Send a **PNG / JPG / WebP** image (a transparent PNG works best) - or press
   **Skip overlay**.
4. Choose a **position** (↖ top-left, ↗ top-right, ⏺ center, ↙ bottom-left,
   ↘ bottom-right) and a **size** (original, 25 %, 50 %, 75 %, 100 % of the
   canvas width). Aspect ratio and transparency are always preserved.
5. Watch the five steps:

   ```
   Step 1/5 — Extracting photos...
   [███░░░░░░░] 28%

   extracted 42 of 300 photos
   ```

6. **Processing complete.** and the file **final_video.mp4** arrives with the
   caption *"Your 20-minute voice-only video is ready."*

Multiple users are supported: each request gets its own job directory
(`temp/job_<user id>_<timestamp>_<random>/`), its own queue slot and its own
status message. Everything is deleted from disk once the video has been sent
(or when the job fails, unless `KEEP_TEMP_ON_ERROR=true`).

## Configuration

Everything is configured through environment variables (`.env`); see
[`.env.example`](.env.example) for the full list with defaults.

| variable                | default                  | meaning                                                     |
| ----------------------- | ------------------------ | ----------------------------------------------------------- |
| `TELEGRAM_BOT_TOKEN`    | –                        | bot token from @BotFather (*required*)                      |
| `TELEGRAM_API_ID/HASH`  | –                        | only for the local Bot API server                           |
| `TELEGRAM_API_BASE_URL` | official API             | e.g. `http://localhost:8081/bot`                            |
| `TELEGRAM_BASE_FILE_URL`| derived                  | e.g. `http://localhost:8081/file/bot`                       |
| `MAX_VIDEO_SIZE`        | 2000 MB                  | largest accepted source video                               |
| `MAX_OUTPUT_SIZE`       | 0                        | cap for the final video (0 = derived from the API mode; for web jobs this is the only cap - no Telegram limit) |
| `TEMP_DIRECTORY`        | `./temp`                 | per-job working directories                                 |
| `OUTPUT_DIRECTORY`      | `./output`               | only used when `SAVE_OUTPUT_COPY=true`                      |
| `MODELS_DIRECTORY`      | `./models`               | separation weights (`models/hub`)                           |
| `MODEL_NAME`            | `htdemucs`               | `htdemucs`, `htdemucs_ft`, `mdx_extra`, …                   |
| `DEMUCS_DEVICE`         | `auto`                   | `auto` → CUDA when available, otherwise CPU                 |
| `DEMUCS_JOBS`           | `0`                      | `0` = auto (½ the CPU cores, max 8; ≈1.6× faster); `1` = lowest RAM |
| `DEMUCS_OVERLAP`        | `0.1`                    | segment crossfade; `0.1` is ~15% faster than Demucs' own `0.25` at the same precision |
| `MAX_OUTPUT_WIDTH/HEIGHT`| `1280` / `720`         | output canvas cap (“720p”); photos come from the source at this size |
| `SLIDESHOW_FPS`         | `10`                     | stills don’t need 30 fps; 10 fps renders ~3× faster         |
| `MAX_CONCURRENT_JOBS`   | `1`                      | jobs processed in parallel                                  |
| `MAX_JOBS_PER_USER`     | `1`                      | jobs one user may queue                                     |
| `JOB_TIMEOUT_SECONDS`   | 10800                    | overall time budget per job                                 |
| `MIN_FREE_DISK_GB`      | 5                        | refuse to start below this much free space                  |
| `AUDIO_LOUDNESS_NORM`   | `true`                   | normalise the vocals to −16 LUFS                            |
| `DELETE_TEMP_AFTER_SEND`| `true`                   | remove the working directory afterwards                     |
| `KEEP_TEMP_ON_ERROR`    | `false`                  | keep it when a job fails (debugging)                        |
| `LOG_LEVEL`             | `INFO`                   | `DEBUG`, `INFO`, `WARNING`, …                               |
| `LOG_FILE`              | *(empty)*                | also write a rotating log file, e.g. `logs/bot.log`         |
| `WEB_USERNAME`          | `admin`                  | login for the web interface                                 |
| `WEB_PASSWORD`          | *(empty)*                | web password; **set it before exposing the page**           |
| `WEB_HOST` / `WEB_PORT` | `0.0.0.0` / `8080`       | listen address of `python -m web.app`                       |
| `WEB_PUBLISH_PORT`      | `80`                     | host port the Docker `web` service is published on          |
| `WEB_JOB_TTL_SECONDS`   | 3600                     | how long a finished ZIP stays downloadable                  |
| `WEB_DOWNLOAD_NAME`     | `150-pictures`           | suggested filename of the web download                      |

## How it works

```
bot/
  telegram_bot.py   application wiring, logging, polling entry point
  handlers.py       per-user state machine (video → overlay → position → size → processing)
  jobs.py           queue, worker pool, throttled status-message editing, delivery
  messages.py       every user-facing string and the progress bar
  file_io.py        streamed downloads/uploads with progress (no full-file RAM)
config/
  settings.py       all environment-driven configuration
  logging_setup.py  console + rotating file logging
processing/
  video_info.py     ffprobe wrapper, output geometry, duration formatting
  frame_extractor.py
                    300 evenly spaced segments → one clean photo each
  face_center.py    Haar-cascade face detection (frontal + profile);
                    face-centred crops for the pair halves (subject/centre
                    fallback when no face is found, never distorts)
  pair_builder.py   photo 1+2 … 299+300 → 150 two-half 16:9 pairs (left/right,
                    both visible for the full 8 s, completely static)
  slideshow.py      concat demuxer → 150 pairs × 8 s H.264 slideshow (overlay baked in)
  audio_extractor.py
                    WAV extraction, silence padding, EBU R128 loudness normalisation
  vocal_separator.py
                    Demucs (--two-stems=vocals) → vocals.wav
  overlay.py        validation, aspect-preserving scaling, positioning,
                    full-screen mode (covers the canvas, near-black keyed
                    to transparent so frame overlays show the video through)
  renderer.py       final mux: copies the picture, encodes the separated vocals
  pipeline.py       runs the five stages, reports progress, cleans up
web/
  app.py            FastAPI app: upload, progress JSON, download, auth gate
  jobs.py           job registry + worker pool behind the browser interface
  static/index.html the whole page (single file, no external assets)
scripts/
  demucs_runner.py  Demucs wrapper (keeps the cache in models/, torch.load shim)
  download_model.py pre-fetch the separation weights
  e2e_check.py      end-to-end self test (see below)
models/             separation weights (git-ignored)
temp/  output/  logs/
```

Notable decisions:

- **Two-stem separation, not a filter.** `demucs --two-stems=vocals` runs a
  Hybrid Transformer model; the `no_vocals` (music) stem is deleted.
- **Demucs, not Spleeter.** Spleeter (Deezer) was evaluated and left out: it
  is unmaintained (TensorFlow 1.x era) and separates less cleanly. Running
  both models would double the runtime without a quality gain - Demucs alone
  is the accuracy leader.
- **Exactly 20:00, always.** If the separated vocals are longer than 1200 s they
  are trimmed; if they are shorter, the rest is padded with *silence*
  (`apad`) - nothing is ever synthesised.
- **Separation never runs on material that is thrown away.** The extracted
  audio is capped at the slideshow length *before* Demucs starts, so sources
  longer than 20 minutes separate proportionally faster with an identical
  result (no quality trade-off).
- **Aspect ratio is never broken.** Each photo is *cropped* to the exact
  aspect of its half (left / right of the 16:9 canvas) - never stretched,
  never letterboxed. The crop window is centred on the primary face (the
  largest detection, profile faces included); without a face the strongest
  image structure becomes the centre. Both photos stay visible side by side
  for the full 8 seconds and are completely static - no zoom, pan or
  animation of any kind. The overlay keeps its own ratio and is clamped to
  the canvas.
- **No file is ever loaded into RAM as a whole** - FFmpeg, Demucs and the
  Telegram transfers all stream.
- **Temp files are removed** after a successful send, after a failure, when a
  job is cancelled and on shutdown.
- **Web results live until they are downloaded.** The finished MP4 stays on
  disk until the download finishes (then it is removed at once), or until
  `WEB_JOB_TTL_SECONDS` pass - whichever comes first - so a slow download can
  never be cut off by cleanup.

## Verifying the pipeline

`scripts/e2e_check.py` builds a synthetic test video (real photos rendered
from `vibe_images/` when present, otherwise generated slides - speech from
FFmpeg's `flite` filter + a music bed), runs the full pipeline and checks the
acceptance criteria: 300 photos, 150 pairs (left = photo 2k-1, right =
photo 2k), 8 s per pair, 1200 s slideshow, per-half face centring without
cut-off, completely static photos, separated vocals, the overlay in the
picture, an MP4 with a video **and** an audio stream, a Telegram-acceptable
size, and a clean temp directory.

```bash
python scripts/e2e_check.py --source-size 1280x720   # full acceptance run
python scripts/e2e_check.py --keep                   # keep the artefacts
```

(Keep `--duration` at the 80 s default: shorter sources can mis-align the
photo boundaries - the script warns when that happens.)

The bot's own acceptance is then just `python -m bot.telegram_bot` (or
`docker compose up`) plus sending a video.

## Troubleshooting

| symptom                                          | fix                                                                                          |
| ------------------------------------------------ | -------------------------------------------------------------------------------------------- |
| `FFmpeg is not installed` at start-up            | install FFmpeg and make sure `ffmpeg`/`ffprobe` are on the `PATH`                             |
| `File is too big to download`                    | the official API caps downloads at 20 MB - use the local Bot API server from `docker-compose.yml` |
| Bot answers `HTTP 413` / `Request Entity Too Large` | the *output* exceeds the upload limit - use the local Bot API server                        |
| Separation is very slow                         | CPU is the bottleneck - `DEMUCS_JOBS=0` (default) already uses half the cores; a bigger server (16 vCPU ≈ 5 min) or a GPU build (`docker-compose.gpu.yml`) is the fix |
| `Only … free` error                              | the job needs roughly 4× the source size; free space or lower `MIN_FREE_DISK_GB`               |
| Bot is silent                                    | `docker compose logs -f bot` - usually a missing/incorrect `TELEGRAM_BOT_TOKEN`                |
| Model download at the first job                  | run `python scripts/download_model.py` once (Docker does it at build time)                     |
| Web page unreachable from the internet           | open the published port (default 80) in the cloud firewall; `docker compose ps` should show `video-web` as healthy |
| Web upload fails with *larger than the limit*    | the video is over `MAX_VIDEO_SIZE` (default 2000 MB) - raise it in `.env` and restart         |
| *Result has already been downloaded* (web)       | normal: a finished video is deleted right after its download; run the job again                |

## Security

The bot token and the API hash are read from the environment only. `.env` is
git-ignored and is never baked into the image - the Telegram API never sees a
credential from the source code.

The bot talks to Telegram via long polling: it only makes *outbound*
connections, so a deployment needs no inbound ports beyond SSH. The bundled
Bot API server binds to `127.0.0.1` and must never be exposed publicly -
anyone who can reach it can act as your bot.

The web interface is meant to be reachable from the internet, so it is
protected by HTTP Basic auth instead: set a strong `WEB_PASSWORD`, and use
HTTPS if the page is exposed publicly (e.g. a reverse proxy such as Caddy or
nginx with a certificate - Basic auth over plain HTTP is readable in
transit). Without `WEB_PASSWORD` the page and every download are open to
anyone who finds the address.

---

Licence: MIT.

#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# One-shot setup for a fresh Ubuntu/Debian VPS (run as root or via sudo user).
#
#   1. upload the bundle to the server (see README, "Deploy on a VPS")
#   2. then, on the server:
#
#        mkdir -p video-bot && tar -xzf video-bot-deploy.tar.gz -C video-bot
#        cd video-bot && bash scripts/vps_setup.sh
#
# The script installs Docker if missing, creates .env from .env.example with
# the credentials you type in, then builds and starts the compose stack.
# ---------------------------------------------------------------------------
set -euo pipefail

say() { printf '\n==> %s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

[[ -f docker-compose.yml && -f .env.example ]] \
    || die "run this from the project root (docker-compose.yml not found)"

# --- docker ----------------------------------------------------------------
if ! command -v docker >/dev/null 2>&1; then
    say "Docker not found - installing it (official convenience script)..."
    curl -fsSL https://get.docker.com | sh
fi

# Make sure the daemon is running (a no-op on a healthy machine).
if command -v systemctl >/dev/null 2>&1; then
    systemctl enable --now docker >/dev/null 2>&1 || true
fi

docker_cmd=(docker)
if ! docker info >/dev/null 2>&1; then
    if [[ "$(id -u)" -eq 0 ]]; then
        die "docker is installed but not responding - try: systemctl start docker"
    fi
    docker_cmd=(sudo docker)   # not in the docker group yet
fi
"${docker_cmd[@]}" compose version >/dev/null 2>&1 \
    || die "the Docker Compose plugin is missing - install the docker-compose-plugin package first"

# --- .env ------------------------------------------------------------------
if [[ -f .env ]]; then
    say "An existing .env was found - keeping it."
else
    say "Creating .env - enter your credentials below."
    read -r -p "Telegram bot token (from @BotFather): " token
    read -r -p "Telegram API ID (from my.telegram.org): " api_id
    read -r -s -p "Telegram API hash (input hidden): " api_hash
    echo
    read -r -s -p "Password for the WEB interface (input hidden, empty = no password): " web_password
    echo
    [[ -n "$token" && -n "$api_id" && -n "$api_hash" ]] \
        || die "all three Telegram values are required"
    [[ "$token" == *:* ]] || say "note: that token does not contain ':' - double-check it"
    [[ -n "$web_password" ]] || say "note: no web password - the page at http://<server>/ will be open to anyone who finds it"
    cp .env.example .env
    sed -i \
        -e "s#^TELEGRAM_BOT_TOKEN=.*#TELEGRAM_BOT_TOKEN=${token}#" \
        -e "s#^TELEGRAM_API_ID=.*#TELEGRAM_API_ID=${api_id}#" \
        -e "s#^TELEGRAM_API_HASH=.*#TELEGRAM_API_HASH=${api_hash}#" \
        .env
    if [[ -n "$web_password" ]]; then
        # '#' and '&' are special to sed - escape them (and '\') in the value.
        escaped=$(printf '%s' "$web_password" | sed -e 's/[&#\\]/\\&/g')
        # The template line is commented out ("#WEB_PASSWORD=..."), so the
        # pattern matches that literal '#' - written as \# because '#' is
        # also the delimiter of this s### command.
        sed -i -e "s#^\#WEB_PASSWORD=.*#WEB_PASSWORD=${escaped}#" .env
    fi
    chmod 600 .env
fi

# --- build and start --------------------------------------------------------
say "Building and starting the stack (first build takes ~5-15 minutes)..."
"${docker_cmd[@]}" compose up -d --build

server_ip=$(hostname -I 2>/dev/null | awk '{print $1}') || server_ip=""
say "Started."
[[ -n "${server_ip:-}" ]] && printf '    Web interface:  http://%s/\n' "$server_ip"
say "Useful commands:"
printf '    %s compose logs -f bot     # live logs\n' "${docker_cmd[*]}"
printf '    %s compose ps              # container status\n' "${docker_cmd[*]}"
printf '    %s compose up -d --build   # apply an update\n' "${docker_cmd[*]}"

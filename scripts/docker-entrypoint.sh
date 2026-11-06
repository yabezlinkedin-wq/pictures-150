#!/bin/sh
# Container entrypoint.
#
# The vocal-separation weights are baked into the image at /opt/model_cache.
# docker-compose mounts the host's ./models over /app/models, which hides the
# baked copy - so on the very first start (empty host directory) we seed it
# from the image.  Later runs reuse whatever is in the volume.
set -e

BAKED_CACHE="${MODEL_CACHE_BAKED:-/opt/model_cache}"
MODELS_PATH="${MODELS_DIRECTORY:-/app/models}"

if [ -d "$BAKED_CACHE" ] && [ -d "$MODELS_PATH" ] && [ -z "$(ls -A "$MODELS_PATH" 2>/dev/null)" ]; then
    echo "entrypoint: seeding $MODELS_PATH from the image's model cache (one time only)"
    cp -a "$BAKED_CACHE/." "$MODELS_PATH/" 2>/dev/null || \
        echo "entrypoint: warning - could not seed the model cache, it will be downloaded on demand"
fi

exec "$@"

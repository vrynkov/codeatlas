#!/bin/sh
# On first boot (no indexed repos yet), pre-index a demo repo so a fresh deployment has
# something to click on immediately instead of an empty rail. This runs at container
# startup, not at build time, so it doesn't slow down the image build and doesn't need
# network access during `docker build`.
set -e

if [ ! -d "data/repos/requests" ]; then
    echo "First run: pre-indexing the demo repo (psf/requests). This can take a minute..."
    python -m app.indexing.indexer index https://github.com/psf/requests \
        || echo "Pre-indexing failed (no GROQ_API_KEY needed for this step, only network access) - you can still index a repo from the UI."
fi

exec "$@"

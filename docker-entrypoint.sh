#!/bin/sh
# Index the demo repo (vrynkov/codeatlas - CodeAtlas itself) in the BACKGROUND, not before starting the server.
# Platforms like Cloud Run health-check the container's port within a tight startup window,
# and this indexing step (downloading the embedding model, then embedding hundreds of code
# chunks) can take minutes and real memory - doing it BEFORE the server starts previously
# made the container fail Cloud Run's startup check entirely. The server now starts right
# away; the demo repo simply appears in the UI a little after boot instead of blocking it.
set -e

if [ ! -d "data/repos/codeatlas" ]; then
    (
        echo "Pre-indexing the demo repo (vrynkov/codeatlas) in the background..."
        python -m app.indexing.indexer index https://github.com/vrynkov/codeatlas \
            || echo "Pre-indexing failed (no GROQ_API_KEY needed for this step, only network access) - you can still index a repo from the UI."
    ) &
fi

exec "$@"

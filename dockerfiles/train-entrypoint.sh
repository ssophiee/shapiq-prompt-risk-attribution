#!/usr/bin/env bash
set -euo pipefail

cd /app

hardware_profile="${TRAIN_HARDWARE_PROFILE:-local}"
commit_stage="${DVC_COMMIT_STAGE:-train_vertex_ddp}"

if [[ "${PUSH_DVC_ON_SUCCESS:-false}" == "true" \
    && "$commit_stage" == "train_vertex_ddp" \
    && "$hardware_profile" != "ddp" ]]; then
    echo "Refusing to commit train_vertex_ddp outputs from hardware profile '$hardware_profile'." >&2
    exit 1
fi

if [[ "${SKIP_DVC_PULL:-false}" != "true" ]]; then
    echo "Pulling DVC training data..."
    /app/.venv/bin/dvc pull \
        data/processed/train.jsonl \
        data/processed/val.jsonl \
        data/processed/test.jsonl
fi

/app/.venv/bin/python \
    -m shapiq_attribution.train \
    "hardware=${hardware_profile}" \
    "$@"

if [[ "${PUSH_DVC_ON_SUCCESS:-false}" == "true" ]]; then
    echo "Committing and pushing DVC training outputs..."
    /app/.venv/bin/dvc commit "$commit_stage" --force
    /app/.venv/bin/dvc push

    if [[ -n "${DVC_METADATA_BUCKET:-}" ]]; then
        echo "Uploading DVC training metadata..."
        /app/.venv/bin/python -c "from google.cloud import storage; import os; client = storage.Client(); bucket = client.bucket(os.environ['DVC_METADATA_BUCKET']); prefix = os.environ.get('DVC_METADATA_PREFIX', 'vertex-ddp/latest').strip('/'); [bucket.blob(f'{prefix}/{path}').upload_from_filename(path) for path in ('dvc.lock', 'reports/metrics.json')]"
    fi
fi

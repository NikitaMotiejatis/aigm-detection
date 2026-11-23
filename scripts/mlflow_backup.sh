#!/usr/bin/env bash
# Machine A (WP3 step 6): snapshot the MLflow server into mlflow-mirror/, then copy it onward.
#
#   BACKUP_DISK=/mnt/e/aigm-detection  copy to the external disk, if that directory exists
#   BACKUP_B=aigm-b                    rsync target host for machine B ("" to skip)
#   BACKUP_B_DIR=aigm-detection        repository path on B, relative to its home
#
# The snapshot replaces the mirror only after it has been fully received and passes an
# integrity check. An unreachable disk or machine B is a warning, not a failure.
set -euo pipefail

server="${BACKUP_SERVER:-root@aigm-mlflow}"
remote_dir="${BACKUP_REMOTE_DIR:-/srv/mlflow}"
mirror="${BACKUP_MIRROR:-mlflow-mirror}"
disk="${BACKUP_DISK:-}"
machine_b="${BACKUP_B-aigm-b}"
b_dir="${BACKUP_B_DIR:-aigm-detection}"

incoming="$mirror.incoming"
rm -rf "$incoming"
mkdir -p "$incoming"
trap 'rm -rf "$incoming"' EXIT

# `.backup` gives a consistent copy while the server is running. tar needs nothing extra there.
ssh "$server" "set -e; t=\$(mktemp -d); trap 'rm -rf \$t' EXIT; \
    sqlite3 '$remote_dir/mlflow.db' \".backup \$t/mlflow.db\"; \
    sqlite3 \"\$t/mlflow.db\" 'PRAGMA integrity_check;' | grep -qx ok; \
    tar -C \"\$t\" -cf - mlflow.db -C '$remote_dir' artifacts" | tar -C "$incoming" -xf -

test -s "$incoming/mlflow.db"
test -d "$incoming/artifacts"

rm -rf "$mirror.old"
if [ -e "$mirror" ]; then mv "$mirror" "$mirror.old"; fi
mv "$incoming" "$mirror"
rm -rf "$mirror.old"
echo "snapshot: $mirror"

if [ -n "$disk" ] && [ -d "$disk" ]; then
    rsync -rt --delete "$mirror/" "$disk/mlflow-mirror/" && echo "copied to $disk" \
        || echo "WARNING: copy to $disk failed" >&2
elif [ -n "$disk" ]; then
    echo "WARNING: $disk is not mounted, skipped" >&2
fi

if [ -n "$machine_b" ]; then
    rsync -rt --delete -e "ssh -o ConnectTimeout=10" "$mirror/" \
        "$machine_b:$b_dir/mlflow-mirror/" && echo "copied to $machine_b" \
        || echo "WARNING: $machine_b not reachable, skipped" >&2
fi

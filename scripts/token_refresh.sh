#!/bin/bash
# Continuously refresh the auth token every 45 minutes.
# Run this in a tmux/screen session as the proid:
#
#   suu -tr "<ticket>" <proid>
#   ./token_refresh.sh &
#   exit  # DON'T exit — keep this session alive
#
# Or in a separate tmux pane:
#   tmux new-session -d -s token './token_refresh.sh'

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${ALPHALAB_PYTHON:-python3}"
INTERVAL=2700  # 45 minutes (token lasts 60 min, refresh with 15 min buffer)

echo "Token refresh daemon started (every ${INTERVAL}s / 45min)"
echo "Press Ctrl+C to stop"

while true; do
    echo ""
    echo "$(date): Refreshing token..."
    "$PYTHON" "$SCRIPT_DIR/prefetch_token.py"
    if [ $? -eq 0 ]; then
        echo "$(date): Token refreshed. Next refresh in ${INTERVAL}s."
    else
        echo "$(date): WARNING — Token refresh failed! Retrying in 60s..."
        sleep 60
        continue
    fi
    sleep $INTERVAL
done

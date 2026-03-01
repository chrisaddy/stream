#!/bin/bash
# Usage: ./record.sh <section_name> <duration_seconds>
# Records screen capture to demo/clips/<section_name>.mp4

SECTION=$1
DURATION=$2
OUTDIR="$(dirname "$0")/clips"

mkdir -p "$OUTDIR"

echo "Recording $SECTION for ${DURATION}s..."
ffmpeg -y -f avfoundation -framerate 30 -capture_cursor 1 -i "4:none" \
  -t "$DURATION" \
  -c:v libx264 -pix_fmt yuv420p -preset ultrafast -crf 18 \
  "$OUTDIR/${SECTION}.mp4" 2>/dev/null

echo "Saved: $OUTDIR/${SECTION}.mp4"

#!/usr/bin/env sh
# Fetch the 2-minute example soundscape that ships with the BirdNET-Analyzer
# repo, so you can try Dawn Chorus before you have a recording of your own.
# (We don't redistribute it: it belongs to the BirdNET project, CC BY-NC-SA 4.0.)
set -e
mkdir -p "$(dirname "$0")/../examples"
curl -fsSL -o "$(dirname "$0")/../examples/soundscape.wav" \
  https://raw.githubusercontent.com/kahst/BirdNET-Analyzer/main/birdnet_analyzer/example/soundscape.wav
echo "saved examples/soundscape.wav"

#!/usr/bin/env bash
# Look at the nacelle.
#   ./view.sh --cramera [scene]        CRAMERA in the browser (default scene: windturbine)
#                                      scenes: windturbine_<scenario>, e.g. ..._after_maintenance
#   ./view.sh                          Blender, healthy model
#   ./view.sh urdf/scenarios/x.urdf    Blender, a fault scenario
#   ./view.sh --rviz [urdf]            RViz with joint sliders (needs ROS 2 Jazzy)
set -eo pipefail
cd "$(dirname "$0")"
CRAMERA_REPO="${CRAMERA_REPO:-$HOME/workspace/cramera-port}"
PORT="${CRAMERA_PORT:-8711}"
if [ "$1" = "--cramera" ]; then
  scene="${2:-windturbine}"
  python3 scripts/build_cramera_bundle.py
  url="http://localhost:$PORT/?scene=$scene"
  if curl -s -o /dev/null "http://localhost:$PORT/scenes/index.json"; then
    echo "viewer already running: $url"
    xdg-open "$url" >/dev/null 2>&1 || true
    exit 0
  fi
  (sleep 2; xdg-open "$url" >/dev/null 2>&1 || true) &
  echo "viewer: $url  (Ctrl-C to quit)"
  exec "$CRAMERA_REPO/.venv/bin/cramera" --no-browser "$PORT"
fi
if [ "$1" = "--rviz" ]; then
  shift
  urdf=$(realpath "${1:-urdf/windturbine.urdf}")
  source /opt/ros/jazzy/setup.bash
  if [ ! -f install/setup.bash ]; then
    colcon build --symlink-install --base-paths . --packages-select windturbine_model
  fi
  source install/setup.bash
  exec ros2 launch windturbine_model display.launch.py urdf:="$urdf"
fi
urdf="${1:-urdf/windturbine.urdf}"
shift || true
exec blender --factory-startup -P scripts/render_preview.py -- "$urdf" --view "$@"

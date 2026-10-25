#!/bin/bash
# Build the Flightmare Unity renderer (camera images) as a native macOS app and
# install it where envsim / flightgym expect it: $FLIGHTMARE_PATH/flightrender/.
# Needs Unity Editor 2021.3 (Hub, signed in with an activated license).
# Run inside the workspace env:  pixi run build-renderer
set -euo pipefail

WS="$PIXI_PROJECT_ROOT"
UNITY_VERSION="${UNITY_VERSION:-2021.3.45f2}"
UNITY="${UNITY:-/Applications/Unity/Hub/Editor/$UNITY_VERSION/Unity.app/Contents/MacOS/Unity}"
PROJECT="${FLIGHTMARE_UNITY_PROJECT:-$WS/unity/flightmare_unity}"
RENDER_DIR="$FLIGHTMARE_PATH/flightrender"
APP="$RENDER_DIR/RPG_Flightmare.app"

if [ ! -x "$UNITY" ]; then
  echo "Unity $UNITY_VERSION not found at: $UNITY"
  echo "Install it with Unity Hub (or set UNITY=/path/to/Unity), then sign in and activate a license."
  exit 1
fi

if [ ! -d "$PROJECT/.git" ]; then
  echo ">>> Cloning uzh-rpg/flightmare_unity (dev/agile_flight)"
  mkdir -p "$(dirname "$PROJECT")"
  git clone --branch dev/agile_flight --single-branch \
    https://github.com/uzh-rpg/flightmare_unity.git "$PROJECT"
fi

# CameraController.cs uses Newtonsoft.Json, which 2021.2 only got indirectly via
# collab-proxy 1.x; the 2021.3 upgrade drops it, so declare it explicitly.
python - "$PROJECT/Packages/manifest.json" <<'EOF'
import json, sys
path = sys.argv[1]
manifest = json.load(open(path))
deps = manifest["dependencies"]
if "com.unity.nuget.newtonsoft-json" not in deps:
    deps["com.unity.nuget.newtonsoft-json"] = "3.2.1"
    json.dump(manifest, open(path, "w"), indent=2)
    print(">>> Added com.unity.nuget.newtonsoft-json to Packages/manifest.json")
EOF

# Racing scenes (renderer/RaceScenes.cs). flightlib accepts scene ids < 5
# only, so they take the slots of upstream scenes that are not public:
# 0 warehouse (was Warehouse), 2 garage (was Mountain), and 4 arena (new).
# 1 is Industrial, 3 Wasteland (not public).
python - "$PROJECT/Assets/Flightmare/Scripts/sceneSchedule.cs" <<'EOF'
import sys
path = sys.argv[1]
src = open(path, encoding="utf-8-sig").read()
new = src
for old in ("Arena/Arena", "RaceWarehouse/RaceWarehouse", "RaceGarage/RaceGarage"):
    new = new.replace('      scenes_list.Add("Environments/%s");\n' % old, "")
new = new.replace("      scenes_list.Add(SCENE_WAREHOUSE);\n",
                  '      scenes_list.Add("Environments/RaceWarehouse/RaceWarehouse");  // macOS: generated\n')
new = new.replace("      scenes_list.Add(SCENE_MOUNTAIN);\n",
                  '      scenes_list.Add("Environments/RaceGarage/RaceGarage");  // macOS: generated\n')
new = new.replace("      scenes_list.Add(SCENE_WASTELAND);\n",
                  "      scenes_list.Add(SCENE_WASTELAND);\n"
                  '      scenes_list.Add("Environments/Arena/Arena");  // macOS: generated\n')
if new != src:
    open(path, "w", encoding="utf-8-sig").write(new)
    print(">>> Registered the race scenes in sceneSchedule.cs (0 warehouse, 2 garage, 4 arena)")
EOF

# The gate prefabs carry a non-kinematic Rigidbody (rpg_gate even with
# gravity), so gates fall or get knocked away by the drone model and drift.
# Make them kinematic: gates stay where the track puts them.
for prefab in "$PROJECT/Assets/Resources/gate_default.prefab" \
              "$PROJECT/Assets/Flightmare/Templates/RPGGate/rpg_gate.prefab"; do
  sed -i '' 's/m_IsKinematic: 0/m_IsKinematic: 1/' "$prefab"
done

mkdir -p "$PROJECT/Assets/Editor"
rm -f "$PROJECT/Assets/Editor/RaceArena.cs" "$PROJECT/Assets/Editor/RaceArena.cs.meta"
cp "$WS/renderer/BuildMacOS.cs" "$WS/renderer/RaceScenes.cs" "$PROJECT/Assets/Editor/"

echo ">>> Building $APP (first run imports all assets and can take a long time)"
LOG="$WS/logs/unity_build.log"
mkdir -p "$(dirname "$LOG")"
if ! "$UNITY" -batchmode -nographics -quit \
    -projectPath "$PROJECT" \
    -executeMethod BuildMacOS.Build \
    -buildOutput "$APP" \
    -logFile "$LOG"; then
  echo "Unity build failed, see $LOG"
  grep -E "error CS|Error|License" "$LOG" | tail -20 || true
  exit 1
fi
grep "\[BuildMacOS\]" "$LOG" || true

# envsim's launch file and flightgym's demo start "RPG_Flightmare.x86_64" from this
# folder; on macOS that name is a small launcher for the .app.
cat > "$RENDER_DIR/RPG_Flightmare.x86_64" <<'EOF'
#!/bin/bash
# macOS launcher for the Flightmare Unity renderer (see agile-flight-macos)
exec "$(dirname "$0")/RPG_Flightmare.app/Contents/MacOS/RPG_Flightmare" "$@"
EOF
chmod +x "$RENDER_DIR/RPG_Flightmare.x86_64"

echo ">>> Done. Run the simulator with images: pixi run demo --render"

import os
import json
from collections import defaultdict

# Ruta base on es troben les seqüències
base_path = "/data-fast/data-server/ccorbi/SoccerNetGS/gamestate-2024/valid"
output_path = "/data-fast/data-server/ccorbi/SN-Tracking/tracking/val"

for seq_name in os.listdir(base_path):
    seq_path = os.path.join(base_path, seq_name)
    json_path = os.path.join(seq_path, "Labels-GameState.json")

    if not os.path.isfile(json_path):
        continue

    # Carregar anotacions
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Agrupar informació per tracklet_id
    tracklet_info = {}
    for ann in data.get("annotations", []):
        track_id = ann.get("track_id")
        if track_id is None:
            continue
        if track_id in tracklet_info:
            continue  # ja tenim la info d'aquest tracklet

        attributes = ann.get("attributes", {})
        role = attributes.get("role")
        team = attributes.get("team", "")
        jersey = attributes.get("jersey", "")

        if role == "player":
            if jersey:
                j_str = jersey
            else:
                j_str = "Y" if team == "left" else "X"
            role_str = f"player team {team};{j_str}"
        elif role == "goalkeeper":
            j_str = jersey if jersey else ("y" if team == "left" else "X")
            role_str = f"goalkeeper team {team};{j_str}"
        elif role == "referee":
            # Intenta distingir quin tipus de referee
            referee_type = attributes.get("refereeType", "unknown").lower()
            if "main" in referee_type:
                role_str = "referee;main"
            elif "side top" in referee_type:
                role_str = "referee;side top"
            elif "side bottom" in referee_type:
                role_str = "referee;side bottom"
            else:
                role_str = "referee;unknown"
        elif role == "ball":
            role_str = "ball;1"
        else:
            continue  # saltem rols desconeguts

        tracklet_info[track_id] = role_str

    # Crear fitxer gameinfo.ini
    seq_output_path = os.path.join(output_path, seq_name)
    os.makedirs(seq_output_path, exist_ok=True)
    ini_path = os.path.join(seq_output_path, "gameinfo.ini")

    with open(ini_path, "w") as ini_file:
        ini_file.write("[Sequence]\n")
        ini_file.write(f"name={seq_name}\n\n")
        ini_file.write(f"num_tracklets={len(tracklet_info)}\n")

        # Assignar un trackletID_1, trackletID_2... ordenats per track_id
        for i, (track_id, role_str) in enumerate(sorted(tracklet_info.items()), start=1):
            ini_file.write(f"trackletID_{i}={role_str}\n")

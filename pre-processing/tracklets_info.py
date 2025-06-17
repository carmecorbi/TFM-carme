import os
from collections import defaultdict

# Path base de training
base_path = "/data-fast/data-server/ccorbi/SN-Tracking/tracking/train"

# Iterar per cada seqüència
for seq_name in os.listdir(base_path):
    seq_path = os.path.join(base_path, seq_name)
    gt_path = os.path.join(seq_path, "gt", "gt.txt")
    gameinfo_path = os.path.join(seq_path, "gameinfo.ini")
    output_ini_path = os.path.join(seq_path, "tracklets_info.ini")

    if not os.path.isfile(gt_path) or not os.path.isfile(gameinfo_path):
        print(f"✗ Falten fitxers a: {seq_name}")
        continue

    # Map trackletID_x -> (tracklet_index, description)
    tracklet_map = {}
    for line in open(gameinfo_path, "r"):
        line = line.strip()
        if line.startswith("trackletID_"):
            key, value = line.split("=")
            tracklet_id = int(key.split("_")[1])
            tracklet_map[tracklet_id] = (key, value.strip())

    # Agrupar frames per track_id
    track_frames = defaultdict(list)
    for line in open(gt_path, "r"):
        parts = line.strip().split(",")
        if len(parts) < 2:
            continue
        frame_id = int(parts[0])
        track_id = int(parts[1])
        track_frames[track_id].append(frame_id)

    # Escriure tracklets_info.ini
    with open(output_ini_path, "w") as out_f:
        out_f.write("[Tracklets]\n")
        for track_id, frames in sorted(track_frames.items()):
            if track_id not in tracklet_map:
                continue
            tracklet_key, description = tracklet_map[track_id]
            frames = sorted(frames)
            start = frames[0]
            end = frames[-1]
            duration = len(frames)

            # Buscar gaps (discontinuïtats)
            gaps = []
            for i in range(1, len(frames)):
                if frames[i] != frames[i - 1] + 1:
                    gap_start = frames[i - 1] + 1
                    gap_end = frames[i] - 1
                    gaps.append((gap_start, gap_end))

            # Format dels gaps: "missing: 14–17; 100–102"
            if gaps:
                gaps_str = "; ".join(f"{g[0]}–{g[1]}" for g in gaps)
                out_f.write(f"{tracklet_key}={description} | frames: {duration}, start: {start}, end: {end}, missing: {gaps_str}\n")
            else:
                out_f.write(f"{tracklet_key}={description} | frames: {duration}, start: {start}, end: {end}\n")

    print(f"✓ Creat tracklets_info.ini amb gaps per {seq_name}")

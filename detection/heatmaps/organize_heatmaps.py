import os
import shutil

origen_base = "/home-net/ccorbi/detection/heatmaps/heatmap_val"
destinacio_base = "/data-fast/data-server/ccorbi/ball/heatmaps/val"

seq_list = [seq for seq in os.listdir(origen_base) if os.path.isdir(os.path.join(origen_base, seq))]

for seq in seq_list:
    left_path = os.path.join(origen_base, seq, "left")
    right_path = os.path.join(origen_base, seq, "right")

    for frame_num in range(6, 751):
        frame_str = f"{frame_num:06d}"
        dest_frame_path = os.path.join(destinacio_base, seq, frame_str)
        dest_left_path = os.path.join(dest_frame_path, "left")
        dest_right_path = os.path.join(dest_frame_path, "right")

        os.makedirs(dest_left_path, exist_ok=True)
        os.makedirs(dest_right_path, exist_ok=True)

        # Copiar els 5 frames anteriors al frame actual
        for prev_num in range(frame_num - 5, frame_num):
            prev_str = f"{prev_num:06d}.png"

            orig_left_file = os.path.join(left_path, prev_str)
            dest_left_file = os.path.join(dest_left_path, prev_str)
            if os.path.exists(orig_left_file):
                shutil.copy2(orig_left_file, dest_left_file)
            else:
                print(f"No existeix {orig_left_file}")

            orig_right_file = os.path.join(right_path, prev_str)
            dest_right_file = os.path.join(dest_right_path, prev_str)
            if os.path.exists(orig_right_file):
                shutil.copy2(orig_right_file, dest_right_file)
            else:
                print(f"No existeix {orig_right_file}")

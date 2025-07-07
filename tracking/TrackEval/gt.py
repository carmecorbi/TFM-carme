import os
import shutil

# Origen de les seqüències amb gt
src_base = "/data-fast/data-server/ccorbi/SN-Tracking/tracking/test"
# Destí on TrackEval espera les gt
dst_base = "/home-net/ccorbi/tracking/TrackEval/data/gt/mot_challenge/SoccerNet"
# Carpeta on posar el fitxer seqmap
seqmap_dir = "/home-net/ccorbi/tracking/TrackEval/data/gt/mot_challenge/seqmaps"
os.makedirs(dst_base, exist_ok=True)
os.makedirs(seqmap_dir, exist_ok=True)

# Llista per guardar els noms de les seqüències correctament copiades
copied_seqs = []

for seq_name in sorted(os.listdir(src_base)):
    src_seq_path = os.path.join(src_base, seq_name)
    if not os.path.isdir(src_seq_path):
        continue

    src_gt = os.path.join(src_seq_path, "gt", "gt.txt")
    src_seqinfo = os.path.join(src_seq_path, "seqinfo.ini")
    if not (os.path.exists(src_gt) and os.path.exists(src_seqinfo)):
        print(f"Saltant {seq_name}: falta gt.txt o seqinfo.ini")
        continue

    # Copiar fitxers al destí
    dst_seq_path = os.path.join(dst_base, seq_name)
    dst_gt_dir = os.path.join(dst_seq_path, "gt")
    os.makedirs(dst_gt_dir, exist_ok=True)

    shutil.copy2(src_gt, os.path.join(dst_gt_dir, "gt.txt"))
    shutil.copy2(src_seqinfo, os.path.join(dst_seq_path, "seqinfo.ini"))

    copied_seqs.append(seq_name)
    print(f"✔ Copiat: {seq_name}")

# Crear fitxer SoccerNet-test.txt amb totes les seqüències copiades
seqmap_test_path = os.path.join(seqmap_dir, "SoccerNet-test.txt")
with open(seqmap_test_path, "w") as f:
    f.write("name\n")
    for seq in copied_seqs:
        f.write(f"{seq}\n")

print(f"\n✅ Fitxer SoccerNet-test.txt creat amb {len(copied_seqs)} seqüències.")
print(f"📄 Ruta: {seqmap_test_path}")


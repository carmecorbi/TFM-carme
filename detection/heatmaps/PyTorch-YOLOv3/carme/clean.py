import os

root_folder = "/data-fast/data-server/ccorbi/ball/labels/val"

for subdir, _, files in os.walk(root_folder):
    for filename in files:
        if filename.endswith(".txt"):
            file_path = os.path.join(subdir, filename)
            new_lines = []

            with open(file_path, "r") as f:
                for line in f:
                    stripped = line.strip()
                    if stripped.startswith("1 "):
                        parts = stripped.split()
                        parts[0] = "32"
                        new_lines.append(" ".join(parts))

            with open(file_path, "w") as f:
                if new_lines:
                    f.write("\n".join(new_lines) + "\n")
                else:
                    f.write("")  # Es deixa el fitxer buit, no s'elimina

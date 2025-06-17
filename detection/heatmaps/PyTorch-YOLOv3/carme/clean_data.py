import os

base_path = "/data-fast/data-server/ccorbi/ball/images/test"

seq_list = [seq for seq in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, seq))]

for seq in seq_list:
    seq_path = os.path.join(base_path, seq)
    
    for num in range(1, 6):  # només del 000001 al 000005
        filename = f"{num:06d}.jpg"
        file_path = os.path.join(seq_path, filename)
        
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Eliminat: {file_path}")
        else:
            print(f"No existeix: {file_path}")

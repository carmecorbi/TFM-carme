import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np


class SoccerDataset(Dataset):
    def __init__(self, image_root, label_root, heatmap_root, sequences, transform=None):
        self.image_root = image_root  #Path images
        self.label_root = label_root  # Path labels 
        self.heatmap_root = heatmap_root #Path heatmap
        self.sequences = sequences  # ['name_seq1', 'name_seq2', ...]
        self.transform = transform or transforms.ToTensor()

        self.samples = []
        for seq in sequences:
            img_dir = os.path.join(image_root, seq)
            frames = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
            for frame in frames:
                self.samples.append((seq, frame))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq, frame = self.samples[idx]  #Id sequence and frame
        frame_id = os.path.splitext(frame)[0]  # ex: "000123"

        # --- Load RGB image ---
        img_path = os.path.join(self.image_root, seq, frame)
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)  # (3, H, W)

        # --- Load YOLO label file ---
        label_path = os.path.join(self.label_root, seq, f"{frame_id}.txt")
        labels = self.load_labels(label_path)  # (M, 5)

        # --- Load heatmaps (left and right separately) ---
        base_hmap_path = os.path.join(self.heatmap_root, seq, frame_id)

        heatmaps_left = self.load_heatmaps_from_dir(os.path.join(base_hmap_path, 'left'))
        heatmaps_right = self.load_heatmaps_from_dir(os.path.join(base_hmap_path, 'right'))

        return image, heatmaps_left, heatmaps_right, labels

    def load_labels(self, label_path):
        if not os.path.exists(label_path):
            return torch.zeros((0, 5))  # no deteccions
        with open(label_path, 'r') as f:
            lines = f.readlines()
        labels = [list(map(float, line.strip().split())) for line in lines]
        return torch.tensor(labels)  # (M, 5)

    def load_heatmaps_from_dir(self, dir_path):
        heatmap_files = sorted([f for f in os.listdir(dir_path) if f.endswith(('.png', '.jpg'))])
        maps = []
        for f in heatmap_files:
            hmap_path = os.path.join(dir_path, f)
            hmap = Image.open(hmap_path).convert('L')  # grayscale
            hmap_tensor = self.transform(hmap)  # (1, H, W)
            maps.append(hmap_tensor)
        if len(maps) == 0:
            # cap heatmap? retorna tensor buit
            return torch.zeros((0, *self.transform(Image.new('L', (1, 1))).shape[1:]))
        return torch.cat(maps, dim=0)  # (N, H, W)

    @staticmethod
    def collate_fn(batch):
        images, heatmaps_left, heatmaps_right, labels = zip(*batch)
        
        # Stack images
        images = torch.stack(images)
        
        # Stack heatmaps: cada heatmap és (N, H, W), vols (batch, N, H, W)
        heatmaps_left = torch.stack(heatmaps_left)
        heatmaps_right = torch.stack(heatmaps_right)
        
        # Labels poden tenir nombre variable de deteccions, afegim batch_index
        for i, label in enumerate(labels):
            if label.numel() > 0:  # si no està buit
                label[:, 0] = i  # primer camp és batch index
        
        labels = torch.cat(labels, dim=0)
        
        return images, heatmaps_left, heatmaps_right, labels


if __name__ == '__main__':
    image_root = '/data-fast/data-server/ccorbi/ball/images/train'
    label_root = '/data-fast/data-server/ccorbi/ball/labels/train'
    heatmap_root = '/data-fast/data-server/ccorbi/ball/heatmaps/train'
    sequences = ['SNMOT-060','SNMOT-073']  # Posa una seqüència real

    # Crea el dataset
    dataset = SoccerDataset(
        image_root=image_root,
        label_root=label_root,
        heatmap_root=heatmap_root,
        sequences=sequences,
        transform=transforms.ToTensor()
    )

    print(f"Nombre total de mostres: {len(dataset)}")

    # Prova de carregar una mostra
    image, heatmaps_left, heatmaps_right, labels = dataset[0]

    # Mostra informació de les formes
    print("Image shape:", image.shape)  # (3, H, W)
    print("Heatmaps LEFT shape:", heatmaps_left.shape)  # (N, H, W)
    print("Heatmaps RIGHT shape:", heatmaps_right.shape)  # (N, H, W)
    print("Labels shape:", labels.shape)  # (M, 5)
    print("Labels:", labels)

    dataloader = DataLoader(dataset, batch_size=16, collate_fn = dataset.collate_fn)

    num_batches = (len(dataset) + 16 - 1) // 16  # càlcul amb "ceil" enter

    print(f"Nombre total de mostres: {len(dataset)}")
    print(f"Nombre total de batches: {num_batches}")

    # Prova de carregar un batch
    for batch in dataloader:
        images, heatmaps_left, heatmaps_right, labels = batch
        print("Batch d'imatges:", images.shape)        # (B, 3, H, W)
        print("Batch heatmaps LEFT:", heatmaps_left.shape)   # (B, N_left, H, W)
        print("Batch heatmaps RIGHT:", heatmaps_right.shape) # (B, N_right, H, W)
        print("Batch labels:", labels.shape)            # (total deteccions en batch, 5)
        print("Primeres 5 labels:\n", labels[:16])
        break  # Només provar el primer batch
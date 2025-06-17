from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import glob
import random
import os
import warnings
import numpy as np
from PIL import Image
from PIL import ImageFile
from pytorchyolo.utils.augmentations import AUGMENTATION_TRANSFORMS
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import DataLoader
from pytorchyolo.utils.utils import to_cpu, load_classes, print_environment_info, worker_seed_set
from torchvision import transforms
import torchvision.utils as vutils
from pytorchyolo.utils.transforms import ToTensor, PadSquare, RelativeLabels, AbsoluteLabels, ImgAug

def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


class ImageFolder(Dataset):
    def __init__(self, folder_path, transform=None):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.transform = transform

    def __getitem__(self, index):

        img_path = self.files[index % len(self.files)]
        img = np.array(
            Image.open(img_path).convert('RGB'),
            dtype=np.uint8)

        # Label Placeholder
        boxes = np.zeros((1, 5))

        # Apply transforms
        if self.transform:
            img, _ = self.transform((img, boxes))

        return img_path, img

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    def __init__(self, list_path, img_size=416, multiscale=True, transform=None):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()

        self.label_files = []
        for path in self.img_files:
            image_dir = os.path.dirname(path)
            #print("image_dir",image_dir)
            label_dir = "labels".join(image_dir.rsplit("images", 1))
            #print("label_dir",label_dir)
            assert label_dir != image_dir, \
                f"Image path must contain a folder named 'images'! \n'{image_dir}'"
            label_file = os.path.join(label_dir, os.path.basename(path))
            label_file = os.path.splitext(label_file)[0] + '.txt'
            self.label_files.append(label_file)

        self.img_size = img_size
        self.max_objects = 100
        self.multiscale = multiscale
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0
        self.transform = transform

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------
        try:

            img_path = self.img_files[index % len(self.img_files)].rstrip()

            img = np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8)
        except Exception:
            print(f"Could not read image '{img_path}'.")
            return 

        # ---------
        #  Label
        # ---------
        try:
            label_path = self.label_files[index % len(self.img_files)].rstrip()

            # Ignore warning if file is empty
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                boxes = np.loadtxt(label_path).reshape(-1, 5)
        except Exception:
            print(f"Could not read label '{label_path}'.")
            return

        # -----------
        #  Transform
        # -----------
        if self.transform:
            try:
                img, bb_targets = self.transform((img, boxes))
            except Exception:
                print("Could not apply transform.")
                return
        #print(f"[OK] {img_path} loaded. Img shape: {img.shape}, Targets shape: {bb_targets.shape}")
        return img_path, img, bb_targets

    def collate_fn(self, batch):
        self.batch_count += 1

        # Drop invalid images
        batch = [data for data in batch if data is not None]
        if len(batch) == 0:
            print(f"[WARNING] Batch buit detectat a batch #{self.batch_count}")
            raise ValueError(f"[ERROR] Batch buit detectat a batch #{self.batch_count}")

        paths, imgs, bb_targets = list(zip(*batch))

        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(
                range(self.min_size, self.max_size + 1, 32))

        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])

        # Add sample index to targets
        for i, boxes in enumerate(bb_targets):
            boxes[:, 0] = i
        bb_targets = torch.cat(bb_targets, 0)

        return paths, imgs, bb_targets

    def __len__(self):
        return len(self.img_files)


class PadSquareHeatmap(object):
    def __call__(self, img):
        # Assume img is a PIL Image, not np.array
        w, h = img.size
        dim_diff = np.abs(h - w)
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2

        if h < w:
            padding = (0, pad1, 0, pad2)
        else:
            padding = (pad1, 0, pad2, 0)

        return transforms.functional.pad(img, padding, fill=0, padding_mode='constant')

class OwnDataset(Dataset):
     def __init__(self, list_path, img_size=416, multiscale=True, transform=None, base_hmap_path=None):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()

        self.label_files = []
        for path in self.img_files:
            image_dir = os.path.dirname(path)
            label_dir = "labels".join(image_dir.rsplit("images", 1))
            assert label_dir != image_dir, \
                f"Image path must contain a folder named 'images'! \n'{image_dir}'"
            label_file = os.path.join(label_dir, os.path.basename(path))
            label_file = os.path.splitext(label_file)[0] + '.txt'
            self.label_files.append(label_file)

        self.img_size = img_size
        self.max_objects = 100
        self.multiscale = multiscale
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0
        self.transform = transform
        self.base_hmap_path = base_hmap_path  # heatmap base path

     def load_heatmaps_from_dir(self, dir_path, center_frame):
        """
        Load 5 heatmaps before `center_frame` (e.g. 000006 → 000001 to 000005)
        """
        maps = []
        center_idx = int(center_frame)
        for i in range(center_idx - 5, center_idx):
            fname = f"{i:06d}.png"  # zero-padded
            hmap_path = os.path.join(dir_path, fname)
            #print(hmap_path)
            if os.path.exists(hmap_path):
                hmap = Image.open(hmap_path)
                transform = transforms.Compose([
                    PadSquareHeatmap(),
                    transforms.Resize((self.img_size, self.img_size)) ,
                    transforms.ToTensor()
                ])
                hmap_tensor = transform(hmap) 
            
            maps.append(hmap_tensor)
        return torch.stack(maps)  # shape: [5, 1, H, W]

     def __getitem__(self, index):
            try:
                img_path = self.img_files[index % len(self.img_files)].rstrip()
                #print(f"[INFO] Carregant imatge: {img_path}")
                img = np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8)
            except Exception:
                return

            try:
                label_path = self.label_files[index % len(self.img_files)].rstrip()
                #print(f"[INFO] Carregant etiquetes: {label_path}")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    boxes = np.loadtxt(label_path).reshape(-1, 5)
            except Exception:
                return

            # Transform image + boxes
            if self.transform:
                try:
                    img, bb_targets = self.transform((img, boxes))
                except Exception:
                    return

            if self.base_hmap_path:
                img_path_clean = img_path.rstrip()
                rel_path = img_path_clean.split("/images/")[1]     # train/SNGS-026/000006.jpg
                folder_path = os.path.splitext(rel_path)[0]        # train/SNGS-026/000006
                hmap_folder = os.path.join(self.base_hmap_path, folder_path)

                frame_id = os.path.basename(folder_path)  # "000006"
                left_path = os.path.join(hmap_folder, 'left')
                right_path = os.path.join(hmap_folder, 'right')

                #print(f"[INFO] Heatmaps LEFT des de: {left_path}")
                #print(f"[INFO] Heatmaps RIGHT des de: {right_path}")

                heatmap_left = self.load_heatmaps_from_dir(os.path.join(hmap_folder, 'left'), frame_id)
                heatmap_right = self.load_heatmaps_from_dir(os.path.join(hmap_folder, 'right'), frame_id)

            return img_path, img, bb_targets, heatmap_left, heatmap_right

          

        

     def collate_fn(self, batch):
            self.batch_count += 1
            batch = [data for data in batch if data is not None]

            paths, imgs, bb_targets, heatmap_lefts, heatmap_rights = list(zip(*batch))

            if self.multiscale and self.batch_count % 10 == 0:
                self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))

            imgs = torch.stack([resize(img, self.img_size) for img in imgs])

            for i, boxes in enumerate(bb_targets):
                boxes[:, 0] = i
            bb_targets = torch.cat(bb_targets, 0)

            # You might want to resize heatmaps too
            heatmap_lefts = torch.stack(heatmap_lefts)
            heatmap_rights = torch.stack(heatmap_rights)

            return paths, imgs, bb_targets, heatmap_lefts, heatmap_rights
     def __len__(self):
            return len(self.img_files)

'''

AUGMENTATION = transforms.Compose([
    AbsoluteLabels(),
    PadSquare(),
    RelativeLabels(),
    ToTensor(),
])

dataset = OwnDataset(
    list_path='/data-fast/data-server/ccorbi/ball/images/train_ball.txt',
    img_size=608,
    multiscale=False,
    transform=AUGMENTATION,  # o None si vols sense augmentació
    base_hmap_path='/data-fast/data-server/ccorbi/ball/heatmaps'
)

# 3. Accedeix a un sample concret (per exemple, el primer)
sample_index = 0  # canvia-ho si vols provar un altre
data = dataset[sample_index]

# 4. Desem la imatge i un heatmap per comprovar que tot va bé
if data is not None:
    img_path, img, bb_targets, heatmap_left, heatmap_right = data

    os.makedirs("debug_output", exist_ok=True)

    # Guarda la imatge transformada
    vutils.save_image(img, f"debug_output/img_{sample_index}.png")

    # Guarda un heatmap de l'esquerra (primer dels 5)
    vutils.save_image(heatmap_left[1], f"debug_output/heatmap_left_{sample_index}.png")

    # Opcional: heatmap dret
    vutils.save_image(heatmap_right[1], f"debug_output/heatmap_right_{sample_index}.png")

    print(f"Imatge i heatmaps guardats a 'debug_output/' per index {sample_index}")
else:
    print(f"No s'ha pogut carregar el sample amb index {sample_index}")
'''
'''
dataset = OwnDataset(
    list_path='/data-fast/data-server/ccorbi/ball/images/train_ball.txt',
    img_size=608,
    multiscale=False,
    transform=AUGMENTATION_TRANSFORMS,  # posa aquí la teva funció o deixa None
    base_hmap_path='/data-fast/data-server/ccorbi/ball/heatmaps'
)

batch_size =  64 // 16
dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        collate_fn=dataset.collate_fn,
        worker_init_fn=worker_seed_set)


for batch in dataloader:
    paths, imgs, bb_targets, heatmap_lefts, heatmap_rights = batch
    print("imgs.shape:", imgs.shape)
    print("bb_targets.shape:", bb_targets.shape)
    print("heatmap_lefts.shape:", heatmap_lefts.shape)
    print("heatmap_rights.shape:", heatmap_rights.shape)
    break  # només el primer batc

'''
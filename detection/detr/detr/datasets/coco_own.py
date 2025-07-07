import os
import torch
import torchvision
from pathlib import Path
import configparser
from pycocotools import mask as coco_mask
import sys
#detr_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
#if detr_path not in sys.path:
 #   sys.path.insert(0, detr_path)

import datasets.transforms as T

import sys
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import torchvision.transforms as T2

def draw_boxes_with_players_and_team(dataset, split_name="train", num_images=5, img_width=1920, img_height=1080):
    os.makedirs(f"debug_images/{split_name}_boxes", exist_ok=True)

    BALL_CATEGORY_ID = 0  # Pilota té etiqueta 0 segons com comentes

    for i in range(num_images):
        img_tensor, target = dataset[i]  # tensor [C,H,W]
        _, img_height, img_width = img_tensor.shape

        # Desnormalització (si cal, adapta segons la teva normalització)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
        img_denorm = img_tensor * std + mean
        img_denorm = img_denorm.clamp(0, 1)
        img = img_denorm.mul(255).byte().permute(1, 2, 0).cpu().numpy()

        fig, ax = plt.subplots(1)
        ax.imshow(img)

        # Dibuixa la pilota a partir de labels i boxes
        boxes = target["boxes"].cpu().numpy()
        labels = target["labels"].cpu().numpy()
        for j, label in enumerate(labels):
            if label == BALL_CATEGORY_ID:
                x, y, w, h = boxes[j]

                # Denormalitzar
                x *= img_width
                w *= img_width
                y *= img_height
                h *= img_height

                #print(f"Ball detected - x: {x:.1f}, y: {y:.1f}, w: {w:.1f}, h: {h:.1f}")

                rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='red', facecolor='none')
                ax.add_patch(rect)
                ax.text(x, y - 5, "Ball", color='red', fontsize=12, weight='bold')

        # Dibuixa els jugadors a partir de players
        players = target.get("players", [])
        for player in players:
            px, py, pw, ph = player["bbox"]  # normalized [0,1]
            team = player["team"]

            # Passa a píxels
            xmin = px * img_width
            ymin = py * img_height
            w = pw * img_width
            h = ph * img_height

            rect = patches.Rectangle((xmin, ymin), w, h, linewidth=2, edgecolor='blue', facecolor='none')
            ax.add_patch(rect)
            ax.text(xmin, ymin - 5, f"team: {team}", color='blue', fontsize=10, weight='bold')

        ax.axis('off')
        plt.tight_layout()
        save_path = f"debug_images/{split_name}_boxes/img_{i}_bbox_team.png"
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        #print(f"Saved {save_path}")


def get_tracking_root(image_set):
    base_path = '/data-fast/data-server/ccorbi/SN-Tracking/tracking'
    if image_set == 'train':
        return f"{base_path}/train"
    elif image_set == 'val':
        return f"{base_path}/val"
    elif image_set == 'test':
        return f"{base_path}/test"
    else:
        raise ValueError(f"Unknown image_set {image_set}")


def load_tracking_info(seq_name, tracking_root):
    """
    Llegeix el fitxer gameinfo.ini i retorna un diccionari trackid -> team_id (0:left, 1:right)
    """
    gameinfo_path = os.path.join(tracking_root, seq_name, "gameinfo.ini")
    config = configparser.ConfigParser()
    config.read(gameinfo_path)

    trackid_to_team = {}
    for k, v in config['Sequence'].items():
        if not k.startswith("trackletid_"):
            continue
        trackid = int(k.split('_')[1])
        role_team = v.strip().lower()
        if "player team left" in role_team or "goalkeeper team left" in role_team:
            trackid_to_team[trackid] = 0
        elif "player team right" in role_team or "goalkeeper team right" in role_team:
            trackid_to_team[trackid] = 1
    return trackid_to_team


def load_players(seq_name, frame_id, trackid_to_team, tracking_root):
    """
    Llegeix el gt.txt i retorna les deteccions dels jugadors/porters per a un frame donat.
    Cada jugador conté: bbox normalitzat, team, i center.
    """
    gt_path = os.path.join(tracking_root, seq_name, "gt", "gt.txt")
    players = []

    with open(gt_path, "r") as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 7:
                continue

            fid = int(parts[0])
            trackid = int(parts[1])
            if fid != frame_id:
                continue

            if trackid not in trackid_to_team:
                continue  # ignore referees or ball

            x, y, w, h = map(float, parts[2:6])
            team = trackid_to_team[trackid]

            # Normalize bounding box
            x_norm = x / 1920
            y_norm = y / 1080
            w_norm = w / 1920
            h_norm = h / 1080
            bbox = [x_norm, y_norm, w_norm, h_norm]

            # Center point normalized
            cx = x + w / 2
            cy = y + h / 2
            center = [cx / 1920, cy / 1080]

            players.append({
                "bbox": bbox,
                "team": team,
                "center": center
            })

    return players


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks, tracking_root):
        super(CocoDetection,self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self.tracking_root = tracking_root

    def __getitem__(self, idx):
        img, target = super(CocoDetection,self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)

        img_path = self.coco.loadImgs(image_id)[0]["file_name"]
        parts = img_path.split('_')
        seq_name = parts[1]
        frame_num_str = parts[2].replace('frame', '').replace('.jpg', '')
        frame_id = int(frame_num_str)

        try:
            trackid_to_team = load_tracking_info(seq_name, self.tracking_root)
            players = load_players(seq_name, frame_id, trackid_to_team, self.tracking_root)
            target["players"] = players
        except Exception as e:
            print(f"[Warning] Error loading players for {img_path}: {e}")
            target["players"] = []

        if self._transforms is not None:
            '''
            print("Target abans transformacions:")
            for k, v in target.items():
                if isinstance(v, torch.Tensor):
                    print(f"{k}: tensor shape {v.shape}")
                    if k == "boxes":
                        print(f"{k} contingut: {v}")
                elif isinstance(v, list):
                    print(f"{k}: list len {len(v)}")
                else:
                    print(f"{k}: {type(v)}")
            '''
            img, target = self._transforms(img, target)
            '''
            print("Target després transformacions:")
            for k, v in target.items():
                if isinstance(v, torch.Tensor):
                    print(f"{k}: tensor shape {v.shape}")
                    if k == "boxes":
                        print(f"{k} contingut: {v}")
                elif isinstance(v, list):
                    print(f"{k}: list len {len(v)}")
                else:
                    print(f"{k}: {type(v)}")
            '''

        return img, target


def make_coco_transforms(image_set):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return normalize
        '''return T.Compose([
            
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])
    '''
    if image_set == 'val' or image_set == 'test':
        return normalize
        '''
        return T.Compose([
            #T.RandomResize([800], max_size=1333),
            normalize,
        ]) '''

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
        "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
        "test":  (root / "test2017",  root / "annotations" / f'{mode}_test2017.json')
    }

    img_folder, ann_file = PATHS[image_set]
    tracking_root = get_tracking_root(image_set)

    dataset = CocoDetection(
        img_folder, ann_file,
        transforms=make_coco_transforms(image_set),
        return_masks=args.masks,
        tracking_root=tracking_root
    )
    return dataset


# EXEMPLE D'US
if __name__ == "__main__":
    class Args:
        coco_path = '/data-fast/data-server/ccorbi/ball_detr'
        masks = False  # o True si vols carregar masks

    args = Args()

    dataset_train = build('train', args)
    dataset_val = build('val', args)
    dataset_test = build('test', args)

    draw_boxes_with_players_and_team(dataset_train, split_name="train", num_images=5)
    draw_boxes_with_players_and_team(dataset_val, split_name="val", num_images=5)
    draw_boxes_with_players_and_team(dataset_test, split_name="test", num_images=5)
    #print("Imatges guardades amb bounding boxes i teams.")


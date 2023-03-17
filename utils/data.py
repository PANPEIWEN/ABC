# @Time    : 2022/4/6 14:41
# @Author  : PEIWEN PAN
# @Email   : 121106022690@njust.edu.cn
# @File    : data.py
# @Software: PyCharm
import random
import sys
import os.path as osp
import os
from PIL import Image, ImageOps, ImageFilter
import torchvision.transforms as transforms
import torch.utils.data as Data
import torch
import numpy as np


class DatasetLoad(Data.Dataset):
    def __init__(self, data_root, base_size, crop_size, mode, train_dir, test_dir, data_aug=True, suffix='png',
                 **kwargs):
        self.base_size = base_size
        self.crop_size = crop_size
        self.mode = mode
        self.data_aug = data_aug
        assert mode in ['train', 'test'], 'The mode should be train or test'
        if mode == 'train':
            self.data_dir = osp.join(data_root, train_dir)
        else:
            self.data_dir = osp.join(data_root, test_dir)

        self.img_names = []
        for img in os.listdir(osp.join(self.data_dir, 'images')):
            if img.endswith(suffix):
                self.img_names.append(img)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])

    def _sync_transform(self, img, mask):
        if self.mode == 'train' and self.data_aug:
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            crop_size = self.crop_size
            long_size = random.randint(
                int(self.base_size * 0.5), int(self.base_size * 2.0))
                # int(self.base_size * 0.8), int(self.base_size * 1.2))
            w, h = img.size
            if h > w:
                oh = long_size
                ow = int(1.0 * w * long_size / h + 0.5)
                short_size = ow
            else:
                ow = long_size
                oh = int(1.0 * h * long_size / w + 0.5)
                short_size = oh
            img = img.resize((ow, oh), Image.BILINEAR)
            mask = mask.resize((ow, oh), Image.NEAREST)
            if short_size < crop_size:
                padh = crop_size - oh if oh < crop_size else 0
                padw = crop_size - ow if ow < crop_size else 0
                img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
                mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
            w, h = img.size
            x1 = random.randint(0, w - crop_size)
            y1 = random.randint(0, h - crop_size)
            img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
            mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
            if random.random() < 0.5:
                img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))
            img, mask = np.array(img), np.array(mask)
            img = self.transform(img)
            mask = transforms.ToTensor()(mask)
        else:
            img = img.resize((self.base_size, self.base_size), Image.BILINEAR)
            mask = mask.resize((self.base_size, self.base_size), Image.NEAREST)
            img, mask = np.array(img), np.array(mask)
            img = self.transform(img)
            mask = transforms.ToTensor()(mask)
        return img, mask

    def __getitem__(self, item):
        img_name = self.img_names[item]
        img_path = osp.join(self.data_dir, 'images', img_name)
        label_path = osp.join(self.data_dir, 'masks', img_name)
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(label_path)
        img, mask = self._sync_transform(img, mask)
        return img, mask

    def __len__(self):
        return len(self.img_names)


# SIRST_8000
class SirstAugDataset(Data.Dataset):
    def __init__(self, base_dir='/data1/ppw/works/All_ISTD/datasets/SIRST_AUG', mode='train'):
        assert mode in ['train', 'test']

        if mode == 'train':
            self.data_dir = osp.join(base_dir, 'trainval')
        elif mode == 'test':
            self.data_dir = osp.join(base_dir, 'test')
        else:
            raise NotImplementedError

        self.names = []
        for filename in os.listdir(osp.join(self.data_dir, 'images')):
            if filename.endswith('png'):
                self.names.append(filename)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])

    def __getitem__(self, i):
        name = self.names[i]
        img_path = osp.join(self.data_dir, 'images', name)
        label_path = osp.join(self.data_dir, 'masks', name)

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(label_path)

        img, mask = self.transform(img), transforms.ToTensor()(mask)
        return img, mask

    def __len__(self):
        return len(self.names)


# SIRST_40000
class Sirst40000(Data.Dataset):
    def __init__(self, base_dir='/data1/ppw/works/All_ISTD/datasets/', data_name='SIRST_4M_T86', mode='train',
                 crop_size=256, base_size=256):
        self.mode = mode
        base_dir += data_name
        assert mode in ['train', 'val', 'test']
        if mode == 'train':
            self.data_dir = osp.join(base_dir, 'train')
        elif mode == 'val':
            self.data_dir = osp.join(base_dir, 'val')
        elif mode == 'test':
            self.data_dir = osp.join(base_dir, 'test')
        else:
            raise NotImplementedError
        self.crop_size = crop_size
        self.base_size = base_size

        self.names = []
        for filename in os.listdir(osp.join(self.data_dir, 'images')):
            if filename.endswith('png'):
                self.names.append(filename)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # Default mean and std
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])

    def _sync_transform(self, img, mask):
        if self.mode == 'train':
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            crop_size = self.crop_size
            long_size = random.randint(
                int(self.base_size * 0.5), int(self.base_size * 2.0))
            w, h = img.size
            if h > w:
                oh = long_size
                ow = int(1.0 * w * long_size / h + 0.5)
                short_size = ow
            else:
                ow = long_size
                oh = int(1.0 * h * long_size / w + 0.5)
                short_size = oh
            img = img.resize((ow, oh), Image.BILINEAR)
            mask = mask.resize((ow, oh), Image.NEAREST)
            if short_size < crop_size:
                padh = crop_size - oh if oh < crop_size else 0
                padw = crop_size - ow if ow < crop_size else 0
                img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
                mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
            w, h = img.size
            x1 = random.randint(0, w - crop_size)
            y1 = random.randint(0, h - crop_size)
            img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
            mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
            if random.random() < 0.5:
                img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))
            img, mask = np.array(img), np.array(mask)
        else:
            img = img.resize((self.base_size, self.base_size), Image.BILINEAR)
            mask = mask.resize((self.base_size, self.base_size), Image.NEAREST)
            img, mask = np.array(img), np.array(mask)

        # img = img.resize((self.base_size, self.base_size), Image.BILINEAR)
        # mask = mask.resize((self.base_size, self.base_size), Image.NEAREST)
        return img, mask

    def __getitem__(self, i):
        name = self.names[i]
        img_path = osp.join(self.data_dir, 'images', name)
        label_path = osp.join(self.data_dir, 'masks', name)
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(label_path).convert('L')
        img, mask = self._sync_transform(img, mask)
        img, mask = self.transform(img), transforms.ToTensor()(mask)
        return img, mask

    def __len__(self):
        return len(self.names)


class MDFADataset(Data.Dataset):
    def __init__(self, base_dir='../data/MDFA', mode='train', base_size=256):
        assert mode in ['train', 'test']

        self.mode = mode
        if mode == 'train':
            self.img_dir = osp.join(base_dir, 'training')
            self.mask_dir = osp.join(base_dir, 'training')
        elif mode == 'test':
            self.img_dir = osp.join(base_dir, 'test_org')
            self.mask_dir = osp.join(base_dir, 'test_gt')
        else:
            raise NotImplementedError

        self.img_transform = transforms.Compose([
            transforms.Resize((base_size, base_size),
                              interpolation=Image.BILINEAR),
            transforms.ToTensor(),
            # Default mean and std
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((base_size, base_size),
                              interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])

    def __getitem__(self, i):
        if self.mode == 'train':
            img_path = osp.join(self.img_dir, '%06d_1.png' % i)
            mask_path = osp.join(self.mask_dir, '%06d_2.png' % i)
        elif self.mode == 'test':
            img_path = osp.join(self.img_dir, '%05d.png' % i)
            mask_path = osp.join(self.mask_dir, '%05d.png' % i)
        else:
            raise NotImplementedError

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        img, mask = self.img_transform(img), self.mask_transform(mask)
        return img, mask

    def __len__(self):
        if self.mode == 'train':
            return 9978
        elif self.mode == 'test':
            return 100
        else:
            raise NotImplementedError


class TrainSetLoader(Data.Dataset):
    NUM_CLASS = 1

    def __init__(self, dataset_dir, img_id, base_size=512, crop_size=480, transform=None, suffix='.png'):
        super(TrainSetLoader, self).__init__()
        self.transform = transform
        self._items = img_id
        self.masks = dataset_dir + '/' + 'masks'
        self.images = dataset_dir + '/' + 'images'
        self.base_size = base_size
        self.crop_size = crop_size
        self.suffix = suffix

    def _sync_transform(self, img, mask):
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        crop_size = self.crop_size
        long_size = random.randint(
            int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            oh = long_size
            ow = int(1.0 * w * long_size / h + 0.5)
            short_size = ow
        else:
            ow = long_size
            oh = int(1.0 * h * long_size / w + 0.5)
            short_size = oh
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))
        img, mask = np.array(img), np.array(mask, dtype=np.float32)
        return img, mask

    def __getitem__(self, idx):
        img_id = self._items[idx]
        img_path = self.images + '/' + img_id + self.suffix
        label_path = self.masks + '/' + img_id + self.suffix

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(label_path)

        img, mask = self._sync_transform(img, mask)
        if self.transform is not None:
            img = self.transform(img)
        mask = np.expand_dims(mask, axis=0).astype('float32') / 255.0
        return img, torch.from_numpy(mask)

    def __len__(self):
        return len(self._items)


class TestSetLoader(Data.Dataset):
    NUM_CLASS = 1

    def __init__(self, dataset_dir, img_id, transform=None, base_size=512, crop_size=480, suffix='.png'):
        super(TestSetLoader, self).__init__()
        self.transform = transform
        self._items = img_id
        self.masks = dataset_dir + '/' + 'masks'
        self.images = dataset_dir + '/' + 'images'
        self.base_size = base_size
        self.crop_size = crop_size
        self.suffix = suffix

    def _testval_sync_transform(self, img, mask):
        base_size = self.base_size
        img = img.resize((base_size, base_size), Image.BILINEAR)
        mask = mask.resize((base_size, base_size), Image.NEAREST)

        img, mask = np.array(img), np.array(mask, dtype=np.float32)
        return img, mask

    def __getitem__(self, idx):
        img_id = self._items[idx]
        img_path = self.images + '/' + img_id + self.suffix
        label_path = self.masks + '/' + img_id + self.suffix
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(label_path)
        img, mask = self._testval_sync_transform(img, mask)
        if self.transform is not None:
            img = self.transform(img)
        mask = np.expand_dims(mask, axis=0).astype('float32') / 255.0
        return img, torch.from_numpy(mask)

    def __len__(self):
        return len(self._items)


class DemoLoader(Data.Dataset):
    NUM_CLASS = 1

    def __init__(self, dataset_dir, transform=None, base_size=512, crop_size=480, suffix='.png'):
        super(DemoLoader, self).__init__()
        self.transform = transform
        self.images = dataset_dir
        self.base_size = base_size
        self.crop_size = crop_size
        self.suffix = suffix

    def _demo_sync_transform(self, img):
        base_size = self.base_size
        img = img.resize((base_size, base_size), Image.BILINEAR)
        img = np.array(img)
        return img

    def img_preprocess(self):
        img_path = self.images
        img = Image.open(img_path).convert('RGB')
        img = self._demo_sync_transform(img)
        if self.transform is not None:
            img = self.transform(img)
        return img


def load_dataset(root, dataset, split_method='idx_427'):
    train_txt = root + '/' + dataset + '/' + split_method + '/' + 'trainval.txt'
    test_txt = root + '/' + dataset + '/' + split_method + '/' + 'test.txt'
    train_img_ids = []
    val_img_ids = []
    with open(train_txt, "r") as f:
        line = f.readline()
        while line:
            train_img_ids.append(line.split('\n')[0])
            line = f.readline()
        f.close()
    with open(test_txt, "r") as f:
        line = f.readline()
        while line:
            val_img_ids.append(line.split('\n')[0])
            line = f.readline()
        f.close()
    return train_img_ids, val_img_ids, test_txt

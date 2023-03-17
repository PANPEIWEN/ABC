# @Time    : 2022/4/7 17:01
# @Author  : PEIWEN PAN
# @Email   : 121106022690@njust.edu.cn
# @File    : visual.py
# @Software: PyCharm
import os
import shutil

import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt


def make_show_dir(show_dir):
    if not os.path.exists(show_dir):
        os.mkdir(show_dir)

    if os.path.exists(os.path.join(show_dir, 'result')):
        shutil.rmtree(os.path.join(show_dir, 'result'))  # 删除目录，包括目录下的所有文件
    os.mkdir(os.path.join(show_dir, 'result'))

    if os.path.exists(os.path.join(show_dir, 'fuse')):
        shutil.rmtree(os.path.join(show_dir, 'fuse'))  # 删除目录，包括目录下的所有文件
    os.mkdir(os.path.join(show_dir, 'fuse'))


def save_Pred_GT(pred, labels, show_dir, num, cfg):
    img_name = os.listdir(os.path.join(cfg.data['data_root'], cfg.data['test_dir'], 'images'))
    val_img_ids = []
    for img in img_name:
        val_img_ids.append(img.split('.')[0])
    # predsss = ((torch.sigmoid((pred)).cpu().numpy()) * 255).astype('int64')
    predsss = np.array((pred > 0).cpu()).astype('int64') * 255
    predsss = np.uint8(predsss)
    labelsss = labels * 255
    labelsss = np.uint8(labelsss.cpu())

    img = Image.fromarray(predsss.reshape(cfg.data['crop_size'], cfg.data['crop_size']))
    img.save(show_dir + '/result/' + '%s_Pred' % (val_img_ids[num]) + '.' + cfg.data['suffix'])
    img = Image.fromarray(labelsss.reshape(cfg.data['crop_size'], cfg.data['crop_size']))
    img.save(show_dir + '/result/' + '%s_GT' % (val_img_ids[num]) + '.' + cfg.data['suffix'])


def save_Pred_GT_visulize(pred, img_demo_dir, img_demo_index, suffix, cfg):
    predsss = np.array((pred > 0).cpu()).astype('int64') * 255
    predsss = np.uint8(predsss)

    img = Image.fromarray(predsss.reshape(cfg.data['crop_size'], cfg.data['crop_size']))
    img.save(img_demo_dir + '/' + '%s_Pred' % (img_demo_index) + suffix)

    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    img = plt.imread(img_demo_dir + '/' + img_demo_index + suffix)
    plt.imshow(img, cmap='gray')
    plt.xlabel("Raw Imamge", size=11)

    plt.subplot(1, 2, 2)
    img = plt.imread(img_demo_dir + '/' + '%s_Pred' % (img_demo_index) + suffix)
    plt.imshow(img, cmap='gray')
    plt.xlabel("Predicts", size=11)

    plt.savefig(img_demo_dir + '/' + img_demo_index + "_fuse" + suffix, facecolor='w', edgecolor='red')
    plt.show()


def total_show_generation(show_dir, cfg):
    source_image_path = os.path.join(cfg.data['data_root'], cfg.data['test_dir'], 'images')
    ids = []
    img_name = os.listdir(source_image_path)
    for img in img_name:
        ids.append(img.split('.')[0])
    for i in range(len(ids)):
        source_image = source_image_path + '/' + ids[i] + '.' + cfg.data['suffix']
        target_image = show_dir + '/result/' + ids[i] + '.' + cfg.data['suffix']
        shutil.copy(source_image, target_image)
    for i in range(len(ids)):
        source_image = show_dir + '/result/' + ids[i] + '.' + cfg.data['suffix']
        img = Image.open(source_image)
        img = img.resize((cfg.data['crop_size'], cfg.data['crop_size']), Image.ANTIALIAS)
        img.save(source_image)
    for m in range(len(ids)):
        print('Processing the %d image' % (m + 1))
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 3, 1)
        img = plt.imread(show_dir + '/result/' + ids[m] + '.' + cfg.data['suffix'])
        plt.imshow(img, cmap='gray')
        plt.xlabel("Raw Imamge", size=11)

        plt.subplot(1, 3, 2)
        img = plt.imread(show_dir + '/result/' + ids[m] + '_GT' + '.' + cfg.data['suffix'])
        plt.imshow(img, cmap='gray')
        plt.xlabel("Ground Truth", size=11)

        plt.subplot(1, 3, 3)
        img = plt.imread(show_dir + '/result/' + ids[m] + '_Pred' + '.' + cfg.data['suffix'])
        plt.imshow(img, cmap='gray')
        plt.xlabel("Predicts", size=11)
        plt.savefig(show_dir + '/fuse/' + ids[m].split('.')[0] + "_fuse" + '.' + cfg.data['suffix'],
                    facecolor='w', edgecolor='red')

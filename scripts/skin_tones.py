import os
import numpy as np
from PIL import Image
from skimage.filters import gaussian
from skimage.color import rgb2lab, lab2rgb
from sklearn import cluster
import cv2
from skimage.io import imread
import os
import pandas as pd
import copy
import argparse


def read_img(fpath):
    img = Image.open(fpath).convert('RGB')
    return img


def isadjacent(pos, newpos):
    """
    Check whether two coordinates are adjacent
    """
    # check for adjacent columns and rows
    return np.all(np.abs(np.array(newpos) - np.array(pos)) < 2)


def count_patches(A):
    """
    Count the number of non-zero patches in an array.
    """
    
    # get non-zero coordinates
    coords = np.nonzero(A)

    # add them to a list
    inipatches = list(zip(*coords))
    
    # list to contain all patches
    allpatches = []

    while len(inipatches) > 0:
        patch = [inipatches.pop(0)]

        i = 0
        # check for all points adjacent to the points within the current patch
        while True:
            plen = len(patch)
            curpatch = patch[i]
            remaining = copy.deepcopy(inipatches)
            for j in range(len(remaining)):
                if isadjacent(curpatch, remaining[j]):
                    patch.append(remaining[j])
                    inipatches.remove(remaining[j])
                    if len(inipatches) == 0:
                        break
        
            if len(inipatches) == 0 or plen == len(patch):
                # nothing added to patch or no points remaining
                break

            i += 1
    
        allpatches.append(patch)
    return len(allpatches)

def get_hue(a_values, b_values, eps=1e-8):
    """Compute hue angle"""
    return np.degrees(np.arctan(b_values / (a_values + eps)))


def mode_hist(x, bins='sturges'):
    """Compute a histogram and return the mode"""
    hist, bins = np.histogram(x, bins=bins)
    mode = bins[hist.argmax()]
    return mode


def clustering(x, n_clusters=5, random_state=2024):
    model = cluster.KMeans(n_clusters, random_state=random_state)
    model.fit(x)
    return model.labels_, model


def get_scalar_values(skin_smoothed_lab, labels, topk=3, bins='sturges'):
    hue_angle = get_hue(skin_smoothed_lab[:, :, 1], skin_smoothed_lab[:, :, 2])
    skin_smoothed = lab2rgb(skin_smoothed_lab)

    stacked_array = np.stack([skin_smoothed_lab[:, :, 0], skin_smoothed_lab[:, :, 1], skin_smoothed_lab[:, :, 2], hue_angle,
                                 skin_smoothed[:, :, 0], skin_smoothed[:, :, 1], skin_smoothed[:, :, 2]], axis=-1)
    reshaped_array = stacked_array.reshape(-1, 7)
    data_to_cluster = reshaped_array

    # Extract skin pixels for each mask (by clusters)
    n_clusters = len(np.unique(labels))
    #n_clusters usually = 5
    masked_skin = [data_to_cluster[labels == i, :] for i in range(n_clusters)]
    n_pixels = np.asarray([np.sum(labels == i) for i in range(n_clusters)])
    keys = ['lum', 'a*', 'b*', 'hue', 'red', 'green', 'blue']
    res = {}

    for i, key in enumerate(keys):
        res[key] = np.array([mode_hist(part[:, i], bins=bins)
                            for part in masked_skin])

    idx = np.argsort(res['lum'])[::-1][:topk]

    res_topk = {}
    for key in keys:
        res_topk[key] = np.average(res[key][idx], weights=n_pixels[idx])
        res_topk[key+'_std'] = np.sqrt(np.average((res[key][idx]-res_topk[key])**2, weights=n_pixels[idx]))
    return res_topk


def get_skin_values(img, mask, n_clusters=5):
    img_smoothed = gaussian(img, sigma=(1, 1), truncate=4, channel_axis=-1)

    skin_smoothed = np.where(mask, img_smoothed, 0)
    im = Image.fromarray((skin_smoothed * 255).astype(np.uint8))
    im.show()

    skin_smoothed_lab = rgb2lab(skin_smoothed)

    res = {}
    hue_angle = get_hue(skin_smoothed_lab[:, :, 1], skin_smoothed_lab[:, :, 2])

    stacked_array = np.stack((skin_smoothed_lab[:, :, 0], hue_angle), axis=-1)
    reshaped_array = stacked_array.reshape(-1, 2)

    labels, model = clustering(reshaped_array, n_clusters=n_clusters)

    tmp = get_scalar_values(skin_smoothed_lab, labels)
    res['lum'] = tmp['lum']
    res['hue'] = tmp['hue']
    res['lum_std'] = tmp['lum_std']
    res['hue_std'] = tmp['hue_std']

    # also extract RGB for visualization purposes
    res['red'] = tmp['red']
    res['green'] = tmp['green']
    res['blue'] = tmp['blue']
    res['red_std'] = tmp['red_std']
    res['green_std'] = tmp['green_std']
    res['blue_std'] = tmp['blue_std']
    res['a*'] = tmp['a*']
    res['b*'] = tmp['b*']
    res['a*_std'] = tmp['a*_std']
    res['b*_std'] = tmp['b*_std']
    
    return res

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, help="Name of T2I model")
    parser.add_argument("--images-dir", "-i", type=str, help="Path to the folder of generated images")
    parser.add_argument("--masks-dir", "-a", type=str, help="Path to the folder of skin tone masks")
    parser.add_argument("--results-dir", "-r", type=str, help="Path to save results")
    args = parser.parse_args()

    model = args.model
    images_dir = args.images_dir
    masks_dir = args.masks_dir
    results_dir = args.results_dir

    attrs = ['lum', 'a*', 'b*', 'hue', 'red', 'green', 'blue']
    res = {'img': [], 'mask_size': [], 'cloth_size': [], 'hat_size': []}
    for attr in attrs:
        res[attr] = []
        res[attr+'_std'] = []

    files = [x for x in os.listdir(images_dir) if not x.startswith(".")]
    files = [x for x in files if x.split("-")[0][:-1].isdigit()]
    files = [x for x in files if int(x.split("-")[0][:-1]) == 93]
    files.sort()

    for t, f in enumerate(files): 
        # reading images
        fimg = f'{images_dir}/{f}'
        images = []
        image = cv2.imread(str(fimg))

        img_full = image
        height, width, channels = image.shape

        # Number of pieces Horizontally
        W_SIZE  = 5
        # Number of pieces Vertically to each Horizontal
        H_SIZE = 5

        counter = 0
        idx = f.split("-")[0][:-1]
        l = f.split("-")[0][-1]

        for i in range(2,25):
            fmask = f'{masks_dir}/{idx}/{i}_{l}_output.png' 
            res['img'].append(f'{idx}_{i}_{l}')

            for ih in range(H_SIZE ):
                for iw in range(W_SIZE ):
                    x = width/W_SIZE * iw
                    y = height/H_SIZE * ih
                    h = (height / H_SIZE)
                    w = (width / W_SIZE )
                    counter+=1
                    img = img_full[int(y):int(y+h), int(x):int(x+w)]
                    images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
            img = images[i]
            img_original = Image.fromarray(img)

            mask = imread(fmask)
            skin_mask = np.all(mask == np.array([173, 21, 6]), axis=-1)
            mask_size = len(skin_mask[np.nonzero(skin_mask)])
            skin_mask = np.repeat(skin_mask[:, :, np.newaxis], 3, axis=2)

            cloth_mask = np.all(mask == np.array([42, 188, 182]), axis=-1)
            cloth_size = len(cloth_mask[np.nonzero(cloth_mask)])
            hat_mask = np.all(mask == np.array([10, 14, 4]), axis=-1)
            hat_size = len(hat_mask[np.nonzero(hat_mask)])

            tmp = get_skin_values(np.asarray(img_original),
                                skin_mask)
            for attr in attrs:
                res[attr].append(tmp[attr])
                res[attr+'_std'].append(tmp[attr+'_std'])

            res['mask_size'].append(mask_size)
            res['cloth_size'].append(cloth_size)
            res['hat_size'].append(hat_size)

    df = pd.DataFrame.from_dict(res)
    df.to_csv(f'{results_dir}/{model}_skin_tones_results.csv', index=False)

if __name__ == "__main__":
    main()

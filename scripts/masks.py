import argparse, pathlib, json, logging
import cv2
import torch
from torchvision import transforms
from src.utils import construct_image, create_model, setup_logging
import numpy as np
from PIL import Image
import os

logger = logging.getLogger('eval')


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mapping', type=pathlib.Path, required=True,
                        help='Path to the mapping')
    parser.add_argument('--model', type=pathlib.Path, default=None,
                        help='Path to the model weights')
    parser.add_argument('-cmap', '--color-map', type=pathlib.Path, required=True,
                        help='Path to the color map')
    parser.add_argument('-i', '--images-dir', type=pathlib.Path, required=True,
                        help='Path to folder of generated images')
    parser.add_argument('-a', '--masks-dir', type=pathlib.Path,
                        help='Path to save masks')
    return parser.parse_args()


def load_input(image_path: pathlib.Path, i) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            transforms.Resize(size=(512, 512))
        ]
    )
    images = []
    image = cv2.imread(str(image_path))
    
    img_full = image
    height, width, channels = image.shape

    # Number of pieces Horizontally
    W_SIZE  = 5
    # Number of pieces Vertically to each Horizontal
    H_SIZE = 5

    counter = 0

    for ih in range(H_SIZE ):
        for iw in range(W_SIZE ):
            x = width/W_SIZE * iw
            y = height/H_SIZE * ih
            h = (height / H_SIZE)
            w = (width / W_SIZE )
            counter+=1
            img = img_full[int(y):int(y+h), int(x):int(x+w)]
            images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                
    image = images[i]
    shape = image.shape[:2][::-1]
    input = transform(image)
    return torch.unsqueeze(input, dim=0), shape


def load_mapping(mapping_path: pathlib.Path) -> dict:
    with mapping_path.open() as mapping_f:
        result = json.load(mapping_f)
    return result


if __name__ == '__main__':
    setup_logging()
    args = parse_arguments()

    images_dir = args.images_dir
    output_dir = args.masks_dir
    files = os.listdir(images_dir)
    files.sort()

    for f in files:
        idx = f.split("-")[0][:-1]
        l = f.split("-")[0][-1]
        if f.split("-")[0][:-1].isdigit() and not os.path.isfile(f'{output_dir}/{idx}/24_{l}_output.png'):
            if not os.path.isdir(f'{output_dir}/{idx}'):   
                os.mkdir(f'{output_dir}/{idx}')
    
            for i in range(25):
                input = load_input(f'{images_dir}/{f}', i)

                input_image = input[0]
                init_shape = input[1]
                logger.info('Loaded input image.')
                mapping = load_mapping(mapping_path=args.mapping)
                color_map = load_mapping(mapping_path=args.color_map)
                logger.info('Loaded mapping')
                model = create_model(output_channels=len(mapping.keys()), weights=args.model)
                logger.info('Instantiated model.')

                logger.debug('Start inferencing.')
                model_output = model(input_image)
                segmentation = model_output['out']
                logger.debug('Ended inferencing.')
                output_image = construct_image(output=segmentation[0], mapping=mapping, color_mapping=color_map)
                output_image = cv2.resize(output_image, dsize=init_shape)
                logger.debug('Constructed image.')
                cv2.imwrite(f'{output_dir}/{idx}/{i}_{l}_output.png', output_image)
                logger.info('Saved image.')

## Image-Generation-Bias

This repo contains code and data for our paper [Bias Amplification in Stable Diffusion’s Representation of Stigma Through Skin Tones and Their Homogeneity](https://arxiv.org/abs/2508.17465) published at AIES 2025.

🚧 Under construction... 🚧

---

### Experiment 1

To extract skin tone regions from generated images of faces, run the script ```masks.py``` (adapted from [Or-El et al. 2020's implementation](https://github.com/stupidcucumber/DeepLabV3-CelebHQ/blob/main/eval.py)). We also use Or-El et al. 2020's [model](https://drive.google.com/file/d/1YR4LTi-CIYl8zr7JmtJj5jcrpdsJx9Nd/view?usp=share_link) and [color map](https://github.com/stupidcucumber/DeepLabV3-CelebHQ/blob/main/example/color_mapping.json) and the mapping from the original [CelebAMask-HQ dataset](https://github.com/switchablenorms/CelebAMask-HQ?tab=readme-ov-file).

```
python masks.py --mapping mapping.json --model best_weights.pt -cmap color_mapping.json -i data/images/SDXL -a data/masks/SDXL

options:
  --mapping             Path to the mapping
  --model               Path to the model weights
  -cmap, --color-map    Path to the color map
  -i, --images-dir      Path to the folder of generated images
  -a, --masks-dir       Path to the folder to save masks
```

Run the script ```skin_tones.py``` (adapted from [Thong et al. (2023)'s implementation](https://github.com/SonyResearch/apparent_skincolor/blob/main/extract/predict.py)) to calculate luminance (_L*_) and hue (_h*_) for generated images. 

```
python skin_tones.py -m SDXL -i data/images/SDXL -a data/masks/SDXL -r data/results

options:
  -m, --model           Name of T2I model
  -i, --images-dir      Path to the folder of generated images
  -a, --masks-dir       Path to the folder of masks
  -r, --results-dir     Path to save results
```

### Experiment 2

For human face datasets, we use the results of skin tone analysis provided by Thong et al. (2023) available in their [project repository](https://github.com/SonyResearch/apparent_skincolor/blob/main/extract/results). 

### Experiment 3

_Calculating delta E..._

### Experiment 4

_Case study of stigmatized racial identities..._

---

If you found this helpful, you can cite our work with the following information:
```
@inproceedings{wilson2025bias,
  title={Bias Amplification in Stable Diffusion’s Representation of Stigma Through Skin Tones and Their Homogeneity},
  author={Wilson, Kyra and Ghosh, Sourojit and Caliskan, Aylin},
  booktitle={Proceedings of the AAAI/ACM Conference on AI, Ethics, and Society},
  volume={8},
  number={3},
  pages={2705--2717},
  year={2025}
}
```

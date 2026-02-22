```markdown
# Weakly Supervised Remote Sensing Segmentation

This repository contains a PyTorch implementation for semantic segmentation of high-resolution aerial imagery using weakly supervised learning. Specifically, it demonstrates how to train a dense prediction model using only sparse point annotations (100 pixels per image) via a custom **Partial Cross-Entropy (pfCE) Loss**.

## 🚀 Project Overview

Standard image segmentation requires fully annotated, dense pixel masks, which are expensive and time-consuming to create for satellite imagery. This project solves that by:
1. **Simulating Weak Supervision:** Aggressively downsampling dense ground-truth masks into 100 random point labels per image.
2. **Custom Loss Function:** Implementing a Partial Cross-Entropy Loss ($pfCE$) that calculates pixel-wise Focal Loss, masks out all unknown/unlabeled pixels, and averages the error exclusively over the known points.
3. **Robust Baseline:** Utilizing a U-Net architecture with a pre-trained ResNet-34 backbone to extract rich spatial features.

## 📊 Dataset

This project uses the **Semantic segmentation of aerial imagery** dataset by Humans in the Loop (via Kaggle). 
* **Images:** 72 high-resolution aerial tiles of Dubai.
* **Classes:** 6 distinct classes (Building, Land, Road, Vegetation, Water, Unlabeled).

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/syntaxland/partial-ce-segmentation-pytorch.git](https://github.com/syntaxland/partial-ce-segmentation-pytorch.git)

```

2. **Install dependencies:**
```bash
pip install -r requirements.txt

```


3. **Kaggle API Key:** To download the dataset automatically, you must have a Kaggle account.
* Go to your Kaggle Account Settings and click **Create New Token** to download `kaggle.json`.
* Place the `kaggle.json` file in your working directory (or `~/.kaggle/` if running locally).



## 💻 Usage

The entire pipeline—from data downloading to model training and visualization—is contained within the provided Jupyter Notebook.

1. Open `partial_ce_segmentation.ipynb` in Google Colab or Jupyter Notebook.
2. Ensure your `kaggle.json` is uploaded to the environment.
3. Run all cells.

The notebook will automatically download the data, simulate the point labels, train the model for 5 epochs, and output a side-by-side visualization of the original image, the simulated points, and the model's dense prediction.

## 📈 Results

Despite training on only a fraction of the available annotation data (100 points per 256x256 patch), the model demonstrates highly stable convergence. Over 5 epochs, the average training loss dropped from **0.2450** to **0.1142**, successfully learning dense spatial clustering and boundary detection purely from the sparse point labels.

```

***


```
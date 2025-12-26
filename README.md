# Deep-Car-Recognition
A real-time vehicle detection and classification system powered by **YOLOv8**. It is capable of identifying **100 distinct car models** with high accuracy.

##  Contributors
* **[Burak Fatih Özen](https://github.com/BurakFatihOzen)**: Data preprocessing, Inference optimization, Local testing (RTX 4060).
* **[Mehmet Gökmenoğlu](https://github.com/mehmetgokmenoglu)**: Model training (NVIDIA A100), Hyperparameter tuning, Large-scale training.

##  Project Architecture
This project was executed in two main stages:
1. **Training:** Conducted on **Google Colab Pro** using **NVIDIA A100 GPUs** for faster convergence on a large dataset.
2. **Inference:** Tested locally on **RTX 4060 GPU** for real-time performance.

## 📥 Download Model Weights
The final model file (`bestdeep100.pt`) is hosted externally due to GitHub's storage limits.

 **[Download Pre-trained Model via Google Drive](https://drive.google.com/file/d/1gecPq-LSGhzEOKde2phhvRMcgvdIYaC6/view?usp=sharing)**

## Installation & Usage
1. Clone the repository:
   ```bash
   git clone [https://github.com/BurakFatihOzen/Deep-Car-Recognition.git](https://github.com/BurakFatihOzen/Deep-Car-Recognition.git)

2. Install dependencies:
   pip install -r requirements.txt

# 🧠 CNN Filter Visualization using VGG16

Welcome to this deep learning interpretability project where we explore what individual convolutional filters in a neural network are actually "looking for"! 🔍 Using the pre-trained VGG16 model, we visualize the features that each filter responds to by generating patterns through gradient ascent.

---

## 📌 Project Overview

This notebook performs **filter visualization** on the **VGG16** convolutional neural network using **gradient ascent**.  
Instead of feeding real images, we create input images that *maximize* the activation of specific filters — showing us what the network has truly learned to recognize.

My Colab link : https://colab.research.google.com/drive/1sGdHRvvdtt8dupeVyprHc5RDGz4-y4N4?usp=sharing

---

## 🎯 Objectives

- Understand how CNN filters work internally
- Use gradient ascent to generate feature-maximizing images
- Visualize learned patterns in early and deep layers of VGG16
- Gain insights into how deep learning models interpret the visual world

---

## 🧱 Techniques & Concepts Covered

- What is a Convolutional Neural Network (CNN)?
- How filters and feature maps work
- VGG16 architecture and its relevance
- Gradient ascent for feature visualization
- Image post-processing and normalization
- Visualization grids of filters

---

## 🛠️ Tech Stack

| Tool/Library     | Purpose                      |
|------------------|------------------------------|
| Python 🐍        | Core Programming             |
| TensorFlow 🔮     | Deep Learning Framework      |
| NumPy 🧮         | Numerical Operations         |
| OpenCV 🧼        | Image Manipulation           |
| Matplotlib 📊    | Data & Image Visualization   |

---

## 📁 Files Included

| File Name                                      | Description                                  |
|------------------------------------------------|----------------------------------------------|
| `Visualizing_Filters_of_a_CNN_Starter.ipynb`   | Main project notebook                        |
| `CNN_Filter_Visualization_Concepts_Explained.md`| Tutorial-style concepts + code explanations  |

---

## 🧪 How It Works (Short Summary)

1. **Load** a pretrained VGG16 model (excluding the classifier)
2. **Choose a layer** and a specific filter within that layer
3. **Apply gradient ascent** to modify a blank image until that filter is highly activated
4. **Post-process** the image for readability
5. **Plot a grid** of generated patterns to visualize multiple filters at once

---

## 🚀 How to Run

1. Clone or download this repo.
2. Open the notebook in [Google Colab](https://colab.research.google.com/) or Jupyter.
3. Install necessary libraries (most are pre-installed in Colab).
4. Run each cell sequentially and explore the generated patterns.

---

## 📜 License

This project is open-source and free to use for academic and educational purposes. Attribution is appreciated! 🙏

---

## 🙌 Acknowledgments

- Pretrained model by Keras/TensorFlow (VGG16 - ImageNet)
- Inspired by visualization techniques shared by François Chollet (Keras creator)

---

### > Created by P. Venkat Anjan Kumar

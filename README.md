# Image Steganalysis: In Search for the Invisible
## Introduction 
This repository provides the source code and raw datasets associated with the study of Image Steganalysis.

As digital communication and multimedia technologies evolve, ensuring the security of information becomes increasingly critical. Images often transmitted online are vulnerable to unauthorized modifications and security threats. One such threat is steganography, a technique used to hide secret information within digital images in a way that is imperceptible to the human eye. While it can be used for secure communication, it can also be exploited for malicious purposes, such as hiding illegal content or spreading harmful information.

Steganalysis is the process of detecting hidden messages in digital images that have been concealed using steganography. It involves analyzing images for suspicious patterns or features to uncover concealed information and mitigate the risks posed by the malicious use of steganography.

<img src="steg.png" alt="Steganalysis" width="700">

## Data
The proposed architecture is trained on IStego100K dataset. It contains 208,104 images with the same size of 1024*1024. Among them, 200,000 images (100,000 cover-stego image pairs) are divided as the training set and the remaining 8,104 as testing set. For each image in IStego100K, the quality factors is randomly set in the range of 75-95, the steganographic algorithm is randomly selected from three well-known steganographic algorithms, which are J-uniward, nsF5 and UERD, and the embedding rate is also randomly set to be a value of 0.1-0.4.

## Installation 
### 1. Clone the repo 
```bash
git clone https://github.com/HardiMatholia/Image_Steganalysis.git
```
### 2. Change directory
```bash
cd Image_Steganalysis
```

## Training and Evaluation
* Change the paths of stego_dir, cover_dir, and test_dir to the appropriate dataset locations
* Install the required python dependencies
* Execute the main.py python file from the code directory 











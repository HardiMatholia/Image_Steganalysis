import os
import numpy as np
import cv2
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte
from skimage.feature import graycomatrix, graycoprops
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm
from skimage import filters

# === CONFIG ===
current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, '..', 'DIP_Dataset', 'Alaska_Dataset')

COVER_DIR = os.path.join(data_dir, 'Cover')
STEGO_DIR = os.path.join(data_dir, 'JUNIWARD')

N_SAMPLES = 500

# ----- GLCM Feature Extraction -----
def extract_glcm_features(image):
    if image.ndim == 3:
        image = rgb2gray(image)
    image = img_as_ubyte(image)

    glcm = graycomatrix(image, distances=[1, 2], angles=[0, np.pi/4], levels=256, symmetric=True, normed=True)
    props = ['contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy', 'correlation']
    features = [graycoprops(glcm, prop).mean() for prop in props]
    return np.array(features)

# ----- DCTR Feature Extraction (Your Preferred Method) -----
from scipy.fftpack import dct

def block_dct(image, block_size=8):
    h, w = image.shape
    dct_blocks = []
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = image[i:i+block_size, j:j+block_size]
            if block.shape == (block_size, block_size):
                dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
                dct_blocks.append(dct_block)
    return np.array(dct_blocks)

def dctr_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (256, 256))  # Optional: reduce for speed
    dct_blocks = block_dct(img)
    coeffs = dct_blocks[:, 1:, 1:].flatten()  # skip DC coefficient
    hist, _ = np.histogram(coeffs, bins=8192, range=(-50, 50), density=True)
    return hist

# ----- SFTA Feature Extraction -----
def sfta_features(image_path, block_size=16):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (256, 256))  # Optional: resize for faster processing
    
    # Compute Otsu's threshold
    thresh = filters.threshold_otsu(img)
    segmented_img = img > thresh

    # Fractal dimension calculation (simplified version)
    h, w = segmented_img.shape
    regions = []
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = segmented_img[i:i+block_size, j:j+block_size]
            if block.shape == (block_size, block_size):
                regions.append(np.mean(block))  # You can use other features as well
    return np.array(regions)


# ----- Combine Both Features -----
def extract_features(image_path):
    img = imread(image_path)
    glcm_feat = extract_glcm_features(img)
    dctr_feat = dctr_features(image_path)
    sfta_feat = sfta_features(image_path)
    return np.concatenate([glcm_feat, dctr_feat])
    # return glcm_feat

# ----- Load Dataset -----
def load_dataset(cover_dir, stego_dir, n_samples=500):
    cover_features, stego_features = [], []
    filenames = sorted(os.listdir(cover_dir))[:n_samples]

    for filename in tqdm(filenames, desc="Extracting features"):
        cover_path = os.path.join(cover_dir, filename)
        stego_path = os.path.join(stego_dir, filename)

        if not os.path.exists(stego_path):
            continue

        try:
            cover_feat = extract_features(cover_path)
            stego_feat = extract_features(stego_path)
            cover_features.append(cover_feat)
            stego_features.append(stego_feat)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    X = np.array(cover_features + stego_features)
    y = np.array([0] * len(cover_features) + [1] * len(stego_features))
    return X, y

# ----- Train & Evaluate -----
def main():
    print("Extracting features and preparing dataset...")
    X, y = load_dataset(COVER_DIR, STEGO_DIR, N_SAMPLES)

    print("Training classifier...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()

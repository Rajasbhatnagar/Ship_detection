import streamlit as st
import torch
import segmentation_models_pytorch as smp
import numpy as np
import pandas as pd
import cv2
from PIL import Image

# === CONFIG ===
MODEL_PATH = 'best_model.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === LOAD MODEL ===
@st.cache_resource
def load_model():
    model = smp.Unet('resnet34', encoder_weights='imagenet', classes=1)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# === RLE DECODER ===
def rle_decode(rle, shape=(768,768)):
    if pd.isna(rle): return np.zeros(shape, np.uint8)
    s = list(map(int, rle.split()))
    img = np.zeros(shape[0]*shape[1], np.uint8)
    for i in range(0, len(s), 2):
        img[s[i]-1:s[i]-1+s[i+1]] = 1
    return img.reshape(shape).T

# === METRICS FUNCTION ===
def calculate_metrics(pred_mask, true_mask):
    pred_mask = (pred_mask > 0.5).float()
    true_mask = (true_mask > 0.5).float()
    smooth = 1e-6

    tp = (pred_mask * true_mask).sum()
    fp = (pred_mask * (1 - true_mask)).sum()
    fn = ((1 - pred_mask) * true_mask).sum()
    tn = ((1 - pred_mask) * (1 - true_mask)).sum()

    precision = tp / (tp + fp + smooth)
    recall = tp / (tp + fn + smooth)
    dice = (2 * tp) / (2 * tp + fp + fn + smooth)
    accuracy = (tp + tn) / (tp + tn + fp + fn + smooth)

    return {
        'Precision': precision.item(),
        'Recall': recall.item(),
        'Dice Score': dice.item(),
        'Accuracy': accuracy.item()
    }

# === MAIN APP ===
st.title("🚢 Airbus Ship Detection Inference")

uploaded_file = st.file_uploader("Upload an Image (.jpg/.png)", type=["jpg", "png"])
csv_file = st.file_uploader("Upload train_ship_segmentations_v2.csv", type=["csv"])
image_id_input = st.text_input("Enter the Image ID (e.g. '000155de5.jpg'):")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image = np.array(image)
    image = cv2.resize(image, (768, 768))
    st.image(image, caption="Input Image", use_column_width=True)

    # Preprocess
    img_tensor = torch.from_numpy(image.transpose(2,0,1) / 255.).float().unsqueeze(0).to(DEVICE)

    # Predict
    with torch.no_grad():
        output = model(img_tensor)
        pred_mask = torch.sigmoid(output).squeeze().cpu().numpy()

    # Display predicted mask
    st.image(pred_mask, caption="Predicted Mask", use_column_width=True, clamp=True)

    if csv_file is not None and image_id_input.strip() != "":
        df = pd.read_csv(csv_file)
        rles = df.loc[df['ImageId'] == image_id_input.strip(), 'EncodedPixels']

        if len(rles) > 0:
            # Decode all RLEs for the image and combine them
            mask = np.zeros((768, 768), np.uint8)
            for rle in rles:
                mask = np.clip(mask + rle_decode(rle), 0, 1)

            st.image(mask*255, caption="Ground Truth Mask", use_column_width=True, clamp=True)

            gt_mask = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0).to(DEVICE)

            # Calculate metrics
            metrics = calculate_metrics(output, gt_mask)

            st.subheader("🔍 Evaluation Metrics")
            st.write(f"**Precision**: {metrics['Precision']*100:.2f}%")
            st.write(f"**Recall**: {metrics['Recall']*100:.2f}%")
            st.write(f"**Dice Score**: {metrics['Dice Score']*100:.2f}%")
            st.write(f"**Accuracy**: {metrics['Accuracy']*100:.2f}%")
        else:
            st.warning("⚠️ No ground-truth mask found for the given Image ID in the CSV.")
    else:
        st.info("ℹ️ Upload the CSV and enter the Image ID to compute metrics.")

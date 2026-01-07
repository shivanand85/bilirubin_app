import streamlit as st
import cv2
import numpy as np
import xgboost as xgb
import json

# -------------------------------
# Load model and metadata
# -------------------------------
booster = xgb.Booster()
booster.load_model("bilirubin_xgboost_patient_clean.json")

with open("bilirubin_xgboost_feature_names.json") as f:
    FEATURE_NAMES = json.load(f)

# -------------------------------
# Helper functions
# -------------------------------

def robust_stats(x):
    q1, q3 = np.percentile(x, [25, 75])
    return float(np.median(x)), float(q3 - q1)

def extract_skin_roi_general(img):
    ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(ycbcr)

    skin_mask = (
        (Cr > 135) & (Cr < 180) &
        (Cb > 85)  & (Cb < 135)
    ).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 3000:
        return None

    x, y, w, h = cv2.boundingRect(c)
    return img[y:y+h, x:x+w]

def extract_features_from_roi(roi):
    eps = 1e-6
    features = {}

    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
    ycbcr = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).astype(np.float32)

    L, A, B = cv2.split(lab)
    Y, Cr, Cb = cv2.split(ycbcr)
    H, S, V = cv2.split(hsv)

    features["L_med"], features["L_iqr"] = robust_stats(L)
    features["A_med"], features["A_iqr"] = robust_stats(A)
    features["B_med"], features["B_iqr"] = robust_stats(B)

    features["B_p75"] = float(np.percentile(B, 75))
    features["B_p90"] = float(np.percentile(B, 90))

    features["Cr_med"], features["Cr_iqr"] = robust_stats(Cr)
    features["Cb_med"], features["Cb_iqr"] = robust_stats(Cb)
    features["S_med"],  features["S_iqr"] = robust_stats(S)

    features["B_minus_A"] = features["B_med"] - features["A_med"]
    features["B_over_L"] = features["B_med"] / (features["L_med"] + eps)
    features["Cr_over_Cb"] = features["Cr_med"] / (features["Cb_med"] + eps)

    features["YI_lab"] = (
        (features["B_med"] - features["A_med"]) /
        (features["L_med"] + eps)
    )

    features["B_over_chroma"] = features["B_med"] / (
        np.sqrt(features["A_med"]**2 + features["B_med"]**2) + eps
    )

    features["LAB_hue"] = float(np.arctan2(features["B_med"], features["A_med"] + eps))
    features["YI_ycbcr"] = (features["Cr_med"] - features["Cb_med"]) / (
        features["Cr_med"] + features["Cb_med"] + eps
    )

    features["B_weighted_S"] = features["B_med"] * (features["S_med"] / 255.0)

    return np.array([features[f] for f in FEATURE_NAMES], dtype=np.float32)

def classify_severity(b):
    if b < 5:
        return "Normal 🟢"
    elif b < 10:
        return "Mild 🟡"
    elif b < 15:
        return "Moderate 🟠"
    else:
        return "Severe 🔴"

# -------------------------------
# Streamlit UI
# -------------------------------

st.set_page_config(page_title="Neonatal Bilirubin Estimator", layout="centered")

st.title("🩺 Neonatal Bilirubin Estimation")
st.caption("AI-assisted bilirubin level estimation from infant skin images")

st.info(
    """
    📷 **Photo Guidelines for Best Accuracy**

    • Skin should be clearly visible (face / chest / abdomen)  
    • Avoid clothes, blankets, or heavy background  
    • Use natural or white indoor lighting  
    • Avoid flash glare and shadows  
    • Ensure image is clear and in focus  

    ⚠️ Poor image quality may reduce accuracy.
    """
)

uploaded_file = st.file_uploader(
    "Upload infant image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.subheader("📸 Uploaded Image")
    st.image(img, caption="Uploaded Image", width=400)

    roi = extract_skin_roi_general(img)

    if roi is None:
        st.error("⚠️ Unable to detect sufficient skin region. Please upload a clearer image.")
    else:
        roi = cv2.medianBlur(roi, 3)
        X = extract_features_from_roi(roi).reshape(1, -1)

        dmat = xgb.DMatrix(X)
        pred = float(booster.predict(dmat)[0])

        low = max(0, pred - 1.5)
        high = pred + 1.5

        st.subheader("📊 Prediction Result")
        st.metric("Estimated Bilirubin (mg/dL)", f"{pred:.2f}")
        st.write(f"**Expected Range:** {low:.2f} – {high:.2f} mg/dL")
        st.write(f"**Severity Category:** {classify_severity(pred)}")

        st.warning(
            """
            ⚠️ **Medical Disclaimer**

            This result is generated using an **AI-based estimation model**.
            It is **not a medical diagnosis**.

            • Accuracy depends on image quality and lighting  
            • The model may produce incorrect estimates  
            • Do not rely on this result for treatment decisions  

            🩺 **Always consult a pediatrician or healthcare professional.**
            """
        )

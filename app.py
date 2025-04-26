import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf

# Load model yang sudah dilatih
model = tf.keras.models.load_model('mri_model.h5')  # Ganti dengan path model Anda

def predict_image(image):
    """Memproses gambar dan membuat prediksi."""
    img_resized = image.resize((128, 128)).convert('RGB')  # Resize dan ubah ke RGB
    img_array = np.array(img_resized) / 255.0  # Normalisasi
    img_array = np.expand_dims(img_array, axis=0)  # Menambahkan batch dimension

    prediction = model.predict(img_array)
    predicted_label = "Tumor Detected" if prediction[0][0] > 0.5 else "No Tumor Detected"
    return predicted_label


# UI Streamlit
st.title("Aplikasi Deteksi Tumor Otak")
st.write("Unggah Gambar MRI Untuk Mengetahui Apakah Ada Tumor Atau Tidak.")

# Mengunggah file
uploaded_file = st.file_uploader("Unggah Gambar MRI", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Menampilkan gambar yang diunggah
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang diunggah", use_column_width=True)

    # Melakukan prediksi
    with st.spinner("Memproses gambar..."):
        label = predict_image(image)
    
    st.success(f"Prediksi: {label}")

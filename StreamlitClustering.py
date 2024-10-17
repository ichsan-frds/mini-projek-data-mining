import streamlit as st
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from KMeans import kmeans
from sklearn.metrics import silhouette_score

# Fungsi untuk Menghitung Silhouette Coefficient 
# Semakin Tinggi = Cluster Semakin Menyebar
def silhouette_coef(data, label):
    silhouette = silhouette_score(data, label)
    return silhouette

def clustering(uploaded_file,cluster_numbers):

    # Mengakses uploaded_file
    img = imread(uploaded_file)
    # Menyimpan Size asli Gambar
    tinggi,lebar = img.shape[:2]
    # Resize gambar agar mudah untuk dikomputasi
    img = resize(img,(224,224))
    # Reshape gambar menjadi 2D Array dengan 3 color values (R,G,B)
    data = np.array(img.reshape(-1, 3))
    # Implementasi K-Means Clustering
    cluster,centroids = kmeans(data,cluster_numbers)
    # Buat Copy dari img untuk dimanipulasi
    result_image = img.copy()
    # Reshape gambar ke 2D dan mewarnainya berdasarkan label cluster
    for i in range(len(cluster)):
        result_image[i // 224, i % 224] = centroids[cluster[i]]
    # Resize kembali ke ukuran asli gambar
    result_image = resize(result_image, (tinggi, lebar), anti_aliasing=True)
    # Konversi ke format uint8 agar bisa ditampilkan
    result_image = (result_image * 255).astype(np.uint8)
    # Menghitung Silhouette Coefficient
    silhouette = silhouette_coef(data, cluster)

    return result_image,silhouette

st.title("Air Image Clustering")
st.subheader("UTS DATA MINING 2024")
st.subheader("Kelompok 21")
st.text("Josef Harvey Mangaratua - 140810220023")
st.text("Muhammad Ichsan Firdaus - 140810220025")
st.text("Daffa Burane Nugraha - 140810220039")

uploaded_file = st.file_uploader("Input Air Images")
cluster_numbers = st.selectbox("Jumlah Cluster",(2, 3, 4, 5))
cluster_numbers = int(cluster_numbers)
if(st.button("Lakukan Clustering")):
    if(uploaded_file):
        clustered_image,silhouette = clustering(uploaded_file,cluster_numbers)
        st.image(uploaded_file,caption="Original Image")
        st.image(clustered_image,caption=str(cluster_numbers) + " Clusters")
        st.text(f"Silhouette Coefficient : {round(silhouette, 4)} (Semakin Besar Semakin Baik)")
    else:
        st.header("INPUT GAMBAR TERLEBIH DAHULU")
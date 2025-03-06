import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.losses import MeanSquaredError
import tensorflow as tf
import streamlit as st

st.write(f"✅ TensorFlow version: {tf.__version__}")


# Charger les modèles une seule fois avec les bons noms de fichiers
@st.cache_resource
def load_models():
    """Charge tous les modèles de prédiction avec les bons fichiers."""
    import tensorflow as tf
	# Charger le modèle compatible avec TF 2.13
    model_age = tf.keras.models.load_model("modelAge_fixed.h5", compile=False)

    model_gender = tf.keras.models.load_model("my_modelGenre2.keras")  # Ton modèle de genre
    model_age_gender = tf.keras.models.load_model("best_age_genre_model.h5", compile=False)
    
    return model_age, model_gender, model_age_gender

# Charger les modèles
model_age, model_gender, model_age_gender = load_models()

def preprocess_image(image, target_size=(224, 224), grayscale=True):
    """Prépare l'image pour les prédictions."""
    image = image.resize(target_size)
    if grayscale:
        image = image.convert("L")  # Grayscale
        img_array = np.array(image) / 255.0
        img_array = img_array.reshape(1, target_size[0], target_size[1], 1)  # Ajouter batch et canal
    else:
        image = image.convert("RGB")  # Pour d'autres modèles utilisant RGB
        img_array = np.array(image) / 255.0
        img_array = img_array.reshape(1, target_size[0], target_size[1], 3)
    return img_array

# Interface Streamlit
st.title("Prédiction d'âge et de genre par IA")
st.write("Chargez une image de visage pour obtenir les prédictions.")

# Upload d'image
uploaded_image = st.file_uploader("Téléchargez une image", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Image chargée", use_column_width=True)

    # Choix du modèle
    option = st.radio("Choisissez le modèle :", ("Âge", "Genre", "Âge et Genre"))

    if st.button("Prédire"):
        if option == "Âge":
            img_gray = preprocess_image(image, target_size=(224,224), grayscale=True)
            age_pred = model_age.predict(img_gray)[0][0]
            st.success(f"📌 **Âge prédit** : {int(age_pred)} ans")

        elif option == "Genre":
            img_gray = preprocess_image(image, target_size=(224,224), grayscale=True)
            gender_pred = model_gender.predict(img_gray)[0][0]
            gender_text = "Homme" if gender_pred < 0.5 else "Femme"
            st.success(f"📌 **Genre prédit** : {gender_text}")

        elif option == "Âge et Genre":
            img_gray = preprocess_image(image, target_size=(96,96), grayscale=True)
            
            age_pred, gender_pred = model_age_gender.predict(img_gray)
            # Convertir les valeurs en scalaires
            age_pred = age_pred[0][0]  # Âge
            gender_pred = gender_pred[0][0]  # Genre
            # Déterminer le genre (seuil de 0.5)
            gender_text = "Homme" if gender_pred < 0.5 else "Femme"
            # Afficher les résultats
            st.success(f"📌 **Âge prédit** : {int(age_pred)} ans")
            st.success(f"📌 **Genre prédit** : {gender_text}")
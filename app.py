import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile

# Charger le modèle YOLO 
model = YOLO('bestmardi.pt')

# Les classes définies 
classname = ['HDPE','PET', 'PP', 'PS']

# Initialisation des états dans la session Streamlit
if "selected_arm" not in st.session_state:
    st.session_state.selected_arm = None
if "video_open" not in st.session_state:
    st.session_state.video_open = False
if "count_hdpe" not in st.session_state:
    st.session_state.count_hdpe = 0
if "count_pet" not in st.session_state:
    st.session_state.count_pet = 0    
if "count_pp" not in st.session_state:
    st.session_state.count_pp = 0
if "count_ps" not in st.session_state:
    st.session_state.count_ps = 0

# Fonction pour afficher le flux vidéo et détecter les objets avec YOLO
def open_video(video_path):
    cap = cv2.VideoCapture(video_path)  # Capture depuis la caméra par défaut (id 0)
    if not cap.isOpened():
        st.error("Impossible d'ouvrir la caméra. Vérifiez votre caméra.")
        return

    stframe = st.empty()  # Espace réservé pour afficher le flux vidéo
    while st.session_state.video_open:
        ret, frame = cap.read()
        if not ret:
            st.warning("Impossible de lire le flux vidéo.")
            break

        # Utilisation de YOLO pour la détection d'objets
        results = model(frame, stream=True)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                if box:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    conf = round(float(box.conf[0]), 2)
                    cls = int(box.cls[0])
                    label = classname[cls] if cls < len(classname) else "Unknown"
                    cv2.putText(frame, f"{label} {conf}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                    # Incrémenter les compteurs selon le bras sélectionné
                    if label == "HDPE" :
                        st.session_state.count_hdpe += 1
                    elif label == "PET" :
                        st.session_state.count_pet += 1    
                    elif label == "PP" :
                        st.session_state.count_pp += 1
                    elif label == "PS" :
                        st.session_state.count_ps += 1

        # Convertir l'image pour affichage dans Streamlit
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame, channels="RGB", use_column_width=True)

    cap.release()

# Interface principale Streamlit
st.title("Interface de Supervision des Bras Robotiques")
st.write("Sélectionnez un bras robotique pour visualiser ou consulter le nombre trié.")

# Sélection du bras robotique
st.sidebar.header("Sélection du Bras Robotique")
arm_options = ["Bras robotique 1 (HDPE)", "Bras robotique 2 (PET)","Bras robotique 3 (PP)", "Bras robotique 4 (PS)", "tout les bras"]
selected_arm = st.sidebar.selectbox("Choisissez un bras :", arm_options)

# Mise à jour de l'état du bras sélectionné
st.session_state.selected_arm = selected_arm

# Contrôle de la caméra
st.header(f"Contrôle pour {selected_arm}")
if not st.session_state.video_open:
    uploaded_file = st.file_uploader("Choisissez un fichier vidéo", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name
        if st.button("Ouvrir la vidéo"):
            st.session_state.video_open = True
            open_video(video_path)
else:
    if st.button("Fermer la vidéo"):
        st.session_state.video_open = False

# Affichage du nombre trié
st.header(f"Nombre trié pour {selected_arm}")
if selected_arm == "Bras robotique 1 (HDPE)":
    st.metric("HDPE triés", st.session_state.count_hdpe)
elif selected_arm == "Bras robotique 2 (PET)":
    st.metric("PET triés", st.session_state.count_pet)    
elif selected_arm == "Bras robotique 3 (PP)":
    st.metric("PP triés", st.session_state.count_pp)
elif selected_arm == "Bras robotique 4 (PS)":
    st.metric("PS triés", st.session_state.count_ps)
elif selected_arm == "tout les bras":
    st.metric("HDPE triés", st.session_state.count_hdpe)
    st.metric("PET triés", st.session_state.count_pet) 
    st.metric("PP triés", st.session_state.count_pp)
    st.metric("PS triés", st.session_state.count_ps)
import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile

# Charger le modèle YOLO 
model = YOLO('plastics-Waste-detection-and-Tracking\model.pt')
class_list = model.names
# Les classes définies 

# Initialisation des états dans la session Streamlit
if "video_open" not in st.session_state:
    st.session_state.video_open = False
# Fonction pour afficher le flux vidéo et détecter les objets avec YOLO
def open_video(video_path=None, use_camera=False):
    if use_camera:
        cap = cv2.VideoCapture(0)  
    else:
        cap = cv2.VideoCapture(video_path) 

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
        results = model.track(frame, persist=True) 

        # Ensure results are not empty
        if results[0].boxes.data is not None:
        # Get the detected boxes, their class indices, and track IDs
            boxes = results[0].boxes.xyxy
            if results[0].boxes.data is not None:
               # Assurez-vous que `id` n'est pas None
                if results[0].boxes.id is not None:
                   track_ids = results[0].boxes.id.int()
                else:
                   track_ids = []
            class_indices = results[0].boxes.cls.int()
            confidences = results[0].boxes.conf

        # Loop through each detected object
            for box, track_id, class_idx, conf in zip(boxes, track_ids, class_indices, confidences):
                x1, y1, x2, y2 = map(int, box)
                
                class_name = class_list[int(class_idx)]
            
                cv2.putText(frame, f"ID: {track_id} {class_name}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                

        # Convertir l'image pour affichage dans Streamlit
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame, channels="RGB", use_column_width=True)

    cap.release()

# Interface principale Streamlit
st.title("Interface de Supervision ")


# Ajouter un sélecteur pour choisir entre vidéo et caméra
option = st.selectbox("Choisissez la source de vidéo :", ["Vidéo Uploadée", "Caméra"])

if not st.session_state.video_open:
    if option == "Vidéo Uploadée":
        uploaded_file = st.file_uploader("Choisissez un fichier vidéo", type=["mp4", "avi", "mov", "mkv"])
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.read())
                video_path = tmp_file.name
            if st.button("Ouvrir la vidéo"):
                st.session_state.video_open = True
                open_video(video_path=video_path)
    elif option == "Caméra":
        if st.button("Ouvrir la caméra"):
            st.session_state.video_open = True
            open_video(use_camera=True)
else:
    if st.button("Fermer la vidéo / caméra"):
        st.session_state.video_open = False

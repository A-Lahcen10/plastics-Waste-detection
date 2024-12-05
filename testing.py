from  ultralytics import YOLO
import cv2
import math

# Initialisation de la caméra
video_path = "video4.mp4"
cap = cv2.VideoCapture(video_path)  # Utilise la caméra par défaut
cap.set(3, 1280)  # Définir la largeur de la vidéo
cap.set(4, 720)   # Définir la hauteur de la vidéo

# Chargement du modèle YOLO
model = YOLO("best11.pt")  # Chemin vers le fichier YOLOv8 pré-entraîné
classname = ["HDPE", "PET", "PP", "PS"]  # Classes définies
mycolor = (0, 0, 255)  # Couleur des boîtes

# Boucle principale pour capturer les images et effectuer la détection
while True:
    success, img = cap.read()  # Lecture d'une image depuis la caméra
    if not success:
        print("Erreur lors de la capture de l'image.")
        break

    # Détection des objets
    results = model(img, stream=True)

    # Traitement des résultats de la détection
    for result in results:
        # Extraire les boîtes de détection
        boxes = result.boxes
        for box in boxes:
            # Coordonnées de la boîte englobante
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # Dessin d'un rectangle autour de chaque objet détecté
            cv2.rectangle(img, (x1, y1), (x2, y2), mycolor, 3)

            # Confiance et classe
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])

            # Ajouter une étiquette avec le nom de l'objet détecté
            label = classname[cls]  # Accéder à la classe définie dans classname
            cv2.putText(
                img,
                f'{label} {conf}',
                (max(0, x1), max(35, y1)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,  # Taille du texte
                (255, 255, 255),  # Couleur du texte
                2  # Épaisseur du texte
            )

    # Affichage de l'image avec les boîtes de détection
    cv2.imshow("Real-Time Detection", img)

    # Quitter la boucle si la touche 'q' est pressée
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libération des ressources
cap.release()
cv2.destroyAllWindows()
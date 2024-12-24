Pipeline Step by step 
======================

L'idée générale de ce projet
----------------------------
L'idée générale de ce projet repose sur l'automatisation intelligente d'une ligne de tri des déchets à l'aide d'une caméra, d'un modèle de reconnaissance d'objets et d'un bras robotique. Concrètement, une caméra capture des vidéos en temps réel provenant de la ligne de tri. Ces vidéos servent d'entrée au système et sont analysées par un modèle d'intelligence artificielle spécialisé dans la détection d'objets. Le modèle identifie le type de déchet et transmet ces informations à une unité de prise de décision. En fonction du type détecté, cette unité détermine les actions à effectuer, notamment le tri et la manipulation des déchets par un bras robotique.
L'output de notre projet est plus spécifiquement une interface de supervision. Cette interface permet au superviseur de la ligne de tri de suivre en temps réel le déroulement du processus, notamment les objets détectés, les décisions prises par le système et l'état global de la ligne. Cela facilite la surveillance et l'intervention rapide en cas de besoin, tout en optimisant l'efficacité du tri dans un environnement industriel.

.. figure:: /Documentation/images/1.jpeg
   :width: 100%
   :alt: Alternative text for the image


1er étape : Collecte de données
-------------------------------

La collecte de données constitue la première étape essentielle du projet. Elle consiste à collecter manuellement des images et des vidéos sur différents types de déchets plastiques réels à partir de notre environnement proche. Voici quelques exemples de types de plastiques que nous avons collectés :

**PET :**

.. figure:: /Documentation/images/pet1.jpg
   :width: 40%
   :alt: Alternative text for the image

**HDPE :**

.. figure:: /Documentation/images/hdpe1.jpeg
   :width: 40%
   :alt: Alternative text for the image

**PP :**

.. figure:: /Documentation/images/pp1.jpg
   :width: 40%
   :alt: Alternative text for the image

**PS :**

.. figure:: /Documentation/images/ps1.jpeg
   :width: 40%
   :alt: Alternative text for the image


Ces données (images et vidéos) constituent une base solide pour entraîner notre modèle de reconnaissance d'objets.

2eme étape : Prétraitement des Données
--------------------------------------

**Nettoyage des données :** Suppression des images inutilisables, bruitées ou de mauvaise qualité.

**Annotation des données :** Étiquetage manuel des objets présents dans les images à l'aide de Roboflow.

.. figure:: /Documentation/images/roboflow.jpg
   :width: 50%
   :alt: Alternative text for the image

**Redimensionnement des images :** Adaptation de la taille des images pour qu'elles soient compatibles avec le modèle. Pour cela, nous avons choisi une dimension de 640x640.

**Équilibrage des classes :** Assurer un équilibre dans les types d'objets afin d'éviter un biais d'entraînement. Cela garantit que le modèle reconnaisse tous les types de déchets avec précision.

3eme étape : Choix du modèle
----------------------------

Sélection d'un modèle d'IA adapté à la tâche de détection d'objets. Nous avons choisi de travailler avec le modèle YOLOv11, car cette version est la plus précise par rapport aux autres versions de YOLO. Plus spécifiquement, nous avons opté pour YOLOv11n, car il présente le meilleur compromis entre précision et performance. En effet, ce modèle offre un temps de traitement des données inférieur à 2 ms par image, ce qui est essentiel pour garantir une détection en temps réel et une efficacité optimale sur une ligne de tri. Ainsi, le choix final s'est porté sur YOLOv11n pour sa rapidité et sa précision.

.. figure:: /Documentation/images/yolov11.jpeg
   :width: 100%
   :alt: Alternative text for the image

4eme étape : Entraînement du modèle
-----------------------------------
Utilisation des données prétraitées pour entraîner le modèle. L'entraînement permet au modèle d'apprendre à identifier et localiser les différents types de déchets en se basant sur les annotations fournies dans les images.
L'entraînement du modèle est une étape clé pour permettre à YOLOv11n d'apprendre à détecter et localiser les déchets plastiques dans les images. Lors de cette phase, nous utilisons les données prétraitées, c'est-à-dire les images annotées qui indiquent les positions des objets d'intérêt (les différents types de déchets plastiques). 
L'entraînement  nécessite plusieurs étapes essentielles. Voici le processus détaillé avec des explications à chaque étape :

**1-Monter Google Drive :**
La première étape consiste à monter Google Drive afin d'y accéder directement depuis Colab. Cela permet d'accéder aux Dataset et de stocker les fichiers de données et d'enregistrer les résultats d'entraînement dans le Drive. Le code pour cette étape est :

.. code-block:: python

    from google.colab import drive
    drive.mount('/content/drive')


**2-Installer la bibliothèque Ultralytics :**
La deuxième étape consiste à installer la bibliothèque Ultralytics, qui contient l'implémentation du modèle YOLOv11n, ainsi que ses outils nécessaires pour l'entraînement et l'évaluation. Nous installons la bibliothèque via la commande suivante :

.. code-block:: python
   
    %pip install ultralytics
    import ultralytics
    ultralytics.checks()



**3-Définir le chemin d'enregistrement et Création du fichier YAML :**
La troisième étape consiste à définir le chemin où le fichier YAML contenant la configuration de l'entraînement sera enregistré.Le fichier YAML contient des informations sur les chemins d'accès aux ensembles de données (d'entraînement et de validation), ainsi que les classes que le modèle doit apprendre à détecter.
et ce fichier YAML est essentiel car il spécifie les chemins des données d'entraînement et de validation, ainsi que les classes à détecter. Voici comment nous générons ce fichier YAML :

.. code-block:: python

    # Chemin où le fichier sera enregistré
    yaml_path = "/content/drive/MyDrive/object_detection/data/dataset.yaml"

    # Contenu du fichier YAML
    yaml_content = """\
    path: /content/drive/MyDrive/object_detection/data
    train: /content/drive/MyDrive/object_detection/data/train
    val: /content/drive/MyDrive/object_detection/data/valid

    nc: 4
    names: ["HDPE", "PET", "PP", "PS"]
    """

    # Création du fichier YAML
    try:
        with open(yaml_path, "w") as yaml_file:
            yaml_file.write(yaml_content)
            print(f"Fichier 'dataset.yaml' créé avec succès à l'emplacement : {yaml_path}")
        except Exception as e:
        print(f"Erreur lors de la création du fichier : {e}")




**4.Entraînement du modèle :**
La dernière étape consiste à entraîner le modèle YOLO11n en utilisant le fichier YAML comme configuration. Le modèle est entraîné pour 60 époques avec la commande suivante :

.. code-block:: python

    !yolo task=detect train model=yolo11n.pt data=/content/drive/MyDrive/object_detection/data/dataset.yaml epochs=60


Voila les resultats de l'entrainement :

.. figure:: /Documentation/images/resultatt.jpeg
   :width: 100%
   :alt: Alternative text for the image

Les graphiques montrent une convergence régulière des pertes et une amélioration constante des métriques. Le modèle atteint une haute précision et un rappel élevé, indiquant qu'il est bien entraîné pour la tâche de détection des objets.


5eme etape : Évaluation du modèle
---------------------------------

La validation de la performance du modèle est effectuée à l'aide de plusieurs métriques clés, telles que la précision, le rappel et le mAP (Mean Average Precision). Ces indicateurs permettent d'évaluer la capacité du modèle à détecter et classifier correctement les déchets plastiques dans de nouvelles images, garantissant ainsi une détection fiable et efficace.
Pour notre modèle. Voila le code pour faire la validation de modele :

.. code-block:: python

    !yolo task=detect mode=val model=/content/runs/detect/train/weights/best.pt data=/content/drive/MyDrive/object_detection/data/dataset.yaml


nous avons obtenu les résultats suivants lors de l'évaluation du modèle YOLOv11n sur différentes classes de déchets :

.. figure:: /Documentation/images/val.jpeg
   :width: 100%
   :alt: Alternative text for the image


Et après la phase de validation, nous avons réalisé un test du modèle pour évaluer ses performances. Voici quelques exemples qui illustrent sa précision

.. figure:: /Documentation/images/res1.jpeg
   :width: 50%
   :alt: Alternative text for the image


.. figure:: /Documentation/images/res2.jpeg
   :width: 50%
   :alt: Alternative text for the image


.. figure:: /Documentation/images/res3.jpeg
   :width: 40%
   :alt: Alternative text for the image

6eme etape : Déploiement du Modèle : Création d'une Interface de Supervision pour une Ligne de Tri des Déchets
--------------------------------------------------------------------------------------------------------------

Le déploiement du modèle inclut la créatiDans le cadre du développement d'une ligne de tri des déchets automatisée, une interface de supervision a été conçue pour visualiser et suivre en temps réel le processus de tri. Cette application, développée avec Streamlit, utilise un modèle YOLO préentraîné pour la détection et le suivi des objets. Le système permet de traiter des flux vidéo provenant d'une vidéo uploadée ou d'une caméra en direct et d'afficher les résultats en temps réel.

Les étapes ci-dessous détaillent les fonctionnalités du code et le rôle de chaque composant dans l'implémentation de cette solution.
**1.Chargement du Modèle YOLO**

Dans cette étape, nous chargeons notre modèle YOLO pré-entraîné pour détecter les objets dans une vidéo ou un flux en direct.

.. code-block:: python

    import cv2
    from ultralytics import YOLO
    import tempfile
    import streamlit as st

    # Charger le modèle YOLO 
    model = YOLO('plastics-Waste-detection-and-Tracking/model.pt')
    class_list = model.names

**cv2** : Utilisé pour capturer et traiter des flux vidéo.
**ultralytics.YOLO **: Permet d'utiliser un modèle YOLO pour la détection et le suivi des objets.
**tempfile** : Utilisé pour gérer temporairement les fichiers vidéos uploadés.
**streamlit** : Framework interactif pour créer une interface utilisateur.

**2.Fonction pour Ouvrir le Flux Vidéo et Détecter les Objets**

Cette fonction gère l'ouverture de la vidéo, qu'elle provienne d'un fichier téléchargé ou de la caméra. Elle traite ensuite le flux image par image pour appliquer la détection d'objets avec YOLO.

.. code-block:: python
    def open_video(video_path=None, use_camera=False):
      if use_camera:
         cap = cv2.VideoCapture(0)  
      else:
         cap = cv2.VideoCapture(video_path) 

      if not cap.isOpened():
         st.error("Impossible d'ouvrir la caméra. Vérifiez votre caméra.")
         return

      stframe = st.empty()  # Espace pour afficher le flux vidéo
      while st.session_state.video_open:
        ret, frame = cap.read()
        if not ret:
            st.warning("Impossible de lire le flux vidéo.")
            break

        # Détection d'objets avec YOLO
        results = model.track(frame, persist=True)


**3.Détection des Objets et Suivi**

YOLO renvoie les coordonnées des boîtes de détection, les classes détectées et les IDs des objets suivis. Le code permet d'afficher ces informations sur le flux vidéo.


.. code-block:: python

    if results[0].boxes.data is not None:
        boxes = results[0].boxes.xyxy
        if results[0].boxes.id is not None:
           track_ids = results[0].boxes.id.int()
        else:
           track_ids = []
        class_indices = results[0].boxes.cls.int()
        confidences = results[0].boxes.conf

        for box, track_id, class_idx, conf in zip(boxes, track_ids, class_indices, confidences):
            x1, y1, x2, y2 = map(int, box)
            class_name = class_list[int(class_idx)]
            cv2.putText(frame, f"ID: {track_id} {class_name}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #Conversion et Affichage de l'Image dans Streamlit :
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    stframe.image(frame, channels="RGB", use_column_width=True)


**4. Interface Principale avec Streamlit**

L'interface de supervision développée avec Streamlit et le modèle YOLO permet une visualisation efficace du processus de tri des déchets. cette application facilite le suivi en temps réel des performances du système de tri. Grâce aux fonctionnalités interactives et à la détection précise des objets, cette solution contribue à améliorer l'efficacité des lignes de tri automatisées.
Code :

.. code-block:: python 
   
   # Initialisation des états dans la session Streamlit
    if "video_open" not in st.session_state:
       st.session_state.video_open = False
    if "selected_arm" not in st.session_state:
       st.session_state.selected_arm = None

    st.title("Interface de Supervision ")
    st.write("Sélectionnez un bras robotique pour visualiser ou consulter le nombre trié.")

    # Sélection du bras robotique
    st.sidebar.header("Sélection du Bras Robotique")
    arm_options = ["Bras robotique 1 (HDPE)", "Bras robotique 2 (PP)", "Bras robotique 3 (PS)","Bras robotique 4 (PET)"]
    selected_arm = st.sidebar.selectbox("Choisissez un bras :", arm_options)

    # Mise à jour de l'état du bras sélectionné
    st.session_state.selected_arm = selected_arm
    st.header(f"Contrôle pour {selected_arm}")
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

   
Fonctionnalités de l'interface :L'interface permet de sélectionner facilement quel bras robotique doit être supervisé. Chaque bras est responsable du tri d'un type de déchet spécifique :

Le premier bras trie les déchets en HDPE.
Le deuxième bras trie les déchets en PET .
Le troisième bras trie les déchets en PP .
Le quatrième bras trie les déchets en PS .

.. figure:: /Documentation/images/interface1.jpeg
   :width: 50%
   :alt: Alternative text for the image

Dans le cadre de la supervision d'une ligne de tri réel, une fois le bras sélectionné, l'interface affiche en temps réel les informations sur son état de fonctionnement, permettant ainsi de vérifier si le tri se déroule correctement. Cela permet à l'opérateur de détecter rapidement toute anomalie ou dysfonctionnement. Cependant, dans notre cas, où il n'y a pas de ligne de tri réel, nous avons décidé d'utiliser deux options :

Télécharger une vidéo : Cette option permet de visionner une vidéo et d'afficher le résultat après la détection et le suivi des déchets dans la vidéo.

Ouvrir la caméra : Cette option permet d'activer la caméra pour capturer en temps réel les déchets présents devant celle-ci, afin d'analyser leur classification par le modèle et vérifier l'efficacité du tri en direct.

.. figure:: /Documentation/images/interface2.jpeg
   :width: 100%
   :alt: Alternative text for the image

Finalement, voici l'interface que nous avons développée, qui facilite la supervision et la visualisation d'une ligne de tri.

.. figure:: /Documentation/images/interfacee3.png
   :width: 130%
   :alt: Alternative text for the image


.. figure:: /Documentation/images/interface4.png
   :width: 30%
   :alt: Alternative text for the image
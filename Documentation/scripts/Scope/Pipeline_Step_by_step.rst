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


**-Précision (P) :** De manière générale, la précision est élevée, avec des valeurs exceptionnelles pour certaines classes comme PET et HDPE. Cela signifie que le modèle détecte de manière fiable les déchets sans générer trop de faux positifs.

**-Rappel (R) :** Le rappel élevé montre que le modèle parvient à capturer la majorité des instances des différentes classes de déchets, minimisant ainsi les faux négatifs. 

**-mAP50 et mAP50-95 :** Ces valeurs montrent que le modèle est capable de détecter et de localiser avec une grande précision, en particulier pour des classes comme PET (mAP50 = 0.995) et HDPE (mAP50 = 0.975).

Ces résultats démontrent que notre modèle YOLOv11n offre des performances solides et fiables pour la détection des déchets plastiques. L'évaluation est donc cruciale pour confirmer que le modèle répond aux exigences d'une application en temps réel, capable de détecter et classer les déchets plastiques dans des environnements industriels.

6eme etape : Déploiement du modèle
----------------------------------

Le déploiement du modèle inclut la création d'une interface de supervision visant à suivre en temps réel le processus de tri des déchets. Cette interface sera développée à l'aide de Streamlit, une bibliothèque Python permettant de créer facilement des applications web interactives. L'application aura pour objectifs principaux :
Visualisation en temps réel : L'interface permettra de suivre le processus de tri des déchets, offrant une vue instantanée du fonctionnement du modèle à mesure qu'il détecte et classe les objets sur la ligne de tri.
Suivi des performances : Elle assurera un suivi précis et interactif des performances du système de tri.
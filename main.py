import random
from sklearn.linear_model import RANSACRegressor
import alphashape
import random

from PyQt5.QtWidgets import QMessageBox

import open3d as o3d
from tp_lidar_ui import Ui_MainWindow
from PyQt5.QtWidgets import QFileDialog
from PyQt5 import QtWidgets
import laspy
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QMainWindow

# Création d'une classe qui hérite de Ui_MainWindow et QMainWindow
class MyMainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        # Appel du constructeur de la classe parent MyMainWindow.
        super(MyMainWindow, self).__init__()

        # Initialisation de l'interface utilisateur générée par Qt.
        self.setupUi(self)

        # Appel d'une méthode interne pour initialiser les composants de l'interface utilisateur.
        self.initialize_ui()

    def initialize_ui(self):
        """Initialise l'interface utilisateur."""

        # Connecte les boutons de l'interface à leurs fonctions respectives.
        self.connect_buttons()

        # Initialise les variables nécessaires pour le traitement des données LiDAR.
        self.lidar_points = None  # Points LiDAR
        self.coords_subset = None  # Sous-ensemble des coordonnées
        self.plane_parameters = None  # Paramètres du plan

    def connect_buttons(self):
        """Connecte les boutons à leurs fonctions respectives."""
        self.import_lidar_btn.clicked.connect(self.import_lidar_file)
        self.afficher_lidar_btn.clicked.connect(self.visualize_lidar_data)
        self.executer_calcul1_btn.clicked.connect(self.plan_parametres)
        self.ransac_btn.clicked.connect(self.plot_ransac_results)
        self.distance_btn.clicked.connect(self.calculate_distance_pp)
        self.afficher_plan_points_btn.clicked.connect(self.visualize_plane)
        self.extract_contours_btn.clicked.connect(self.visualisation_forme_alpha)


    def import_lidar_file(self):
        """Fonction permettant d'importer un fichier LiDAR."""

        # Configuration des options pour la boîte de dialogue de sélection de fichier
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog

        # Affichage de la boîte de dialogue et récupération du chemin du fichier sélectionné
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Import Lidar File",  # Titre de la boîte de dialogue
            "",  # Répertoire initial (vide dans cet exemple)
            "LAS Files (*.las);;All Files (*)",  # Filtres pour les types de fichiers
            options=options  # Options de la boîte de dialogue
        )

        # Vérification si un fichier a été sélectionné
        if file_name:
            print("Fichier LiDAR sélectionné :", file_name)

            # Lecture du fichier LAS à l'aide de la bibliothèque laspy
            try:
                las_file = laspy.read(file_name)

                # Traitement des données du fichier LAS
                print("Nombre de points dans le fichier LAS :", len(las_file.points))

                # Stockage des données LiDAR dans une variable de la classe
                self.lidar_points = las_file.points
                self.fromLasToArray(self.lidar_points)  # Conversion des données de laspy en un format adapté

                # Affichage d'une boîte de dialogue pour confirmer l'importation réussie
                msg = QMessageBox()
                msg.setWindowTitle("Succès")
                msg.setText("Le fichier LiDAR a été importé avec succès.")
                msg.exec_()

            except Exception as e:
                # Affichage d'une boîte de dialogue en cas d'erreur lors de l'importation
                msg = QMessageBox()
                msg.setWindowTitle("Erreur")
                msg.setText(f"Une erreur est survenue lors de l'importation du fichier LiDAR : {str(e)}")
                msg.exec_()

            except Exception as e:
                print("Erreur lors de la lecture du fichier LAS :", e)

            # Affichage du nom du fichier importé
            print(f"Fichier LiDAR importé : {file_name}")

    def visualize_lidar_data(self):
        """Visualise les données LiDAR si disponibles."""

        # Vérification si des points LiDAR sont disponibles
        if self.lidar_points is not None:
            print("En cours")

            # Création d'un objet PointCloud à partir des coordonnées sélectionnées
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.coords_subset)
            print("Affichage avec succès")
            # Visualisation de la nuage de points en 3D à l'aide d'Open3D
            o3d.visualization.draw_geometries([pcd])



        else:
            # Affichage d'un message si aucune donnée LiDAR n'est disponible
            print("Aucune donnée LiDAR disponible. Importez d'abord des données LiDAR.")
            msg = QMessageBox()
            msg.setWindowTitle("Erreur")
            msg.setText("Aucune donnée LiDAR disponible. Importez d'abord des données LiDAR.")
            msg.exec_()

    def calculate_distance_from_plane(self, plane_parameters):
        """Calcule la distance de chaque point au plan défini par les paramètres du plan."""

        # Vérification si les coordonnées sont disponibles et au moins 3 points sont présents
        if self.coords_subset is None or len(self.coords_subset) < 3:
            print(
                "Au moins 3 points LiDAR sont nécessaires pour calculer les paramètres du plan. Importez d'abord des données LiDAR.")
            return None

        # Extraction des paramètres du plan
        a, b, c, d = plane_parameters

        # Calcul des distances de chaque point au plan
        distances = np.abs(
            a * self.coords_subset[:, 0] + b * self.coords_subset[:, 1] + c * self.coords_subset[:, 2] + d
        ) / np.sqrt(a ** 2 + b ** 2 + c ** 2)

        return distances

    def calculate_distance_pp(self):
        """Calcule la distance des points au plan et affiche les résultats."""

        # Vérification si des données LiDAR sont disponibles
        if self.lidar_points is not None:

            # Vérification si au moins 3 points sont disponibles pour le calcul
            if self.coords_subset is None or len(self.coords_subset) < 3:
                print(
                    "Au moins 3 points LiDAR sont nécessaires pour calculer les paramètres du plan. Importez d'abord des données LiDAR.")
                return

            # Sélection de 3 indices aléatoires
            random_indices = random.sample(range(len(self.coords_subset)), 3)

            # Extraction des points sélectionnés
            selected_points = self.coords_subset[random_indices]

            # Vérification si les points sélectionnés sont non-collinéaires
            if self.are_points_non_collinear(selected_points[0], selected_points[1], selected_points[2]):

                # Calcul des paramètres du plan
                plane_parameters = self.calculate_plane_parameters(selected_points[0], selected_points[1],
                                                                   selected_points[2])

                # Calcul des distances des points au plan
                distances = self.calculate_distance_from_plane(plane_parameters)

                # Affichage des résultats en utilisant QMessageBox
                if distances is not None:
                    a, b, c, d = plane_parameters
                    msg = QMessageBox()
                    msg.setWindowTitle("Calcul de Distance")
                    msg.setText(
                        "Points Sélectionnés:\n{}\n\nParamètres du Plan (a, b, c, d):\n{} {} {} {}\n\nDistance au Plan:\n{:.4f}".format(
                            selected_points, a, b, c, d,
                            distances[0]))  # Affichage de la distance pour le premier point à titre d'exemple
                    msg.exec_()
            else:
                print("Les points sélectionnés sont collinéaires. Veuillez choisir des points non-collinéaires.")
        else:
            print("Aucune donnée LiDAR disponible. Importez d'abord des données LiDAR.")
            msg = QMessageBox()
            msg.setWindowTitle("Erreur")
            msg.setText("Aucune donnée LiDAR disponible. Importez d'abord des données LiDAR.")
            msg.exec_()

    def calculate_plane_parameters(self, point1, point2, point3):
        """Calcule les paramètres du plan à partir de trois points donnés."""

        # Calcul de deux vecteurs dans le plan
        vector1 = point2 - point1
        vector2 = point3 - point1

        # Calcul du vecteur normal au plan en utilisant le produit vectoriel
        normal_vector = np.cross(vector1, vector2)

        # Normalisation du vecteur normal
        normal_vector /= np.linalg.norm(normal_vector)

        # Calcul de la distance 'd' à partir de l'origine
        d = -np.dot(normal_vector, point1)

        # Retourne les paramètres du plan (a, b, c, d)
        return normal_vector[0], normal_vector[1], normal_vector[2], d

    def are_points_non_collinear(self,point1, point2, point3):
        """Vérifie si trois points sont non collinéaires."""
        cross_product = np.cross(point2 - point1, point3 - point1)
        return not np.allclose(cross_product, [0, 0, 0])

    def visualize_plane(self):
        """Visualise le plan défini par les paramètres du plan."""

        # Vérifie si des données Lidar sont disponibles
        if self.lidar_points is not None:
            # Vérifie si les paramètres du plan sont calculés
            if self.plane_parameters is None:
                print("Les paramètres du plan n'ont pas été calculés. Veuillez exécuter le calcul d'abord.")
                return
            print("En cours")

            # Extraction des paramètres du plan
            a, b, c, d = self.plane_parameters

            # Obtention des points valides qui appartient au plan
            valid_points_mask = self.get_valid_points()
            valid_points = self.coords_subset[valid_points_mask]

            # Calcul de l'étendue des points valides
            extent = [valid_points[:, 0].min(), valid_points[:, 0].max(),
                      valid_points[:, 1].min(), valid_points[:, 1].max(),
                      valid_points[:, 2].min(), valid_points[:, 2].max()]

            # Définition des sommets pour les deux triangles formant le plan avec une certaine épaisseur
            thickness = 0.05  # Vous pouvez ajuster cette valeur
            vertices = [
                [extent[0], extent[2], (-a * extent[0] - b * extent[2] - d) / c],  # bas-gauche
                [extent[1], extent[2], (-a * extent[1] - b * extent[2] - d) / c],  # bas-droite
                [extent[0], extent[3], (-a * extent[0] - b * extent[3] - d) / c],  # haut-gauche
                [extent[1], extent[3], (-a * extent[1] - b * extent[3] - d) / c]  # haut-droite
            ]

            # Définition des triangles avec épaisseur
            triangles = [
                [0, 1, 2],
                [1, 3, 2],
                [2, 3, 2],  # Triangle du bas pour l'épaisseur
                [1, 3, 3]  # Triangle latéral pour l'épaisseur
            ]

            # Création d'un objet TriangleMesh pour la visualisation
            plane_mesh = o3d.geometry.TriangleMesh()
            plane_mesh.vertices = o3d.utility.Vector3dVector(vertices)
            plane_mesh.triangles = o3d.utility.Vector3iVector(triangles)
            plane_mesh.paint_uniform_color([0.0, 0.0, 1.0])  # Couleur bleue

            # Conversion des points valides en nuage de points pour la visualisation
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(valid_points)

            # Configuration de la visualisation
            vis = o3d.visualization.Visualizer()
            vis.create_window()

            # Ajout des géométries
            vis.add_geometry(pcd)  # Ajoute le nuage de points
            vis.add_geometry(plane_mesh)  # Ajoute le maillage du plan

            # Configuration de la caméra pour visualiser le plan
            ctr = vis.get_view_control()
            ctr.set_lookat([0, 0, 0])  # Ajuster si nécessaire
            ctr.set_front([0, 0, 1])  # Ajuster si nécessaire
            ctr.set_zoom(0.8)  # Ajustement du niveau de zoom
            print("Affichage aves succès")

            # Exécute la visualisation
            vis.run()
            vis.destroy_window()
        else:
            print("Aucune donnée Lidar disponible. Importez d'abord les données Lidar.")
            msg = QMessageBox()
            msg.setWindowTitle("Erreur")
            msg.setText("Aucune donnée LiDAR disponible. Importez d'abord des données LiDAR.")
            msg.exec_()

    def plan_parametres(self):
        """Calcule les paramètres du plan en fonction des points sélectionnés."""

        # Vérification si des données Lidar ont été importées
        if self.lidar_points is None:
            msg = QMessageBox()
            msg.setWindowTitle("Erreur")
            msg.setText("Veuillez importer votre fichier Lidar.")
            msg.exec_()
            return

        # Si des points ont été sélectionnés et qu'il y en a au moins trois
        elif self.coords_subset is not None and len(self.coords_subset) >= 3:
            # Sélection de 3 indices aléatoires
            random_indices = random.sample(range(len(self.coords_subset)), 3)

            # Extraction des points sélectionnés
            selected_points = self.coords_subset[random_indices]

            # Vérification si les points sélectionnés sont non-collinéaires
            if self.are_points_non_collinear(selected_points[0], selected_points[1], selected_points[2]):
                # Calcul des paramètres du plan à partir des points sélectionnés
                self.plane_parameters = self.calculate_plane_parameters(selected_points[0], selected_points[1],
                                                                        selected_points[2])

                # Affichage des paramètres du plan dans une boîte de dialogue
                a, b, c, d = self.plane_parameters
                msg = QMessageBox()
                msg.setWindowTitle("Paramètres du Plan")
                msg.setText(
                    "Paramètres du Plan (a, b, c, d):\n{} {} {} {}".format(a, b, c, d)
                )
                msg.exec_()

            else:
                msg = QMessageBox()
                msg.setWindowTitle("Erreur")
                msg.setText("Les points sélectionnés sont collinéaires. Veuillez choisir des points non-collinéaires.")
                msg.exec_()

        # Si des points ont été sélectionnés mais qu'il y en a moins de trois
        elif self.coords_subset is not None and len(self.coords_subset) < 3:
            msg = QMessageBox()
            msg.setWindowTitle("Erreur")
            msg.setText(
                "Au moins 3 points Lidar sont nécessaires pour calculer le plan. Importez d'abord les données Lidar.")
            msg.exec_()
    def get_valid_points(self, tolerance=0.00001):
        """
        Renvoie les points valides par rapport au plan avec une tolérance donnée.

        Paramètres:
        - tolerance (float): La tolérance utilisée pour déterminer si un point est valide par rapport au plan.

        Retour:
        - numpy array: Un tableau booléen indiquant quels points sont valides par rapport au plan.
        """

        # Extraction des paramètres du plan
        a, b, c, d = self.plane_parameters

        # Calcul de la distance de chaque point au plan
        distances = np.abs(
            a * self.coords_subset[:, 0] + b * self.coords_subset[:, 1] + c * self.coords_subset[:, 2] + d) / np.sqrt(
            a ** 2 + b ** 2 + c ** 2)

        # Vérification si les distances sont dans la tolérance spécifiée
        return distances <= tolerance

    def fit_plane_ransac(self):
        """
        Ajuste un plan aux données en utilisant l'algorithme RANSAC.

        L'algorithme RANSAC (Random Sample Consensus) est utilisé pour ajuster un modèle (dans ce cas, un plan)
        aux données en présence de valeurs aberrantes.

        Retour:
        - inlier_points (numpy array): Les points qui appartiennent à l'ajustement du plan.
        - outlier_points (numpy array): Les points qui sont considérés comme des valeurs aberrantes par rapport au plan.
        """

        # Mise en forme des données pour la régression RANSAC
        X = self.coords_subset[:, :2]  # Prend les deux premières colonnes comme X
        y = self.coords_subset[:, 2]  # Prend la troisième colonne comme y

        # Création d'un régresseur RANSAC
        ransac = RANSACRegressor()

        # Ajustement du modèle aux données
        ransac.fit(X, y)

        # Récupération du masque des points inliers
        inlier_mask = ransac.inlier_mask_

        # Extraction des points inliers et outliers basés sur le masque
        inlier_points = self.coords_subset[inlier_mask]
        outlier_points = self.coords_subset[~inlier_mask]

        # Retourne les points inliers et outliers
        return inlier_points, outlier_points

    def plot_ransac_results(self):
        """
        Visualise les résultats de l'ajustement RANSAC.

        Cette fonction ajuste un plan aux données à l'aide de l'algorithme RANSAC, puis visualise les points
        inliers et outliers séparément en utilisant la bibliothèque Open3D.

        Si aucun jeu de données Lidar n'est disponible, un message est affiché.

        Affichage :
        - Une fenêtre contenant les points inliers identifiés après l'ajustement RANSAC.
        - Une autre fenêtre contenant les points outliers détectés après l'ajustement RANSAC.
        """

        if self.lidar_points is not None:
            print("En cours")
            # Appel à la fonction pour obtenir les points inliers et outliers
            inlier_points, outlier_points = self.fit_plane_ransac()

            # Conversion des tableaux numpy en format de nuage de points Open3D
            inlier_pcd = o3d.geometry.PointCloud()
            inlier_pcd.points = o3d.utility.Vector3dVector(inlier_points)

            outlier_pcd = o3d.geometry.PointCloud()
            outlier_pcd.points = o3d.utility.Vector3dVector(outlier_points)
            print("Affichage avec succès")
            # Visualisation en utilisant Open3D
            o3d.visualization.draw_geometries([inlier_pcd], window_name='Inliers', width=800, height=600)
            o3d.visualization.draw_geometries([outlier_pcd], window_name='Outliers', width=800, height=600)


        else:
            print("Aucune donnée Lidar disponible. Importez d'abord les données Lidar.")
            msg = QMessageBox()
            msg.setWindowTitle("Erreur")
            msg.setText("Aucune donnée LiDAR disponible. Importez d'abord des données LiDAR.")
            msg.exec_()

    def fromLasToArray(self, lidar):
        """
        Convertit les données Lidar en tableau numpy.

        Cette fonction prend en entrée un objet Lidar et convertit ses coordonnées (x, y, z) en un tableau numpy.
        Elle stocke également ces coordonnées complètes dans un attribut et sélectionne un sous-ensemble des données
        pour un traitement ultérieur.

        Paramètres :
        - lidar : Objet contenant les données Lidar, typiquement issu d'un fichier LAS.

        Stockage :
        - self.coords_subset :tableau numpy contenant toutes les coordonnées (x, y, z).
        """

        # Conversion des coordonnées (x, y, z) de l'objet Lidar en un tableau numpy
        self.coords_subset = np.vstack((lidar.x, lidar.y, lidar.z)).transpose()

    def compute_alpha_shape(self, points, alpha=1.0):
        """
        Calcule la forme alpha des points.

        Cette fonction prend en entrée un ensemble de points et calcule la forme alpha en utilisant une valeur
        d'alpha spécifique. La forme alpha est une représentation géométrique qui tente de capturer les contours
        et les structures non-linéaires d'un nuage de points.

        Paramètres :
        - points : Tableau numpy contenant les coordonnées (x, y, z) des points.
        - alpha : Valeur d'alpha utilisée pour le calcul de la forme alpha. Par défaut, alpha est fixé à 1.0.

        Retour :
        - alpha_shape : Objet représentant la forme alpha calculée pour les points donnés.
        """
        alpha_shape = alphashape.alphashape(points[:, :2], alpha)  # Calcul de la forme alpha 2D en utilisant x et y
        return alpha_shape

    def visualisation_forme_alpha(self):
        """
        Affiche la forme alpha de la point cloud.
        ... [reste du docstring inchangé] ...
        """
        if self.lidar_points is not None:
            print("Démarrage de la visualisation...")
            print("Note :")
            print("6000 points sélectionnés aléatoirement pour une meilleure clarté visuelle.")

            # Sélection aléatoire de 6000 indices parmi les coordonnées disponibles
            indices_selectionnes = random.sample(range(len(self.coords_subset)), 5000)

            # Extraction d'un sous-ensemble de 6000 points basé sur les indices aléatoires
            points_selectionnes = self.coords_subset[indices_selectionnes]

            # Calcul de la forme alpha pour ce sous-ensemble
            forme_alpha = self.compute_alpha_shape(points_selectionnes)

            # Affichage de la forme alpha et des points correspondants
            plt.figure(figsize=(8, 6))
            plt.scatter(points_selectionnes[:, 0], points_selectionnes[:, 1], color='blue', label='Points')
            coordonnees_exterieures = forme_alpha.exterior.xy
            plt.plot(coordonnees_exterieures[0], coordonnees_exterieures[1], color='red', label='Forme Alpha')
            plt.title('Visualisation de la Forme Alpha')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.legend()
            print("Visualisation terminée avec succès.")
            plt.show()
        else:
            print("Aucune donnée LiDAR n'est disponible pour la visualisation.")
            message_erreur = QMessageBox()
            message_erreur.setWindowTitle("Erreur de Données")
            message_erreur.setText("Veuillez importer des données LiDAR pour continuer.")
            message_erreur.exec_()


# Création de l'application
app = QtWidgets.QApplication([])

# Création d'une instance de votre MainWindow personnalisée
main_window = MyMainWindow()

# Affichage de la fenêtre principale
main_window.show()

# Démarrage de la boucle d'événements de l'application
import sys
sys.exit(app.exec_())
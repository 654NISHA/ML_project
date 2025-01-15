import os
import sys
from dataclasses import dataclass

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def perform_pca(self, data, n_components):
        """
        Perform Principal Component Analysis (PCA) on the given data.
        """
        try:
            logging.info(f"Performing PCA with {n_components} components.")
            pca = PCA(n_components=n_components)
            pca_data = pca.fit_transform(data)
            logging.info("PCA completed.")
            return pca, pca_data
        except Exception as e:
            raise CustomException(e, sys)

    def perform_kmeans(self, data, n_clusters):
        """
        Perform KMeans clustering on the given data.
        """
        try:
            logging.info(f"Performing KMeans clustering with {n_clusters} clusters.")
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            kmeans_labels = kmeans.fit_predict(data)
            silhouette_avg = silhouette_score(data, kmeans_labels)
            logging.info(f"KMeans clustering completed with silhouette score: {silhouette_avg}")
            return kmeans, kmeans_labels, silhouette_avg
        except Exception as e:
            raise CustomException(e, sys)

    def perform_hierarchical_clustering(self, data, n_clusters):
        """
        Perform Agglomerative Clustering on the given data.
        """
        try:
            logging.info(f"Performing hierarchical clustering with {n_clusters} clusters.")
            hc = AgglomerativeClustering(n_clusters=n_clusters)
            hc_labels = hc.fit_predict(data)
            logging.info("Hierarchical clustering completed.")
            return hc, hc_labels
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_trainer(self, train_array, test_array, preprocessor_path):
        """
        Orchestrates the model training process.
        """
        try:
            logging.info("Starting model training process.")

            # Load training and testing data
            train_data, test_data = train_array, test_array

            # Perform PCA on training data
            pca, train_pca_data = self.perform_pca(train_data, n_components=2)

            # Perform KMeans clustering on PCA-transformed training data
            kmeans, kmeans_labels, silhouette_avg = self.perform_kmeans(train_pca_data, n_clusters=2)

            # Perform hierarchical clustering on PCA-transformed training data
            hc, hc_labels = self.perform_hierarchical_clustering(train_pca_data, n_clusters=2)

            # Save the trained model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj={
                    "pca": pca,
                    "kmeans": kmeans,
                    "hierarchical": hc
                }
            )

            logging.info("Model training process completed and model saved.")
            return {
                "pca": pca,
                "kmeans": kmeans,
                "kmeans_labels": kmeans_labels,
                "hierarchical": hc,
                "hc_labels": hc_labels,
                "silhouette_score": silhouette_avg
            }
        except Exception as e:
            raise CustomException(e, sys)

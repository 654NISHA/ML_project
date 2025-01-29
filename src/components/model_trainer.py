import os
import sys
from dataclasses import dataclass

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
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
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
            kmeans_labels = kmeans.fit_predict(data)
            silhouette_avg = silhouette_score(data, kmeans_labels)
            logging.info(f"KMeans clustering completed with silhouette score: {silhouette_avg}")
            return kmeans, kmeans_labels, silhouette_avg
        except Exception as e:
            raise CustomException(e, sys)

    def perform_hierarchical_clustering(self, data, n_clusters):
        """
        Perform Agglomerative Clustering and return the model, labels, and silhouette score.
        """
        try:
            logging.info(f"Performing hierarchical clustering with {n_clusters} clusters.")
            clustering_model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
            labels = clustering_model.fit_predict(data)
            silhouette_avg = silhouette_score(data, labels) if n_clusters > 1 else None
            logging.info(f"Hierarchical clustering completed with silhouette score: {silhouette_avg}")
            return clustering_model, labels, silhouette_avg
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_trainer(self, train_array, test_array):
        """
        Orchestrates the model training process with configurable parameters.
        """
        try:
            n_clusters = 2 
            n_components = 2
            logging.info(f"Starting model training process with n_clusters={n_clusters}, n_components={n_components}")

            # Perform PCA on training and testing data
            pca, train_pca_data = self.perform_pca(train_array, n_components=n_components)
            _, test_pca_data = self.perform_pca(test_array, n_components=n_components)

            # Ensure that the PCA transformation has reduced the data to the correct number of components
            logging.info(f"Train PCA Data Shape: {train_pca_data.shape}, Test PCA Data Shape: {test_pca_data.shape}")

            pca_path = os.path.join('artifacts', 'pca.pkl')
            save_object(file_path=pca_path, obj=pca)
            logging.info("PCA transformer saved successfully.")

            # Perform KMeans clustering on PCA-transformed training data
            kmeans, kmeans_train_labels, kmeans_train_silhouette_score = self.perform_kmeans(train_pca_data, n_clusters=n_clusters)
            kmeans_test_labels = kmeans.predict(test_pca_data)
            kmeans_test_silhouette_score = silhouette_score(test_pca_data, kmeans_test_labels)

            logging.info(f"KMeans Train Silhouette Score: {kmeans_train_silhouette_score}, Test Silhouette Score: {kmeans_test_silhouette_score}")

            # Perform hierarchical clustering on PCA-transformed training data
            hc, hc_train_labels, hc_train_silhouette_score = self.perform_hierarchical_clustering(train_pca_data, n_clusters=n_clusters)
            hc_test_labels = hc.fit_predict(test_pca_data) 
            hc_test_silhouette_score = silhouette_score(test_pca_data, hc_test_labels)

            logging.info(f"Hierarchical Train Silhouette Score: {hc_train_silhouette_score}, Test Silhouette Score: {hc_test_silhouette_score}")

            # Selecting the best model based on test silhouette score
            if kmeans_test_silhouette_score >= hc_test_silhouette_score:
                best_model = kmeans
                best_silhouette_score = kmeans_test_silhouette_score
                best_model_name = "KMeans"
            else:
                best_model = hc
                best_silhouette_score = hc_test_silhouette_score
                best_model_name = "Hierarchical Clustering"

            logging.info(f"Best model selected: {best_model_name} with silhouette score: {best_silhouette_score}")

            # Save the trained models and PCA
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            logging.info("Model training process completed and models saved.")

            # Return results including silhouette scores
            return {
                "best_model_name": best_model_name,
                "best_model": best_model,
                "best_silhouette_score": best_silhouette_score
            }
        except Exception as e:
            raise CustomException(e, sys)

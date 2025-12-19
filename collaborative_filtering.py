import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
import warnings

warnings.filterwarnings("ignore")


class CollaborativeFiltering:
    def __init__(self, user_item_matrix):
        # Convert COO → CSR once (critical)
        if hasattr(user_item_matrix, "tocoo"):
            self.user_item_matrix = user_item_matrix.tocsr()
        else:
            self.user_item_matrix = user_item_matrix

        self.user_similarity = None
        self.item_similarity = None
        self.predicted_ratings = None

    # =========================
    # Helpers
    # =========================
    def _get_user_ratings(self, user_idx):
        """Safe access for pandas or sparse matrix"""
        if hasattr(self.user_item_matrix, "toarray"):
            return self.user_item_matrix[user_idx].toarray().flatten()
        return self.user_item_matrix.iloc[user_idx].values

    def _get_dense_matrix(self):
        if hasattr(self.user_item_matrix, "toarray"):
            return self.user_item_matrix.toarray()
        return self.user_item_matrix.values

    # =========================
    # Similarity calculations
    # =========================
    def calculate_user_similarity(self):
        self.user_similarity = cosine_similarity(self._get_dense_matrix())
        return self.user_similarity

    def calculate_item_similarity(self):
        self.item_similarity = cosine_similarity(self._get_dense_matrix().T)
        return self.item_similarity

    # =========================
    # User-based CF
    # =========================
    def user_based_recommendations(self, user_id, n_recommendations=5):
        if self.user_similarity is None:
            self.calculate_user_similarity()

        user_idx = user_id - 1
        similar_users = self.user_similarity[user_idx]
        user_ratings = self._get_user_ratings(user_idx)
        ratings_matrix = self._get_dense_matrix()

        predicted_ratings = np.zeros(len(user_ratings))

        for item_idx in range(len(user_ratings)):
            if user_ratings[item_idx] == 0:
                numerator = 0
                denominator = 0

                for other_user_idx in range(len(similar_users)):
                    if other_user_idx != user_idx:
                        rating = ratings_matrix[other_user_idx, item_idx]
                        if rating > 0:
                            numerator += similar_users[other_user_idx] * rating
                            denominator += abs(similar_users[other_user_idx])

                if denominator > 0:
                    predicted_ratings[item_idx] = numerator / denominator

        top = np.argsort(predicted_ratings)[::-1][:n_recommendations]
        return top, predicted_ratings[top]

    # =========================
    # Item-based CF
    # =========================
    def item_based_recommendations(self, user_id, n_recommendations=5):
        if self.item_similarity is None:
            self.calculate_item_similarity()

        user_idx = user_id - 1
        user_ratings = self._get_user_ratings(user_idx)
        rated_items = np.where(user_ratings > 0)[0]

        predicted_ratings = np.zeros(len(user_ratings))

        for item_idx in range(len(user_ratings)):
            if user_ratings[item_idx] == 0:
                numerator = 0
                denominator = 0

                for rated_item_idx in rated_items:
                    sim = self.item_similarity[item_idx, rated_item_idx]
                    numerator += sim * user_ratings[rated_item_idx]
                    denominator += abs(sim)

                if denominator > 0:
                    predicted_ratings[item_idx] = numerator / denominator

        top = np.argsort(predicted_ratings)[::-1][:n_recommendations]
        return top, predicted_ratings[top]

    # =========================
    # Matrix Factorization
    # =========================
    def matrix_factorization(self, n_factors=15):
        R = self._get_dense_matrix()

        n_users, n_items = R.shape
        k = max(2, min(n_factors, min(n_users, n_items) - 1))

        try:
            U, sigma, Vt = svds(R, k=k)
            self.predicted_ratings = U @ np.diag(sigma) @ Vt
            return self.predicted_ratings
        except Exception as e:
            print(f"[ERROR] SVD failed: {e}")
            self.predicted_ratings = np.random.rand(n_users, n_items) * 5
            return self.predicted_ratings

    # =========================
    # MF Recommendations
    # =========================
    def mf_recommendations(self, user_id, n_recommendations=5):
        if self.predicted_ratings is None:
            self.matrix_factorization()

        user_idx = user_id - 1
        actual = self._get_user_ratings(user_idx)
        predicted = self.predicted_ratings[user_idx]

        predicted[actual > 0] = 0
        top = np.argsort(predicted)[::-1][:n_recommendations]
        return top, predicted[top]

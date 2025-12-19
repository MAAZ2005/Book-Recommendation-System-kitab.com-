import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

class ContentBasedFiltering:
    def __init__(self, books_df):
        self.books_df = books_df
        self.tfidf_matrix = None
        self.content_similarity = None
        self.feature_vectors = None
        
    def prepare_features(self):
        """Prepare features for content-based filtering"""
        # Create a combined feature string
        self.books_df['features'] = (
            self.books_df['title'].fillna('') + ' ' +
            self.books_df['author'].fillna('') + ' ' +
            self.books_df['genre'].fillna('')
        )
        
        # Use TF-IDF for text features
        tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = tfidf.fit_transform(self.books_df['features'])
        
        # Normalize numerical features
        scaler = MinMaxScaler()
        year_normalized = scaler.fit_transform(self.books_df[['year']])
        rating_normalized = scaler.fit_transform(self.books_df[['rating']])
        
        # Combine all features
        from scipy.sparse import hstack
        self.feature_vectors = hstack([
        self.tfidf_matrix,
         year_normalized,
         rating_normalized
       ]).tocsr()   

        
        # Calculate similarity matrix
        self.content_similarity = cosine_similarity(self.feature_vectors)
        
        return self.content_similarity
    
    def get_similar_books(self, book_id, n_recommendations=5):
        """Get books similar to a given book"""
        if self.content_similarity is None:
            self.prepare_features()
        
        book_idx = book_id - 1  # Assuming book IDs start from 1
        similarity_scores = list(enumerate(self.content_similarity[book_idx]))
        
        # Sort by similarity score
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        
        # Get top N similar books (excluding the book itself)
        top_indices = [i for i, _ in similarity_scores[1:n_recommendations+1]]
        top_scores = [score for _, score in similarity_scores[1:n_recommendations+1]]
        
        return top_indices, top_scores
    
    def recommend_based_on_history(self, rated_books, n_recommendations=5):
        """Recommend books based on user's rating history"""
        if self.content_similarity is None:
            self.prepare_features()
        
        # Create user profile based on rated books
        user_profile = np.zeros(self.feature_vectors.shape[1])
        
        for book_id, rating in rated_books:
            book_idx = book_id - 1
            user_profile += self.feature_vectors[book_idx].toarray().flatten() * rating
        
        # Normalize user profile
        if len(rated_books) > 0:
            user_profile /= len(rated_books)
        
        # Calculate similarity with all books
        similarities = cosine_similarity([user_profile], self.feature_vectors)
        
        # Get top recommendations (excluding already rated books)
        rated_indices = [book_id - 1 for book_id, _ in rated_books]
        similarities = similarities.flatten()
        
        for idx in rated_indices:
            similarities[idx] = -1  # Mark rated books
        
        top_indices = np.argsort(similarities)[::-1][:n_recommendations]
        top_scores = similarities[top_indices]
        
        return top_indices, top_scores
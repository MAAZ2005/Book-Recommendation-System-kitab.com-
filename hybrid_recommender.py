import numpy as np

class HybridRecommender:
    def __init__(self, collaborative_filter, content_based_filter, data_loader):
        self.cf = collaborative_filter
        self.cbf = content_based_filter
        self.data_loader = data_loader
        
    def hybrid_recommendations(self, user_id, n_recommendations=5, alpha=0.5):
        """Combine collaborative and content-based filtering"""
        # Get collaborative filtering recommendations
        cf_indices, cf_scores = self.cf.mf_recommendations(user_id, n_recommendations * 2)
        
        # Get user's rated books for content-based filtering
        user_ratings = self.data_loader.get_user_ratings(user_id)
        rated_books = list(zip(user_ratings['book_id'], user_ratings['rating'])) if user_ratings is not None else []
        
        if rated_books:
            cbf_indices, cbf_scores = self.cbf.recommend_based_on_history(rated_books, n_recommendations * 2)
        else:
            # If no ratings, use popular books as fallback
            cbf_indices = list(range(n_recommendations * 2))
            cbf_scores = [1.0] * (n_recommendations * 2)
        
        # Combine scores
        combined_scores = {}
        
        # Add CF scores
        for idx, score in zip(cf_indices, cf_scores):
            book_id = idx + 1
            combined_scores[book_id] = combined_scores.get(book_id, 0) + alpha * score
        
        # Add CBF scores
        for idx, score in zip(cbf_indices, cbf_scores):
            book_id = idx + 1
            combined_scores[book_id] = combined_scores.get(book_id, 0) + (1 - alpha) * score
        
        # Sort by combined score
        sorted_books = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Get top N recommendations
        top_recommendations = []
        for book_id, score in sorted_books[:n_recommendations]:
            book_info = self.data_loader.get_book_info(book_id)
            if book_info:
                top_recommendations.append({
                    'book_id': book_id,
                    'title': book_info['title'],
                    'author': book_info['author'],
                    'genre': book_info['genre'],
                    'score': round(score, 3)
                })
        
        return top_recommendations
    
    def cold_start_recommendations(self, n_recommendations=5):
        """Recommendations for new users (cold start problem)"""
        # Return popular books based on average rating
        if self.data_loader.ratings_df is not None:
            book_ratings = self.data_loader.ratings_df.groupby('book_id')['rating'].agg(['mean', 'count'])
            book_ratings = book_ratings[book_ratings['count'] >= 5]  # Only books with at least 5 ratings
            
            # Sort by rating (weighted by number of ratings)
            book_ratings['weighted_score'] = book_ratings['mean'] * np.log1p(book_ratings['count'])
            popular_books = book_ratings.sort_values('weighted_score', ascending=False).head(n_recommendations)
            
            recommendations = []
            for book_id in popular_books.index:
                book_info = self.data_loader.get_book_info(book_id)
                if book_info:
                    recommendations.append({
                        'book_id': book_id,
                        'title': book_info['title'],
                        'author': book_info['author'],
                        'genre': book_info['genre'],
                        'avg_rating': round(popular_books.loc[book_id, 'mean'], 2),
                        'rating_count': int(popular_books.loc[book_id, 'count'])
                    })
            
            return recommendations
        return []
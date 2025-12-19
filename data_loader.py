import pandas as pd
import numpy as np

class DataLoader:
    def __init__(self):
        self.books_df = None
        self.ratings_df = None
        self.user_item_matrix = None
        
    def load_data(self, books_path='data/books.csv', ratings_path='data/ratings.csv'):
        """Load books and ratings data"""
        try:
            self.books_df = pd.read_csv(books_path)
            self.ratings_df = pd.read_csv(ratings_path)
            print(f"Loaded {len(self.books_df)} books and {len(self.ratings_df)} ratings")
            return True
        except FileNotFoundError as e:
            print(f"Error loading data: {e}")
            return False
    
    def create_user_item_matrix(self):
        """Create user-item rating matrix"""
        if self.ratings_df is not None:
            self.user_item_matrix = self.ratings_df.pivot_table(
                index='user_id',
                columns='book_id',
                values='rating',
                fill_value=0
            )
            return self.user_item_matrix
        return None
    
    def get_book_info(self, book_id):
        """Get book information by ID"""
        if self.books_df is not None:
            book_info = self.books_df[self.books_df['book_id'] == book_id]
            if not book_info.empty:
                return book_info.iloc[0].to_dict()
        return None
    
    def get_user_ratings(self, user_id):
        """Get ratings by a specific user"""
        if self.ratings_df is not None:
            return self.ratings_df[self.ratings_df['user_id'] == user_id]
        return None
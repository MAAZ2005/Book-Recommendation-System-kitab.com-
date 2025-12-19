import os
import sys
from data_loader import DataLoader
from collaborative_filtering import CollaborativeFiltering
from content_based import ContentBasedFiltering
from hybrid_recommender import HybridRecommender
import pandas as pd

class BookRecommendationSystem:
    def __init__(self):
        self.data_loader = DataLoader()
        self.cf = None
        self.cbf = None
        self.hybrid = None
        
    def initialize(self):
        """Initialize the recommendation system"""
        print("=" * 50)
        print("BOOK RECOMMENDATION SYSTEM")
        print("=" * 50)
        
        # Load data
        if not self.data_loader.load_data():
            print("Error: Could not load data files.")
            print("Please run sample_data_generator.py first.")
            return False
        
        # Create user-item matrix
        user_item_matrix = self.data_loader.create_user_item_matrix()
        
        # Initialize recommender systems
        self.cf = CollaborativeFiltering(user_item_matrix)
        self.cbf = ContentBasedFiltering(self.data_loader.books_df)
        self.hybrid = HybridRecommender(self.cf, self.cbf, self.data_loader)
        
        print("System initialized successfully!")
        print(f"Number of users: {len(user_item_matrix)}")
        print(f"Number of books: {len(self.data_loader.books_df)}")
        return True
    
    def display_book_info(self, book_id):
        """Display information about a book"""
        book_info = self.data_loader.get_book_info(book_id)
        if book_info:
            print(f"\n Book Information:")
            print(f"   Title: {book_info['title']}")
            print(f"   Author: {book_info['author']}")
            print(f"   Genre: {book_info['genre']}")
            print(f"   Year: {book_info['year']}")
            print(f"   Average Rating: {book_info['rating']}")
        else:
            print(f"Book with ID {book_id} not found.")
    
    def display_user_ratings(self, user_id):
        """Display a user's ratings"""
        user_ratings = self.data_loader.get_user_ratings(user_id)
        if user_ratings is not None and len(user_ratings) > 0:
            print(f"\n User {user_id}'s Ratings:")
            for _, row in user_ratings.iterrows():
                book_info = self.data_loader.get_book_info(row['book_id'])
                if book_info:
                    print(f"   - {book_info['title']}: {row['rating']}/5")
        else:
            print(f"No ratings found for user {user_id}")
    
    def collaborative_recommendations(self, user_id):
        """Generate collaborative filtering recommendations"""
        print(f"\n Collaborative Filtering Recommendations for User {user_id}:")
        
        # User-based recommendations
        indices, scores = self.cf.user_based_recommendations(user_id, 3)
        print("\n   User-Based Recommendations:")
        for idx, score in zip(indices, scores):
            book_id = idx + 1
            book_info = self.data_loader.get_book_info(book_id)
            if book_info:
                print(f"   - {book_info['title']} (Score: {score:.3f})")
        
        # Item-based recommendations
        indices, scores = self.cf.item_based_recommendations(user_id, 3)
        print("\n   Item-Based Recommendations:")
        for idx, score in zip(indices, scores):
            book_id = idx + 1
            book_info = self.data_loader.get_book_info(book_id)
            if book_info:
                print(f"   - {book_info['title']} (Score: {score:.3f})")
    
    def content_based_recommendations(self, book_id):
        """Generate content-based recommendations"""
        print(f"\n Content-Based Recommendations similar to Book {book_id}:")
        
        indices, scores = self.cbf.get_similar_books(book_id, 5)
        for idx, score in zip(indices, scores):
            book_id = idx + 1
            book_info = self.data_loader.get_book_info(book_id)
            if book_info:
                print(f"   - {book_info['title']} (Similarity: {score:.3f})")
    
    def hybrid_recommendations(self, user_id):
        """Generate hybrid recommendations"""
        print(f"\n Hybrid Recommendations for User {user_id}:")
        
        recommendations = self.hybrid.hybrid_recommendations(user_id, 5)
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec['title']}")
            print(f"      Author: {rec['author']}, Genre: {rec['genre']}")
            print(f"      Recommendation Score: {rec['score']}")
    
    def cold_start_recommendations(self):
        """Generate recommendations for new users"""
        print(f"\n Recommendations for New Users (Cold Start):")
        
        recommendations = self.hybrid.cold_start_recommendations(5)
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec['title']}")
            print(f"      Author: {rec['author']}, Genre: {rec['genre']}")
            print(f"      Avg Rating: {rec['avg_rating']}/5 ({rec['rating_count']} ratings)")
    
    def run(self):
        """Main application loop"""
        if not self.initialize():
            return
        
        while True:
            print("\n" + "=" * 50)
            print("MAIN MENU")
            print("=" * 50)
            print("1. View Book Information")
            print("2. View User's Ratings")
            print("3. Get Collaborative Filtering Recommendations")
            print("4. Get Content-Based Recommendations")
            print("5. Get Hybrid Recommendations")
            print("6. Get Cold Start Recommendations")
            print("7. View Sample Data Statistics")
            print("8. Exit")
            
            choice = input("\nEnter your choice (1-8): ").strip()
            
            if choice == '1':
                try:
                    book_id = int(input("Enter Book ID: "))
                    self.display_book_info(book_id)
                except ValueError:
                    print("Please enter a valid number.")
            
            elif choice == '2':
                try:
                    user_id = int(input("Enter User ID (1-50): "))
                    self.display_user_ratings(user_id)
                except ValueError:
                    print("Please enter a valid number.")
            
            elif choice == '3':
                try:
                    user_id = int(input("Enter User ID (1-50): "))
                    self.collaborative_recommendations(user_id)
                except ValueError:
                    print("Please enter a valid number.")
            
            elif choice == '4':
                try:
                    book_id = int(input("Enter Book ID: "))
                    self.display_book_info(book_id)
                    self.content_based_recommendations(book_id)
                except ValueError:
                    print("Please enter a valid number.")
            
            elif choice == '5':
                try:
                    user_id = int(input("Enter User ID (1-50): "))
                    self.display_user_ratings(user_id)
                    self.hybrid_recommendations(user_id)
                except ValueError:
                    print("Please enter a valid number.")
            
            elif choice == '6':
                self.cold_start_recommendations()
            
            elif choice == '7':
                self.display_statistics()
            
            elif choice == '8':
                print("Thank you for using the Book Recommendation System!")
                break
            
            else:
                print("Invalid choice. Please enter a number between 1 and 8.")
    
    def display_statistics(self):
        """Display data statistics"""
        print("\n DATA STATISTICS:")
        print(f"   Total Books: {len(self.data_loader.books_df)}")
        print(f"   Total Ratings: {len(self.data_loader.ratings_df)}")
        print(f"   Total Users: {self.data_loader.user_item_matrix.shape[0]}")
        
        # Average rating
        avg_rating = self.data_loader.ratings_df['rating'].mean()
        print(f"   Average Rating: {avg_rating:.2f}/5")
        
        # Most popular genres
        genre_counts = self.data_loader.books_df['genre'].value_counts()
        print(f"\n    Books by Genre:")
        for genre, count in genre_counts.head().items():
            print(f"      {genre}: {count} books")
        
        # Top rated books
        book_ratings = self.data_loader.ratings_df.groupby('book_id')['rating'].mean()
        top_books = book_ratings.sort_values(ascending=False).head(3)
        
        print(f"\n    Top Rated Books:")
        for book_id, rating in top_books.items():
            book_info = self.data_loader.get_book_info(book_id)
            if book_info:
                print(f"      {book_info['title']}: {rating:.2f}/5")

def main():
    """Main function"""
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    os.makedirs('src', exist_ok=True)
    
    # Check if data exists
    if not (os.path.exists('data/books.csv') and os.path.exists('data/ratings.csv')):
        print("Sample data not found. Generating sample data...")
        from sample_data_generator import generate_sample_data
        generate_sample_data()
    
    # Run the recommendation system
    system = BookRecommendationSystem()
    system.run()

if __name__ == "__main__":
    main()
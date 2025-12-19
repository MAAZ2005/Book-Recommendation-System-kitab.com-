import pandas as pd
import numpy as np
import random

def generate_sample_data():
    # Generate sample books data
    books_data = {
        'book_id': range(1, 101),
        'title': [f'Book Title {i}' for i in range(1, 101)],
        'author': [f'Author {random.choice(["A", "B", "C", "D", "E"])}' for _ in range(100)],
        'genre': [random.choice(['Fiction', 'Non-Fiction', 'Mystery', 'Sci-Fi', 'Romance', 'Biography', 'Self-Help']) for _ in range(100)],
        'year': [random.randint(1990, 2023) for _ in range(100)],
        'rating': [round(random.uniform(3.0, 5.0), 1) for _ in range(100)]
    }
    
    # Generate sample ratings data
    ratings_data = []
    user_ids = list(range(1, 51))
    
    for user_id in user_ids:
        # Each user rates 10-30 random books
        rated_books = random.sample(range(1, 101), random.randint(10, 30))
        for book_id in rated_books:
            rating = random.randint(1, 5)
            ratings_data.append([user_id, book_id, rating])
    
    # Create DataFrames
    books_df = pd.DataFrame(books_data)
    ratings_df = pd.DataFrame(ratings_data, columns=['user_id', 'book_id', 'rating'])
    
    # Save to CSV
    books_df.to_csv('data/books.csv', index=False)
    ratings_df.to_csv('data/ratings.csv', index=False)
    
    print(f"Generated {len(books_df)} books and {len(ratings_df)} ratings")
    print("Sample Books:")
    print(books_df.head())
    print("\nSample Ratings:")
    print(ratings_df.head())

if __name__ == "__main__":
    generate_sample_data()
import pandas as pd

books_df = pd.read_csv('data/books.csv')
ratings_df = pd.read_csv('data/ratings.csv')

print("📚 Books CSV Columns:")
print(list(books_df.columns))
print("\nFirst row:")
print(books_df.iloc[0].to_dict())

print("\n\n⭐ Ratings CSV Columns:")
print(list(ratings_df.columns))
print("\nFirst row:")
print(ratings_df.iloc[0].to_dict())
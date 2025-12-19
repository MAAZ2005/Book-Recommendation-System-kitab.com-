import streamlit as st
import pandas as pd
import numpy as np
import sys
import os


# Add your project modules to the path
sys.path.append('.')
try:
    from data_loader import DataLoader
    from collaborative_filtering import CollaborativeFiltering
    from content_based import ContentBasedFiltering
    from hybrid_recommender import HybridRecommender
except ImportError:
    # Try direct import
    from data_loader import DataLoader
    from collaborative_filtering import CollaborativeFiltering
    from content_based import ContentBasedFiltering
    from hybrid_recommender import HybridRecommender


# Page configuration
st.set_page_config(
    page_title="Kitab.com",
    page_icon="📕",
    layout="wide"
)



# Custom CSS
st.markdown("""
<style>


/* MAIN APP BACKGROUND */
.stApp {
    background-color: #f5f5f5;
}


/* HEADER */
.kitab-header {
    background-color: #b91c1c; /* red */
    color: white;
    padding: 20px;
    text-align: center;
    font-size: 2.8rem;
    font-weight: bold;
    border-radius: 8px;
}


/* SIDEBAR BACKGROUND */
section[data-testid="stSidebar"] {
    background-color: black;
}


/* SIDEBAR TEXT */
section[data-testid="stSidebar"] * {
    color: white !important;
}


/* ALL BUTTONS - Normal State (RED BACKGROUND, WHITE TEXT) */
div[data-testid="stButton"] button,
button[kind="primary"],
button,
.stButton > button {
    background-color: #b91c1c !important;
    color: white !important;
    border: 1px solid #b91c1c !important;
    border-radius: 5px !important;
    padding: 0.5rem 1rem !important;
    transition: all 0.3s ease !important;
    font-weight: 500 !important;
}


/* ALL BUTTONS - Hover State */
div[data-testid="stButton"] button:hover,
button[kind="primary"]:hover,
button:hover {
    background-color: #991a1a !important;
    color: white !important;
    border: 1px solid #991a1a !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 8px rgba(185, 28, 28, 0.3) !important;
}


/* ALL BUTTONS - Focus State */
div[data-testid="stButton"] button:focus,
button[kind="primary"]:focus,
button:focus {
    background-color: #b91c1c !important;
    color: white !important;
    border: 2px solid white !important;
    box-shadow: 0 0 0 3px rgba(185, 28, 28, 0.3) !important;
}


/* SIDEBAR BUTTONS - Override for consistency */
section[data-testid="stSidebar"] div[data-testid="stButton"] button,
section[data-testid="stSidebar"] button {
    background-color: #b91c1c !important;
    color: white !important;
    border: 1px solid #b91c1c !important;
}

section[data-testid="stSidebar"] div[data-testid="stButton"]:hover button,
section[data-testid="stSidebar"] button:hover {
    background-color: #991a1a !important;
    color: white !important;
    border: 1px solid #991a1a !important;
}


/* SIDEBAR RADIO BUTTONS */
section[data-testid="stSidebar"] div[role="radiogroup"] label {
    background-color: transparent !important;
    color: white !important;
    border: 1px solid white !important;
    padding: 0.5rem !important;
    border-radius: 5px !important;
    margin-bottom: 5px !important;
}


/* SIDEBAR RADIO BUTTONS - Selected */
section[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"] {
    background-color: white !important;
    color: black !important;
    border: 1px solid white !important;
}


/* SIDEBAR RADIO BUTTONS - Hover */
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background-color: #333 !important;
    color: white !important;
}


/* BOOK CARD */
.book-card {
    background: white;
    padding: 1.5rem;
    border-radius: 10px;
    border-left: 5px solid #b91c1c;
    box-shadow: 0 2px 6px rgba(0,0,0,0.15);
    margin-bottom: 1rem;
}


/* FOOTER */
.kitab-footer {
    background-color: black;
    color: white;
    text-align: center;
    padding: 12px;
    margin-top: 40px;
    border-radius: 6px;
    font-size: 0.9rem;
}


</style>
""", unsafe_allow_html=True)



# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False


def load_data():
    """Load data and initialize recommenders"""
    try:
        data_loader = DataLoader()
        if data_loader.load_data():
            data_loader.create_user_item_matrix()
            
            # Initialize recommenders
            cf = CollaborativeFiltering(data_loader.user_item_matrix)
            cbf = ContentBasedFiltering(data_loader.books_df)
            hybrid = HybridRecommender(cf, cbf, data_loader)
            
            st.session_state.data_loader = data_loader
            st.session_state.cf = cf
            st.session_state.cbf = cbf
            st.session_state.hybrid = hybrid
            st.session_state.data_loaded = True
            
            return True
        return False
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return False


def main():
    # Header
    st.markdown("""
    <div class="kitab-header">
        📕 Kitab.com
        <div style="font-size:1.2rem; margin-top:8px;">
            Welcome to Kitab.com
        </div>
    </div>
    """, unsafe_allow_html=True)


    st.markdown("---")


    # Sidebar
    with st.sidebar:
     

        menu = st.radio(
            "Choose section:",
            ["🏠 Dashboard", "📚 Browse Books", "🤝 Collaborative", 
             "📖 Content-Based", "🌟 Hybrid", "📊 Statistics"]
        )


        st.markdown("---")


        st.subheader("Data Management")
        if st.button("🔄 Reload Data", width='stretch'):
            st.session_state.data_loaded = False
            with st.spinner("Reloading..."):
                if load_data():
                    st.success("Data loaded!")
                else:
                    st.error("Failed to load data")


        if st.button("📊 Generate Sample Data", width='stretch'):
            try:
                from sample_data_generator import generate_sample_data
                generate_sample_data()
                st.success("Sample data generated!")
                st.session_state.data_loaded = False
                st.rerun()
            except Exception as e:
                st.error(f"Error: {str(e)}")


    # Load data
    if not st.session_state.data_loaded:
        with st.spinner("Loading data and initializing system..."):
            if not load_data():
                st.error("⚠️ Failed to load data. Click 'Generate Sample Data' above.")
                return


    data_loader = st.session_state.data_loader
    books_df = data_loader.books_df
    ratings_df = data_loader.ratings_df


    # Main content routing
    if menu == "🏠 Dashboard":
        show_dashboard(books_df, ratings_df)
    elif menu == "📚 Browse Books":
        browse_books(books_df)
    elif menu == "🤝 Collaborative":
        collaborative_recommendations()
    elif menu == "📖 Content-Based":
        content_based_recommendations(books_df)
    elif menu == "🌟 Hybrid":
        hybrid_recommendations()
    elif menu == "📊 Statistics":
        show_statistics(books_df, ratings_df)


    # ✅ FOOTER (ALWAYS VISIBLE)
    st.markdown("""
    <div class="kitab-footer">
        Created By:<br>
        1. Maaz Nizami<br>
        2. Abdul Rehman Zuberi
    </div>
    """, unsafe_allow_html=True)



def show_dashboard(books_df, ratings_df):
    """Display dashboard"""
    st.header("📊 Dashboard Overview")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📚 Total Books", len(books_df))
    with col2:
        st.metric("👥 Total Users", ratings_df['user_id'].nunique())
    with col3:
        st.metric("⭐ Total Ratings", len(ratings_df))
    with col4:
        avg_rating = books_df['rating'].mean() if 'rating' in books_df.columns else 0
        st.metric("📈 Avg Book Rating", f"{avg_rating:.2f}")
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Book Ratings Distribution")
        if 'rating' in books_df.columns:
            rating_counts = books_df['rating'].value_counts().sort_index()
            st.bar_chart(rating_counts)
    
    with col2:
        st.subheader("📚 Books by Genre")
        if 'genre' in books_df.columns:
            genre_counts = books_df['genre'].value_counts()
            st.dataframe(genre_counts, width='stretch')
    
    # Top books
    st.subheader("🏆 Top Rated Books")
    if 'rating' in books_df.columns and 'title' in books_df.columns and 'author' in books_df.columns:
        top_books = books_df.nlargest(10, 'rating')[['title', 'author', 'genre', 'rating']]
        for _, book in top_books.iterrows():
            with st.container():
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"**{book['title']}**")
                    st.caption(f"by {book['author']} | {book['genre']}")
                with col2:
                    st.markdown(f"⭐ **{book['rating']}**")
                st.divider()


def browse_books(books_df):
    """Browse books interface"""
    st.header("📚 Browse Book Collection")
    
    # Search and filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        search = st.text_input("🔍 Search books", placeholder="Title or author...")
    
    with col2:
        if 'genre' in books_df.columns:
            genres = ["All Genres"] + list(books_df['genre'].unique())
            genre_filter = st.selectbox("Filter by genre", genres)
        else:
            genre_filter = "All Genres"
    
    with col3:
        if 'rating' in books_df.columns:
            min_rating = st.slider("⭐ Minimum rating", 1.0, 5.0, 3.0, 0.5)
        else:
            min_rating = 0
    
    # Filter books
    filtered_books = books_df.copy()
    
    if search:
        if 'title' in books_df.columns:
            mask_title = filtered_books['title'].str.contains(search, case=False, na=False)
        else:
            mask_title = False
        
        if 'author' in books_df.columns:
            mask_author = filtered_books['author'].str.contains(search, case=False, na=False)
        else:
            mask_author = False
        
        filtered_books = filtered_books[mask_title | mask_author]
    
    if genre_filter != "All Genres" and 'genre' in books_df.columns:
        filtered_books = filtered_books[filtered_books['genre'] == genre_filter]
    
    if 'rating' in books_df.columns:
        filtered_books = filtered_books[filtered_books['rating'] >= min_rating]
    
    # Display results
    st.subheader(f"📖 Found {len(filtered_books)} books")
    
    if len(filtered_books) > 0:
        # Display 3 per row
        cols = st.columns(3)
        for idx, (_, book) in enumerate(filtered_books.head(30).iterrows()):  # Limit to 30
            with cols[idx % 3]:
                with st.container():
                    title = book.get('title', 'Unknown Title')
                    author = book.get('author', 'Unknown Author')
                    genre = book.get('genre', 'Unknown Genre')
                    year = book.get('year', 'Unknown')
                    rating = book.get('rating', 'N/A')
                    
                    st.markdown(f"""
                    <div class="book-card">
                        <h4>{title}</h4>
                        <p><strong>Author:</strong> {author}</p>
                        <p><strong>Genre:</strong> {genre}</p>
                        <p><strong>Year:</strong> {year}</p>
                        <p><strong>Rating:</strong> ⭐ {rating}/5</p>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.info("No books found matching your criteria.")


def collaborative_recommendations():
    """Collaborative filtering"""
    st.header("🤝 Collaborative Filtering Recommendations")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        user_id = st.number_input("👤 Enter User ID", min_value=1, max_value=50, value=1)
        
        algorithm = st.radio(
            "Select algorithm:",
            ["User-Based", "Item-Based", "Matrix Factorization"]
        )
        
        num_recs = st.slider("Number of recommendations", 3, 10, 5)
        
        if st.button("🎯 Get Recommendations", type="primary", width='stretch'):
            st.session_state.cf_user = user_id
            st.session_state.cf_algo = algorithm
            st.session_state.cf_num = num_recs
    
    with col2:
        if 'cf_user' in st.session_state:
            with st.spinner(f"Generating {st.session_state.cf_algo} recommendations..."):
                try:
                    cf = st.session_state.cf
                    user_id = st.session_state.cf_user
                    
                    if st.session_state.cf_algo == "User-Based":
                        indices, scores = cf.user_based_recommendations(user_id, st.session_state.cf_num)
                    elif st.session_state.cf_algo == "Item-Based":
                        indices, scores = cf.item_based_recommendations(user_id, st.session_state.cf_num)
                    else:
                        cf.matrix_factorization()
                        indices, scores = cf.mf_recommendations(user_id, st.session_state.cf_num)
                    
                    # Display
                    data_loader = st.session_state.data_loader
                    st.success(f"Top {len(indices)} recommendations for User {user_id}:")
                    
                    for i, (idx, score) in enumerate(zip(indices, scores), 1):
                        book_id = idx + 1
                        book_info = data_loader.get_book_info(book_id)
                        
                        if book_info:
                            col1, col2 = st.columns([4, 1])
                            with col1:
                                st.markdown(f"**{i}. {book_info.get('title', f'Book {book_id}')}**")
                                st.caption(f"by {book_info.get('author', 'Unknown')}")
                            with col2:
                                st.metric("Score", f"{score:.3f}")
                            st.divider()
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.info("👈 Enter user ID and click 'Get Recommendations'")


def content_based_recommendations(books_df):
    """Content-based recommendations"""
    st.header("📖 Content-Based Recommendations")
    
    tab1, tab2 = st.tabs(["🔍 Similar Books", "🎯 Your Preferences"])
    
    with tab1:
        st.subheader("Find books similar to:")
        if 'title' in books_df.columns and 'author' in books_df.columns:
            # Create book selection
            book_options = books_df.apply(
                lambda x: f"{x['title']} by {x['author']}", 
                axis=1
            ).tolist()
            
            selected_book = st.selectbox("Select a book", book_options)
            
            if selected_book:
                # Find book ID
                mask = books_df.apply(
                    lambda x: f"{x['title']} by {x['author']}", 
                    axis=1
                ) == selected_book
                
                if mask.any() and st.button("🔍 Find Similar Books", width='stretch'):
                    book_id = books_df[mask]['book_id'].values[0]
                    
                    with st.spinner("Finding similar books..."):
                        try:
                            cbf = st.session_state.cbf
                            indices, scores = cbf.get_similar_books(book_id, 5)
                            
                            st.success(f"Books similar to '{selected_book}':")
                            for idx, score in zip(indices, scores):
                                book_id = idx + 1
                                book_info = st.session_state.data_loader.get_book_info(book_id)
                                
                                if book_info:
                                    with st.expander(f"{book_info.get('title', f'Book {book_id}')} (Score: {score:.3f})"):
                                        st.write(f"**Author:** {book_info.get('author', 'Unknown')}")
                                        st.write(f"**Genre:** {book_info.get('genre', 'Unknown')}")
                                        st.write(f"**Year:** {book_info.get('year', 'Unknown')}")
                                        st.write(f"**Rating:** ⭐ {book_info.get('rating', 'N/A')}/5")
                            
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
    
    with tab2:
        st.subheader("Based on your preferences")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'genre' in books_df.columns:
                genres = st.multiselect("Favorite genres:", books_df['genre'].unique().tolist())
        
        with col2:
            if 'author' in books_df.columns:
                authors = st.multiselect("Favorite authors:", books_df['author'].unique().tolist()[:10])
        
        if st.button("🎯 Get Recommendations", width='stretch'):
            st.info("Preferences-based recommendations coming soon!")


def hybrid_recommendations():
    """Hybrid recommendations"""
    st.header("🌟 Hybrid Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        user_id = st.number_input("👤 User ID", min_value=1, max_value=50, value=1, key="hybrid_input")
        
        weight = st.slider(
            "⚖️ Algorithm Weight",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            help="0.0 = Only Content-Based, 1.0 = Only Collaborative",
            key="hybrid_weight"
        )
        
        if st.button("🚀 Generate Hybrid Recommendations", type="primary", width='stretch'):
            st.session_state.hybrid_user = user_id
            st.session_state.hybrid_alpha = weight
    
    with col2:
        if 'hybrid_user' in st.session_state:
            with st.spinner("Combining algorithms for optimal recommendations..."):
                try:
                    hybrid = st.session_state.hybrid
                    recommendations = hybrid.hybrid_recommendations(
                        st.session_state.hybrid_user,
                        n_recommendations=5,
                        alpha=st.session_state.hybrid_alpha
                    )
                    
                    st.success(f"Hybrid Recommendations for User {st.session_state.hybrid_user}:")
                    
                    for i, rec in enumerate(recommendations, 1):
                        with st.container():
                            st.markdown(f"""
                            <div class="book-card">
                                <h4>#{i}: {rec.get('title', 'Unknown Book')}</h4>
                                <p><strong>Author:</strong> {rec.get('author', 'Unknown')}</p>
                                <p><strong>Genre:</strong> {rec.get('genre', 'Unknown')}</p>
                                <p><strong>Recommendation Score:</strong> {rec.get('score', 'N/A')}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.info("👈 Enter user ID and click 'Generate Recommendations'")


def show_statistics(books_df, ratings_df):
    """Show statistics"""
    st.header("📊 System Statistics")
    
    # Basic stats
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📚 Books Information")
        stats_books = {
            "Total Books": len(books_df),
            "Unique Authors": books_df['author'].nunique() if 'author' in books_df.columns else 0,
            "Unique Genres": books_df['genre'].nunique() if 'genre' in books_df.columns else 0,
            "Average Rating": f"{books_df['rating'].mean():.2f}" if 'rating' in books_df.columns else "N/A",
            "Oldest Year": int(books_df['year'].min()) if 'year' in books_df.columns else "N/A",
            "Newest Year": int(books_df['year'].max()) if 'year' in books_df.columns else "N/A"
        }
        
        for key, value in stats_books.items():
            st.metric(key, value)
    
    with col2:
        st.subheader("⭐ Ratings Information")
        stats_ratings = {
            "Total Ratings": len(ratings_df),
            "Unique Users": ratings_df['user_id'].nunique() if 'user_id' in ratings_df.columns else 0,
            "Average Rating": f"{ratings_df['rating'].mean():.2f}" if 'rating' in ratings_df.columns else "N/A",
            "Min Rating": int(ratings_df['rating'].min()) if 'rating' in ratings_df.columns else "N/A",
            "Max Rating": int(ratings_df['rating'].max()) if 'rating' in ratings_df.columns else "N/A"
        }
        
        for key, value in stats_ratings.items():
            st.metric(key, value)
    
    st.markdown("---")
    
    # Data preview
    st.subheader("📋 Data Preview")
    
    tab1, tab2 = st.tabs(["Books Data", "Ratings Data"])
    
    with tab1:
        st.dataframe(books_df, width='stretch')
    
    with tab2:
        st.dataframe(ratings_df, width='stretch')


if __name__ == "__main__":
    main()

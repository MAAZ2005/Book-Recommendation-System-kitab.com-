#  Book Recommendation System (kitab.com)

A comprehensive **Book Recommendation System** that combines **Collaborative Filtering**, **Content-Based Filtering**, and a **Hybrid approach** to provide personalized book recommendations.

The project includes:

* A **command-line interface (CLI)** application
* An interactive **Streamlit web application (GUI)**
* Automatic **sample data generation** for quick setup



##  Features

### 🔹 Recommendation Techniques

* **Collaborative Filtering**

  * User-based
  * Item-based
  * Matrix Factorization (SVD)

* **Content-Based Filtering**

  * TF-IDF on book metadata (title, author, genre)
  * Numerical feature scaling (year, rating)

* **Hybrid Recommendation System**

  * Weighted combination of collaborative & content-based scores
  * Adjustable balance using `alpha`

* **Cold Start Handling**

  * Popularity-based recommendations for new users

---

##  Interfaces

###  Command-Line Application

Run the main CLI application:

python app.py
```

Features:

* View book details
* View user ratings
* Get collaborative, content-based, and hybrid recommendations
* View dataset statistics

---

###  Streamlit Web Application (GUI)

Launch the GUI:


streamlit run gui_app.py
```

Features:

* Dashboard with statistics
* Browse books with filters
* Interactive recommendation engine
* Data visualization

---

##  Project Structure

```
├── app.py                     # CLI application
├── gui_app.py                 # Streamlit web app
├── data_loader.py             # Data loading & preprocessing
├── collaborative_filtering.py # Collaborative filtering logic
├── content_based.py           # Content-based filtering logic
├── hybrid_recommender.py      # Hybrid recommendation engine
├── sample_data_generator.py   # Generates sample CSV data
├── check_columns.py           # Utility to inspect CSV structure
├── data/
│   ├── books.csv
│   └── ratings.csv
└── README.md
```

---

##  Dataset

If the dataset is missing, it will be generated automatically.

**Books (`books.csv`)**

* `book_id`
* `title`
* `author`
* `genre`
* `year`
* `rating`

**Ratings (`ratings.csv`)**

* `user_id`
* `book_id`
* `rating`

To manually generate sample data:


python sample_data_generator.py
```

---

##  Installation & Setup

### 1️ Clone the Repository


git clone [<repository-url>](https://github.com/MAAZ2005/Book-Recommendation-System-kitab.com-)
cd book-recommendation-system
```

### 2️ Create Virtual Environment (Optional but Recommended)


python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 3️ Install Dependencies


pip install -r requirements.txt
```

---

##  Sample Users

* User IDs range from **1–50**
* Book IDs range from **1–100**

---

##  Technologies Used

* Python
* Pandas & NumPy
* Scikit-learn
* SciPy
* Streamlit

---

##  Authors

* **Maaz Nizami**
* **Abdul Rehman Zuberi**

---

##  License

This project is for **educational purposes**. You are free to use and modify it.

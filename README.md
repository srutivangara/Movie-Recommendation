# 🎬 Movie Recommendation System

## Project Overview

The Movie Recommendation System is a content-based recommendation application built using Python, Pandas, Scikit-learn, and Streamlit.

It recommends movies based on similarities in their available cast and crew information. Users can select a movie from a dropdown menu and view the top five similar movies along with their similarity scores.

## Features

- Movie selection through a dropdown menu.
- Text preprocessing and cleaning.
- TF-IDF vectorization of movie features.
- Cosine similarity for comparing movies.
- Top-five movie recommendations.
- Interactive web interface built with Streamlit.

## Dataset

The project uses `tmdb_ds.csv`, which contains the following columns:

- `movie_id`: Unique movie identifier.
- `title`: Movie title.
- `cast`: Cast information.
- `crew`: Crew information.

The cast and crew names are extracted and combined with movie titles to create text features for recommendation.

**Note:** Recommendations are based on cast and crew information, not movie plot summaries.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Streamlit

## Project Structure

```text
GenAI_Assignment20_SrutiVangara/
├── app.py
├── tmdb_ds.csv
├── requirements.txt
└── README.md
```

## Installation and Setup

1. Install Python.
2. Download or clone the project repository.
3. Open the project folder in VS Code.
4. Open the terminal in the project folder.
5. Install the required libraries:

```bash
python -m pip install -r requirements.txt
```

6. Run the application:

```bash
python -m streamlit run app.py
```

7. Open the local URL displayed in the terminal, usually `http://localhost:8501`.

## How to Use

1. Launch the application.
2. Select a movie from the dropdown menu.
3. Click **Get Recommendations**.
4. View the recommended movies and their similarity scores.

## Recommendation Method

The system uses TF-IDF to convert cleaned movie-related text into numerical vectors. Cosine similarity compares the vectors and identifies movies with similar features.

## Future Improvements

- Include movie plot summaries and genres.
- Display movie posters and ratings.
- Add filters for genres and release years.
- Improve the recommendation results with additional movie metadata.

## Author

Sruti Vangara

## Project Purpose

This project demonstrates text preprocessing, feature extraction, similarity measurement, and deployment preparation for a content-based recommendation system.

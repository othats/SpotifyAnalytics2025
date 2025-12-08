# Spotify Analytics  2025

* **Marina Castellano Blanco** - *4th Year - Mathematical Engineering in Data Science*
* **Julia Othats-Dalès** - *4th Year - Mathematical Engineering in Data Science*

---
This repository contains the final project for the Visual Analytics course. Our goal was to decode the "science" behind music success by combining data visualization, machine learning, and web development.

The project is structured into three main components:

1. Visual Storytelling (Tableau): An in-depth analysis of global music trends, the evolution of genres, and the impact of streaming on song duration.

2. Music Recommendation Engine: A Machine Learning model (KNN) that recommends songs based on their mathematical "sonic DNA" (audio features like energy, danceability, and acousticness).

3. Interactive Web App (Streamlit): A user-friendly application where anyone can explore the data, visualize genre profiles, and get personalized song recommendations in real-time.

## Installation & Usage

Follow these steps to run the project locally on your machine.

**1. Clone the repository**
Download the project files to your local folder:
```bash
git clone [https://github.com/your-username/spotify-visual-analytics.git](https://github.com/your-username/spotify-visual-analytics.git)
cd spotify-visual-analytics
```

2. Create a Virtual Environment (Recommended) It is best practice to create an isolated environment to manage dependencies.

Windows:
```bash
python -m venv venv
.\venv\Scripts\activate
```

Mac/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install Dependencies We have provided a requirements.txt file with all necessary libraries (Streamlit, Pandas, Scikit-learn, etc.):
```bash
pip install -r requirements.txt
```
4. Run the Application Launch the main welcome page using Streamlit:
   
```bash
streamlit run spotify_app/01_Welcome_Page.py
```

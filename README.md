# Music Recommendation Dashboard

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Streamlit-powered interactive dashboard for music discovery, genre exploration, and track recommendations using Spotify audio features and UMAP-based clustering.

<img width="1106" alt="Screenshot 2025-06-05 at 6 45 01 PM" src="https://github.com/user-attachments/assets/ad277c4c-764d-438a-9ea9-17465c5971dc" />

## **Dashboard Navigation**  

The dashboard consists of five interactive tabs:  

### **Home**  
**Overview of the Project**  
Provides an overview of the project, highlighting key features and functionalities. Users can explore how the recommendation system works and navigate to different sections.  

### **Slideshow**  
**Music-Related Visuals**  
Displays a collection of music-related visuals, including album covers, genre trends, and interactive images that enhance the user experience.  

### **Similar Tracks**  
**Find Songs Similar to Your Favorites**  
Allows users to find songs similar to their favorites using **UMAP embeddings** and **cosine similarity**. The system generates recommendations based on audio feature similarity and provides direct **Spotify links** for listening.  
![Screenshot 2025-06-07 at 12 43 47 AM](https://github.com/user-attachments/assets/d11b3606-86e0-4ae0-bc1e-43b47476a906)
<img width="1250" alt="Screenshot 2025-06-05 at 6 44 23 PM" src="https://github.com/user-attachments/assets/1c334f7d-ffd9-4fd6-bb49-168cc18c70ad" />


### **Visualizations**  
**Interactive Music Data Analysis**  
Includes various interactive plots such as:  

- **3D UMAP Clustering** – Genre-based music landscapes
![Screenshot 2025-06-06 at 1 30 52 PM](https://github.com/user-attachments/assets/e272df4f-411a-4d16-ac9d-c9db48960abb)

- **Parallel Coordinates** – Multi-dimensional feature comparison
![Screenshot 2025-06-06 at 3 39 06 PM](https://github.com/user-attachments/assets/b7aa57a2-64ba-4f68-a15d-1ab16ff9ee83)

- **Violin Plots** – Audio feature distribution analysis
![Screenshot 2025-06-06 at 1 41 13 PM](https://github.com/user-attachments/assets/c8412b9e-64be-42c3-b5ac-a20e56c4e475)

## **For Back-End Geeks**  
![Screenshot 2025-06-06 at 1 48 33 PM](https://github.com/user-attachments/assets/c2e4bf3a-f008-42f6-9ce3-f8d537874b97)

## **Technical Overview**  

This section provides a **technical overview** of the recommendation system, focusing on **track metadata, similarity scoring, and clustering techniques**.  

Users can explore key attributes such as **track name, artist, genre, danceability, energy, valence, and acousticness**, which define a song's mood and characteristics.  

Using **cosine similarity**, the system finds tracks that closely match a selected song based on numerical audio features. Users can:  
- Select a song from the dataset  
- Compare its attributes with all available songs  
- Rank and filter recommendations based on similarity scores  

If clustering labels are available, recommendations can be refined further by:  
- Identifying the selected track's cluster  
- Suggesting songs within the same cluster for enhanced accuracy  

Interactive controls include:  
- A **recommendation slider** to adjust the number of suggested tracks  
- **Spotify links** for direct listening  
- A **data table display** presenting recommendations with key attributes and similarity scores  





### Technologies Used

- **Frontend**: Streamlit
- **Machine Learning**: UMAP, Scikit-Learn
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly, Matplotlib, Seaborn
- **API Integration**: Spotipy (Spotify Web API)

## Project Structure

```
Music_Recommendation/
├── Resource/                           # Dataset storage
│   └── SpotifyFeatures.csv            # Main Spotify dataset
├── Output/                             # Processed results
│   ├── umap_results.csv               # UMAP embeddings
│   └── recommended_songs_with_similarity_umap.csv
├── Images/                             # UI assets & banners
├── Slides/                             # Slideshow images
├── scripts/                            # Core logic modules
│   ├── data_utils.py                  # Data preprocessing & utilities
│   └── model_script.py                # UMAP & similarity computation
├── .streamlit/                         # Streamlit configuration
│   └── config.toml                    # App settings
├── requirements.txt                    # Python dependencies
├── dashboard.py                        # Main Streamlit application
├── README.md                          # Project documentation
└── LICENSE                            # Project license
```

## Quick Start
 **Run the application**
   ```bash
   streamlit run dashboard.py
   ```

2. **Open your browser** and navigate to `http://localhost:8501`

## Model Performance

| Model Architecture | Accuracy | Precision | Recall | Computational Cost |
|-------------------|----------|-----------|--------|-------------------|
| Cosine Similarity | ~72% | ~70% | ~85% | Low |
| UMAP Clustering | ~75.6% | ~72% | ~88% | Medium |
| RF + UMAP | ~75.8% | ~71.6% | ~90.5% | High |

## Deployment

### Deploy on Streamlit Cloud

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Deploy music recommendation dashboard"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Visit [Streamlit Community Cloud](https://streamlit.io/cloud)
   - Click "New App"
   - Connect your GitHub repository
   - Select `dashboard.py` as the main file
   - Click "Deploy"

### Alternative Deployment Options

- **Heroku**: Use `Procfile` for Heroku deployment
- **Docker**: Containerized deployment with provided Dockerfile
- **Local Server**: Run on local network for team access

## Dependencies

```txt
streamlit>=1.28.0
plotly>=5.15.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
umap-learn>=0.5.3
spotipy>=2.22.1
matplotlib>=3.7.0
seaborn>=0.12.0
```

## Configuration

### Spotify API Setup (Optional)
For enhanced features, set up Spotify API credentials:

1. Create a Spotify Developer account
2. Register your application
3. Add credentials to `.streamlit/secrets.toml`:
   ```toml
   [spotify]
   client_id = "your_client_id"
   client_secret = "your_client_secret"
   ```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### How to Contribute

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add docstrings to functions
- Include unit tests for new features
- Update documentation as needed

## Known Issues & Troubleshooting

- **UMAP similarity scores showing 1.0**: Under review for improved granularity
- **Fuzzy matching**: May affect recommendation accuracy in some cases
- **Memory usage**: Large datasets may require optimization

## Technical Documentation

### Architecture Overview
- **Data Layer**: Spotify Features CSV processing
- **Model Layer**: UMAP embeddings + cosine similarity
- **Application Layer**: Streamlit interactive dashboard
- **Visualization Layer**: Plotly/Matplotlib charts

### Key Algorithms
- **UMAP**: Uniform Manifold Approximation and Projection for dimensionality reduction
- **Cosine Similarity**: Vector similarity computation for recommendations
- **Random Forest**: Classification for genre prediction

## Contact & Support

**Gurpreet Singh Badrain**  
*Market Research Analyst & Aspiring Data Analyst*

- **GitHub**: [gbadrain](https://github.com/gbadrain)
- **LinkedIn**: [gurpreet-badrain](http://linkedin.com/in/gurpreet-badrain-b258a0219)
- **Email**: gbadrain@gmail.com

---
## **Support from Copilot AI**  

Throughout the development of this project, **Copilot AI** provided valuable assistance in:  

- **Debugging & Problem-Solving** – Identifying and resolving errors in code implementation.  
- **Optimizing Machine Learning Models** – Refining UMAP embeddings, cosine similarity computations, and clustering techniques.  
- **Enhancing Code Efficiency** – Suggesting improvements for data preprocessing, feature selection, and recommendation logic.  
- **Improving Documentation** – Structuring README content, formatting markdown, and ensuring clarity in technical explanations.  
- **Providing Deployment Guidance** – Offering best practices for deploying the Streamlit app on cloud platforms like Streamlit Community Cloud, Heroku, and Docker.  

Copilot AI played a crucial role in streamlining development, ensuring **efficient implementation**, and enhancing the **user experience** of the music recommendation system.  


## Show Your Support

If you found this project helpful, please consider giving it a star on GitHub!

[![GitHub stars](https://img.shields.io/github/stars/gbadrain/Music_Recommendation.svg?style=social&label=Star)](https://github.com/gbadrain/Music_Recommendation)

---

*"Without data, you're just another person with an opinion. But with hard work and the right analysis, data becomes the key to unlocking insights that drive real impact."*

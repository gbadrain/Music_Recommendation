# Music Recommendation Dashboard

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Streamlit-powered interactive dashboard for music discovery, genre exploration, and track recommendations using Spotify audio features and UMAP-based clustering.

<img width="1106" alt="Screenshot 2025-06-05 at 6 45 01 PM" src="https://github.com/user-attachments/assets/ad277c4c-764d-438a-9ea9-17465c5971dc" />

## Project Overview

This project leverages machine learning and Spotify's audio analysis to create an intelligent music recommendation system that:

- **Find Similar Tracks** - Discover music based on audio feature similarity
- **Explore Genre Trends** - Visualize music patterns across different genres  
- **Music Clustering** - Interactive UMAP visualizations of music landscapes
- **Dynamic Playlists** - Generate personalized recommendations with Spotify links
- **Audio Analysis** - Deep dive into track characteristics and features

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

## Dashboard Features

### Home Tab
- Welcome interface with project overview
- Music discovery insights and statistics
- Quick navigation to key features

### Slideshow Tab
- Full-screen image gallery
- Music-related visual content
- Interactive slideshow controls

### Similar Tracks Tab
- **Smart Search**: Find songs similar to your favorites
- **UMAP-Powered**: Uses advanced dimensionality reduction
- **Similarity Scoring**: Cosine similarity for accurate recommendations
- **Direct Links**: Spotify integration for instant listening

### Visualizations Tab
- **Violin Plots**: Audio feature distribution analysis
- **3D UMAP Clustering**: Interactive genre-based music landscapes
- **Parallel Coordinates**: Multi-dimensional feature comparison
- **Genre Analytics**: Deep dive into musical characteristics

### Backend Insights Tab
- Detailed track metadata and audio features
- Advanced similarity scoring metrics
- Technical implementation details
- Model performance analytics

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

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact & Support

**Gurpreet Singh Badrain**  
*Market Research Analyst & Aspiring Data Analyst*

- **GitHub**: [gbadrain](https://github.com/gbadrain)
- **LinkedIn**: [gurpreet-badrain](http://linkedin.com/in/gurpreet-badrain-b258a0219)
- **Email**: gbadrain@gmail.com

---

## Show Your Support

If you found this project helpful, please consider giving it a star on GitHub!

[![GitHub stars](https://img.shields.io/github/stars/gbadrain/Music_Recommendation.svg?style=social&label=Star)](https://github.com/gbadrain/Music_Recommendation)

---

*Built with care using Streamlit and Machine Learning*

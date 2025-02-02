# MusikAI

MusikAI is an AI-powered music recommendation system that detects a user's mood through facial expressions and suggests music accordingly.

## Features
- **Facial Emotion Detection**: Uses a neural network to classify emotions.
- **Music Recommendation**: Suggests songs based on detected mood.
- **User Preferences**: Customizable mood-based recommendations.

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/musikAI.git
   cd musikAI
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage
Run the main application script:
```sh
python src/musikai_app.py
```

## Project Structure
```
musikAI/
│-- src/                     # Source code
│   │-- facial_emotion.py     
│   │-- musikai_app.py        
│   │-- vggnet.py             
│-- config/                   # Config files
│   │-- mood_profiles.json     
│   │-- user_preferences.json  
│-- models/                   # Saved model weights
│   │-- vggnet.h5             
│   │-- vggnet_up.h5          
│-- logs/                     # Logs and caches
│   │-- musikai.log          
│   │-- .cache/               
│   │-- .spotifycache/        
│-- README.md                 # Documentation
│-- requirements.txt          # Dependencies
│-- .gitignore                # Ignore unnecessary files
```

## Contributing
1. Fork the repository.
2. Create a new branch for your feature: `git checkout -b feature-name`
3. Commit your changes: `git commit -m "Add new feature"`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request.

## License
This project is licensed under the MIT License.


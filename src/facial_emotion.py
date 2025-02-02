import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import argparse
import cv2
import mediapipe as mp
import tensorflow as tf
from vggnet import VGGNet
import numpy as np
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import time
from collections import deque
import logging
from datetime import datetime
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('musikai.log'),
        logging.StreamHandler()
    ]
)

class EmotionTracker:
    def __init__(self, history_size=10, stability_threshold=5, confidence_threshold=0.6):
        self.emotion_history = deque(maxlen=history_size)
        self.stability_threshold = stability_threshold
        self.confidence_threshold = confidence_threshold
        self.last_stable_emotion = None
        self.last_update_time = 0
        self.update_interval = 30  # seconds
        self.emotion_confidence = {}
        self.transition_states = {
            'Happy': {'Sad': 'Neutral', 'Angry': 'Neutral'},
            'Sad': {'Happy': 'Neutral', 'Angry': 'Neutral'},
            'Angry': {'Happy': 'Neutral', 'Sad': 'Neutral'},
            'Neutral': {}  # Can transition to any state
        }

    def update(self, emotion, confidence_scores):
        """Update emotion with confidence scoring and smooth transitions"""
        current_time = time.time()
        max_confidence = max(confidence_scores)
        
        if max_confidence >= self.confidence_threshold:
            self.emotion_history.append(emotion)
            self.emotion_confidence[emotion] = max_confidence
            
            if len(self.emotion_history) >= self.stability_threshold:
                recent_emotions = list(self.emotion_history)[-self.stability_threshold:]
                
                if all(e == recent_emotions[0] for e in recent_emotions):
                    stable_emotion = recent_emotions[0]
                    avg_confidence = self.emotion_confidence.get(stable_emotion, 0)
                    
                    if (stable_emotion != self.last_stable_emotion and 
                        current_time - self.last_update_time >= self.update_interval and
                        avg_confidence >= self.confidence_threshold):
                        
                        # Check if we need a transition state
                        if (self.last_stable_emotion and 
                            stable_emotion in self.transition_states.get(self.last_stable_emotion, {})):
                            transition_emotion = self.transition_states[self.last_stable_emotion][stable_emotion]
                            self.last_stable_emotion = stable_emotion
                            self.last_update_time = current_time
                            return True, transition_emotion, avg_confidence
                        
                        self.last_stable_emotion = stable_emotion
                        self.last_update_time = current_time
                        return True, stable_emotion, avg_confidence
        
        return False, None, 0

class PlaylistManager:
    def __init__(self):
        self.load_mood_profiles()
        self.playlist_cache = {}
        self.track_history = deque(maxlen=50)
        self.user_preferences = self.load_user_preferences()

    def load_mood_profiles(self):
        """Load mood profiles from configuration file"""
        try:
            with open('mood_profiles.json', 'r') as f:
                self.mood_profiles = json.load(f)
        except FileNotFoundError:
            # Default mood profiles if file doesn't exist
            self.mood_profiles = {
                'Happy': {
                    'seed_genres': ['pop', 'dance', 'happy'],
                    'audio_features': {
                        'target_valence': 0.8,
                        'target_energy': 0.8,
                        'target_tempo': 120,
                        'min_danceability': 0.6,
                    },
                    'limit': 10
                },
                'Sad': {
                    'seed_genres': ['sad', 'acoustic', 'piano'],
                    'audio_features': {
                        'target_valence': 0.2,
                        'target_energy': 0.3,
                        'target_tempo': 80,
                        'max_danceability': 0.4,
                    },
                    'limit': 10
                },
                'Angry': {
                    'seed_genres': ['rock', 'metal', 'intense'],
                    'audio_features': {
                        'target_valence': 0.4,
                        'target_energy': 0.9,
                        'target_tempo': 140,
                        'target_loudness': -5.0,
                    },
                    'limit': 10
                },
                'Neutral': {
                    'seed_genres': ['indie', 'ambient', 'chill'],
                    'audio_features': {
                        'target_valence': 0.5,
                        'target_energy': 0.5,
                        'target_tempo': 100,
                    },
                    'limit': 10
                }
            }
            # Save default profiles
            self.save_mood_profiles()

    def save_mood_profiles(self):
        """Save mood profiles to configuration file"""
        with open('mood_profiles.json', 'w') as f:
            json.dump(self.mood_profiles, f, indent=4)

    def load_user_preferences(self):
        """Load user preferences from file"""
        try:
            with open('user_preferences.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def save_user_preferences(self):
        """Save user preferences to file"""
        with open('user_preferences.json', 'w') as f:
            json.dump(self.user_preferences, f, indent=4)

    def get_recommendations(self, mood, spotify_client):
        """Get recommendations with user preference adjustments"""
        try:
            profile = self.mood_profiles[mood].copy()
            
            # Adjust profile based on user preferences
            if mood in self.user_preferences:
                for feature, adjustment in self.user_preferences[mood].items():
                    if feature in profile['audio_features']:
                        profile['audio_features'][feature] += adjustment

            # Get initial recommendations
            recommendations = spotify_client.recommendations(
                seed_genres=profile['seed_genres'],
                limit=profile['limit'] * 2,
                **profile['audio_features']
            )
            
            # Filter out recently played tracks
            filtered_tracks = [
                track for track in recommendations['tracks']
                if track['uri'] not in self.track_history
            ]
            
            # Score tracks based on audio features
            scored_tracks = []
            for track in filtered_tracks:
                audio_features = spotify_client.audio_features([track['uri']])[0]
                if audio_features:
                    match_score = self.calculate_feature_match(
                        audio_features, 
                        profile['audio_features']
                    )
                    scored_tracks.append((track, match_score))
            
            # Sort by match score and take top tracks
            best_tracks = sorted(
                scored_tracks, 
                key=lambda x: x[1], 
                reverse=True
            )[:profile['limit']]
            
            # Update track history
            for track, _ in best_tracks:
                self.track_history.append(track['uri'])
            
            return [track for track, _ in best_tracks]
            
        except Exception as e:
            logging.error(f"Error getting recommendations: {str(e)}")
            return None

    def calculate_feature_match(self, track_features, target_features):
        """Calculate how well a track matches the target features"""
        score = 0
        weights = {
            'valence': 1.0,
            'energy': 1.0,
            'tempo': 0.5,
            'danceability': 0.7,
            'instrumentalness': 0.3,
            'acousticness': 0.3,
            'loudness': 0.3
        }
        
        for feature, weight in weights.items():
            target = target_features.get(f'target_{feature}')
            if target and feature in track_features:
                difference = abs(target - track_features[feature])
                score += (1 - difference) * weight
                
        return score / sum(weights.values())

class SpotifyController:
    def __init__(self):
        self.client_id = "64d331ada7a8417686efa89da1901f9e"
        self.client_secret = "ea3b89ae35474be48ae4013fea11cba9"
        self.redirect_uri = "http://localhost:8888/callback"
        self.scope = ("playlist-modify-public playlist-modify-private "
                     "user-read-private user-modify-playback-state "
                     "user-read-playback-state streaming")
        
        self.connect_to_spotify()
        self.playlist_manager = PlaylistManager()
        self.current_mood = None
        self.current_playlist_id = None
        self.last_update_time = 0
        self.is_playing = False

    def connect_to_spotify(self):
        """Establish connection to Spotify API"""
        try:
            logging.info("Connecting to Spotify...")
            self.sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
                client_id=self.client_id,
                client_secret=self.client_secret,
                redirect_uri=self.redirect_uri,
                scope=self.scope,
                open_browser=True,
                cache_path=".spotifycache"
            ))
            
            user = self.sp.current_user()
            logging.info(f"Connected as: {user['display_name']}")
            
            # Check available devices
            devices = self.sp.devices()
            if devices['devices']:
                logging.info("\nAvailable Spotify devices:")
                for device in devices['devices']:
                    logging.info(f"- {device['name']} ({'Active' if device['is_active'] else 'Inactive'})")
            else:
                logging.warning("No Spotify devices found. Please open Spotify on your device.")
            
        except Exception as e:
            logging.error(f"Error connecting to Spotify: {str(e)}")
            raise

    def get_or_create_playlist(self, mood):
        """Get existing playlist or create new one for the given mood"""
        try:
            playlist_name = f"musikAI - {mood} Mood"
            user_id = self.sp.current_user()['id']
            
            # Check cache first
            if mood in self.playlist_manager.playlist_cache:
                return self.playlist_manager.playlist_cache[mood]
            
            # Search for existing playlist
            playlists = self.sp.current_user_playlists()
            for playlist in playlists['items']:
                if playlist['name'] == playlist_name:
                    self.playlist_manager.playlist_cache[mood] = playlist['id']
                    return playlist['id']
            
            # Create new playlist if not found
            logging.info(f"Creating new playlist: {playlist_name}")
            playlist = self.sp.user_playlist_create(user_id, playlist_name, public=False)
            self.playlist_manager.playlist_cache[mood] = playlist['id']
            return playlist['id']
            
        except Exception as e:
            logging.error(f"Error in get_or_create_playlist: {str(e)}")
            return None

    def force_playback(self, track_uris):
        """Force playback of specific tracks"""
        try:
            # Get available devices
            devices = self.sp.devices()
            if not devices['devices']:
                logging.warning("No available devices found. Please open Spotify on your device.")
                return False

            # Find active device
            device_id = None
            for device in devices['devices']:
                if device['is_active']:
                    device_id = device['id']
                    break
            
            if not device_id and devices['devices']:
                device_id = devices['devices'][0]['id']
                # Try to activate the device
                try:
                    self.sp.transfer_playback(device_id=device_id, force_play=True)
                    time.sleep(1)  # Wait for transfer
                except Exception as e:
                    logging.error(f"Error transferring playback: {str(e)}")

            if device_id:
                # Start playback
                try:
                    logging.info("Starting playback...")
                    self.sp.start_playback(device_id=device_id, uris=track_uris)
                    logging.info("Playback started successfully!")
                    return True
                except Exception as e:
                    logging.error(f"Error during playback: {str(e)}")
                    return False
            
            return False

        except Exception as e:
            logging.error(f"Error in force_playback: {str(e)}")
            return False

    def update_playlist(self, mood):
        """Update playlist based on detected mood and start playing"""
        try:
            if mood != self.current_mood:
                logging.info(f"\nUpdating playlist for mood: {mood}")
                self.current_mood = mood
                
                # Get recommendations using PlaylistManager
                tracks = self.playlist_manager.get_recommendations(mood, self.sp)
                
                if not tracks:
                    logging.warning("No recommendations found for this mood")
                    return False
                
                # Get track URIs and details
                track_uris = [track['uri'] for track in tracks]
                
                # Update or create playlist
                playlist_id = self.get_or_create_playlist(mood)
                if playlist_id:
                    self.sp.playlist_replace_items(playlist_id, track_uris)
                    self.current_playlist_id = playlist_id
                    
                    logging.info("\nUpdated playlist with tracks:")
                    for track in tracks[:5]:  # Show first 5 tracks
                        logging.info(f"- {track['name']} by {track['artists'][0]['name']}")
                    
                    # Try to force playback of the tracks directly
                    logging.info("\nAttempting to start playback...")
                    if self.force_playback(track_uris):
                        logging.info("Successfully started playback!")
                    else:
                        logging.warning("Could not start automatic playback. Please ensure Spotify is open and active.")
                    
                    return True
                
        except Exception as e:
            logging.error(f"Error updating playlist: {str(e)}")
            
        return False

def inference(image, face_detection, model_1, model_2, emotions, spotify_controller, emotion_tracker):
    """Process frame and update music based on detected emotion"""
    try:
        H, W, _ = image.shape
        
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_image)
        
        if results.detections:
            faces = []
            pos = []
            
            # Process all detected faces
            for detection in results.detections:
                box = detection.location_data.relative_bounding_box
                confidence = detection.score[0]

                x = int(box.xmin * W)
                y = int(box.ymin * H)
                w = int(box.width * W)
                h = int(box.height * H)

                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(x + w, W)
                y2 = min(y + h, H)

                if x1 < x2 and y1 < y2:  # Valid face region
                    face = image[y1:y2, x1:x2]
                    if face.size > 0:  # Check if face region is not empty
                        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                        faces.append(face)
                        pos.append((x1, y1, x2, y2, confidence))
    
            if faces:
                # Preprocess faces
                x = recognition_preprocessing(faces)

                # Get emotion predictions
                y_1 = model_1.predict(x, verbose=0)
                y_2 = model_2.predict(x, verbose=0)
                combined_predictions = y_1 + y_2
                
                # Process each face
                for i, (x1, y1, x2, y2, face_confidence) in enumerate(pos):
                    # Get emotion with highest probability
                    emotion_probs = combined_predictions[i]
                    emotion_idx = np.argmax(emotion_probs)
                    emotion_name = emotions[emotion_idx][0]
                    confidence_score = emotion_probs[emotion_idx]

                    # Map complex emotions to simplified ones
                    if emotion_name in ['Disgust', 'Fear', 'Surprise']:
                        emotion_name = 'Neutral'
                    
                    # Update emotion tracker
                    should_update, stable_emotion, avg_confidence = emotion_tracker.update(
                        emotion_name,
                        emotion_probs
                    )
                    
                    # Draw visualization
                    color = emotions[emotion_idx][1]
                    text_color = emotions[emotion_idx][2]
                    
                    # Draw face rectangle
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_AA)
                    
                    # Draw background for text
                    cv2.rectangle(image, (x1, y1-40), (x2, y1), color, -1, lineType=cv2.LINE_AA)
                    
                    # Show emotion and confidence
                    confidence_text = f"{int(confidence_score * 100)}%"
                    cv2.putText(image, f"{emotion_name} ({confidence_text})", 
                              (x1+5, y1-25),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2, 
                              lineType=cv2.LINE_AA)
                    
                    # Show stability indicator
                    stability_count = sum(1 for e in emotion_tracker.emotion_history 
                                       if e == emotion_name)
                    cv2.putText(image, f"Stability: {stability_count}/{emotion_tracker.stability_threshold}", 
                              (x1+5, y1-5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2, 
                              lineType=cv2.LINE_AA)
                    
                    # Update playlist if emotion is stable
                    if should_update and stable_emotion in ['Happy', 'Sad', 'Angry', 'Neutral']:
                        spotify_controller.update_playlist(stable_emotion)
        
        # Add UI overlay
        # Show current mood and playlist info
        if spotify_controller.current_mood:
            cv2.putText(image, f"Current Mood: {spotify_controller.current_mood}", 
                      (10, H-60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2,  
                      lineType=cv2.LINE_AA)
            
        return image
        
    except Exception as e:
        logging.error(f"Error in inference: {str(e)}")
        return image

def resize_face(face):
    """Resize face image for emotion detection"""
    x = tf.expand_dims(tf.convert_to_tensor(face), axis=2)
    return tf.image.resize(x, (48,48))

def recognition_preprocessing(faces):
    """Preprocess multiple faces for emotion recognition"""
    x = tf.convert_to_tensor([resize_face(f) for f in faces])
    return x

if __name__ == '__main__':
    try:
        # Set up argument parser
        ap = argparse.ArgumentParser()
        ap.add_argument("-v", "--video",
            help = "path to the (optional) video file")
        args = vars(ap.parse_args())

        # Initialize components
        logging.info("Initializing musikAI...")
        spotify_controller = SpotifyController()
        emotion_tracker = EmotionTracker(
            history_size=10,
            stability_threshold=5,
            confidence_threshold=0.6
        )

        # Initialize video capture
        video = args["video"] if args["video"] is not None else 0 
        cap = cv2.VideoCapture(video)
        
        if not cap.isOpened():
            raise Exception("Could not open video capture device")

        # Initialize face detection
        mp_face_detection = mp.solutions.face_detection
        face_detection = mp_face_detection.FaceDetection(
            model_selection=1,  # 0 for close faces, 1 for far faces
            min_detection_confidence=0.5
        )

        # Define emotion parameters
        emotions = {
            0: ['Angry', (0,0,255), (255,255,255)],
            1: ['Neutral', (160,160,160), (255,255,255)],
            2: ['Neutral', (160,160,160), (255,255,255)],
            3: ['Happy', (153,0,153), (255,255,255)],
            4: ['Sad', (255,0,0), (255,255,255)],
            5: ['Neutral', (160,160,160), (255,255,255)],
            6: ['Neutral', (160,160,160), (255,255,255)]
        }
        
        # Load emotion recognition models
        num_classes = len(emotions)
        input_shape = (48, 48, 1)
        weights_1 = 'saved_models/vggnet.h5'
        weights_2 = 'saved_models/vggnet_up.h5'
            
        logging.info("Loading emotion recognition models...")
        model_1 = VGGNet(input_shape, num_classes, weights_1)
        model_1.load_weights(model_1.checkpoint_path)

        model_2 = VGGNet(input_shape, num_classes, weights_2)
        model_2.load_weights(model_2.checkpoint_path)    

        logging.info("\nmusikAI is ready!")
        logging.info("Processing video feed...")
        print("Press 'q' to quit")
        
        while True:
            success, image = cap.read()
            if success:
                # Process frame
                result = inference(
                    image, 
                    face_detection,
                    model_1,
                    model_2,
                    emotions,
                    spotify_controller,
                    emotion_tracker
                )

                # Display result
                cv2.imshow('musikAI - Mood Detection', result)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
            
    except KeyboardInterrupt:
        logging.info("\nStopping musikAI...")
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        # Save user preferences
        if 'spotify_controller' in locals():
            spotify_controller.playlist_manager.save_user_preferences()
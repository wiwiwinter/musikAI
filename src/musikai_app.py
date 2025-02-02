import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import streamlit as st
import cv2
import mediapipe as mp
import tensorflow as tf
from vggnet import VGGNet
import numpy as np
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import time
from collections import deque
import traceback

# Page config
st.set_page_config(
    page_title="musikAI",
    page_icon="🎵",
    layout="wide"
)

# Custom CSS with enhanced styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton > button {
        background-color: #1DB954;
        color: white;
        border-radius: 20px;
        padding: 10px 20px;
        border: none;
        width: 100%;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #1ed760;
        transform: translateY(-2px);
    }
    .emotion-card {
        padding: 1rem;
        border-radius: 10px;
        background-color: #f8f9fa;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .playlist-card {
        padding: 1rem;
        border-radius: 10px;
        background-color: #f8f9fa;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .sidebar .stSlider > div > div > div {
        background-color: #1DB954;
    }
    </style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables with default values"""
    if 'detection_sensitivity' not in st.session_state:
        st.session_state.detection_sensitivity = 0.5
    if 'emotion_threshold' not in st.session_state:
        st.session_state.emotion_threshold = 5
    if 'volume' not in st.session_state:
        st.session_state.volume = 50
    if 'max_songs' not in st.session_state:
        st.session_state.max_songs = 10
    if 'start_camera' not in st.session_state:
        st.session_state.start_camera = False
    if 'current_emotion' not in st.session_state:
        st.session_state.current_emotion = None
    if 'current_tracks' not in st.session_state:
        st.session_state.current_tracks = None

def create_sidebar():
    """Create sidebar with adjustable settings"""
    with st.sidebar:
        st.title("Settings")
        
        # Detection Settings
        st.subheader("Detection Settings")
        st.session_state.detection_sensitivity = st.slider(
            "Detection Sensitivity",
            min_value=0.1,
            max_value=1.0,
            value=st.session_state.detection_sensitivity,
            help="Higher values mean more accurate but fewer detections"
        )
        
        st.session_state.emotion_threshold = st.slider(
            "Emotion Stability",
            min_value=1,
            max_value=10,
            value=st.session_state.emotion_threshold,
            help="Number of consistent readings needed before changing mood"
        )
        
        # Playlist Settings
        st.subheader("Playlist Settings")
        st.session_state.volume = st.slider(
            "Volume",
            min_value=0,
            max_value=100,
            value=st.session_state.volume,
            help="Spotify playback volume"
        )
        
        st.session_state.max_songs = st.slider(
            "Songs per Playlist",
            min_value=5,
            max_value=20,
            value=st.session_state.max_songs,
            help="Number of songs to include in each playlist"
        )
        
        # Reset button
        if st.button("Reset to Defaults"):
            st.session_state.detection_sensitivity = 0.5
            st.session_state.emotion_threshold = 5
            st.session_state.volume = 50
            st.session_state.max_songs = 10
            st.rerun()

class EmotionTracker:
    def __init__(self):
        self.emotion_history = deque(maxlen=st.session_state.emotion_threshold)
        self.last_stable_emotion = None
        self.last_update_time = 0
        self.update_interval = 30  # seconds

    def track_emotion(self, emotion, confidence):
        """Track emotion and determine if it's stable enough to update playlist"""
        current_time = time.time()
        
        if emotion and confidence > st.session_state.detection_sensitivity:
            self.emotion_history.append(emotion)
            
            if len(self.emotion_history) >= self.emotion_history.maxlen:
                recent_emotions = list(self.emotion_history)
                
                if all(e == recent_emotions[0] for e in recent_emotions):
                    stable_emotion = recent_emotions[0]
                    
                    if (stable_emotion != self.last_stable_emotion and 
                        current_time - self.last_update_time >= self.update_interval):
                        self.last_stable_emotion = stable_emotion
                        self.last_update_time = current_time
                        return True, stable_emotion
        
        return False, None

class SpotifyController:
    def __init__(self):
        self.client_id = "64d331ada7a8417686efa89da1901f9e"
        self.client_secret = "ea3b89ae35474be48ae4013fea11cba9"
        self.redirect_uri = "http://localhost:8888/callback"
        self.scope = ("playlist-modify-public playlist-modify-private "
                    "user-read-private user-modify-playback-state "
                    "user-read-playback-state streaming")
        
        try:
            self.sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
                client_id=self.client_id,
                client_secret=self.client_secret,
                redirect_uri=self.redirect_uri,
                scope=self.scope,
                open_browser=True
            ))
        except Exception as e:
            st.error(f"Failed to connect to Spotify: {str(e)}")
            st.error(traceback.format_exc())
            raise

        self.mood_seeds = {
            'Happy': {
                'seed_genres': ['pop', 'happy', 'dance'],
                'target_valence': 0.8,
                'target_energy': 0.8,
                'limit': st.session_state.max_songs
            },
            'Sad': {
                'seed_genres': ['sad', 'acoustic', 'piano'],
                'target_valence': 0.2,
                'target_energy': 0.3,
                'limit': st.session_state.max_songs
            },
            'Angry': {
                'seed_genres': ['rock', 'metal', 'intense'],
                'target_valence': 0.4,
                'target_energy': 0.9,
                'limit': st.session_state.max_songs
            },
            'Neutral': {
                'seed_genres': ['indie', 'ambient', 'chill'],
                'target_valence': 0.5,
                'target_energy': 0.5,
                'limit': st.session_state.max_songs
            }
        }
        self.current_mood = None
        self.current_device_id = self.get_active_device()

    def get_active_device(self):
        """Get the ID of the active Spotify device"""
        try:
            devices = self.sp.devices()
            if not devices['devices']:
                st.warning("No active Spotify devices found. Please open Spotify on your device.")
                return None
            
            # First try to find an active device
            active_devices = [d for d in devices['devices'] if d['is_active']]
            if active_devices:
                return active_devices[0]['id']
            
            # If no active device, use the first available one
            return devices['devices'][0]['id']
            
        except Exception as e:
            st.warning("Could not get Spotify devices. Please ensure Spotify is open.")
            return None

    def ensure_device_ready(self):
        """Ensure there's an active device and refresh if needed"""
        if not self.current_device_id:
            self.current_device_id = self.get_active_device()
        return self.current_device_id is not None

    def start_playback(self, track_uris):
        """Start playback with error handling and retry logic"""
        if not self.ensure_device_ready():
            return False

        try:
            # Try to set volume first
            self.sp.volume(st.session_state.volume, device_id=self.current_device_id)
            
            # Start playback
            self.sp.start_playback(
                device_id=self.current_device_id,
                uris=track_uris
            )
            return True
            
        except Exception as e:
            if "Player command failed: No active device found" in str(e):
                # Retry once with refreshed device ID
                self.current_device_id = self.get_active_device()
                if self.current_device_id:
                    try:
                        self.sp.start_playback(
                            device_id=self.current_device_id,
                            uris=track_uris
                        )
                        return True
                    except:
                        pass
            
            st.warning("Could not start playback. Please ensure Spotify is open and active.")
            return False

    def get_or_create_playlist(self, mood):
        try:
            playlist_name = f"musikAI - {mood} Mood"
            user_id = self.sp.current_user()['id']
            
            playlists = self.sp.current_user_playlists()
            for playlist in playlists['items']:
                if playlist['name'] == playlist_name:
                    return playlist['id']
            
            playlist = self.sp.user_playlist_create(user_id, playlist_name, public=False)
            return playlist['id']
        except Exception as e:
            st.error(f"Error in playlist creation: {str(e)}")
            return None

    def update_playlist(self, mood):
        """Update playlist based on detected mood and start playback"""
        try:
            if mood != st.session_state.get('current_mood'):
                st.session_state['current_mood'] = mood

                # Update limit based on current settings
                for mood_config in self.mood_seeds.values():
                    mood_config['limit'] = st.session_state.max_songs
                
                # Get recommendations
                recommendations = self.sp.recommendations(
                    seed_genres=self.mood_seeds[mood]['seed_genres'],
                    target_valence=self.mood_seeds[mood]['target_valence'],
                    target_energy=self.mood_seeds[mood]['target_energy'],
                    limit=self.mood_seeds[mood]['limit']
                )

                if not recommendations['tracks']:
                    st.warning("No tracks found for the current mood.")
                    return None
                
                # Get track URIs and create/update playlist
                track_uris = [track['uri'] for track in recommendations['tracks']]
                playlist_id = self.get_or_create_playlist(mood)
                
                if playlist_id:
                    # Update playlist contents
                    self.sp.playlist_replace_items(playlist_id, track_uris)
                    
                    # Attempt to start playback
                    playback_started = self.start_playback(track_uris)
                    
                    if playback_started:
                        st.success(f"Now playing {mood} mood playlist!")
                    
                    # Update session state with tracks regardless of playback status
                    st.session_state.current_tracks = recommendations['tracks']
                    return recommendations['tracks']
                
        except Exception as e:
            st.error(f"Error updating playlist: {str(e)}")
            st.error(traceback.format_exc())
            return None

def resize_face(face):
    x = tf.expand_dims(tf.convert_to_tensor(face), axis=2)
    return tf.image.resize(x, (48,48))

def recognition_preprocessing(faces):
    x = tf.convert_to_tensor([resize_face(f) for f in faces])
    return x

def inference(image, face_detection, model_1, model_2, emotions):
    """Process frame and detect emotions"""
    H, W, _ = image.shape
    
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_image)
    
    detected_emotion = None
    max_confidence = 0

    if results.detections:
        faces = []
        pos = []
        for detection in results.detections:
            box = detection.location_data.relative_bounding_box
            
            x = int(box.xmin * W)
            y = int(box.ymin * H)
            w = int(box.width * W)
            h = int(box.height * H)
            
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(x + w, W)
            y2 = min(y + h, H)
            
            face = image[y1:y2,x1:x2]
            if face.size > 0:  # Check if face region is valid
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                faces.append(face)
                pos.append((x1, y1, x2, y2))
    
        if faces:  # Process if faces were detected
            x = recognition_preprocessing(faces)
            
            y_1 = model_1.predict(x, verbose=0)
            y_2 = model_2.predict(x, verbose=0)
            predictions = y_1 + y_2
            
            for i, pred in enumerate(predictions):
                emotion_idx = np.argmax(pred)
                confidence = float(pred[emotion_idx] / 2)  # Average confidence
                
                emotion_name = emotions[emotion_idx][0]
                
                if confidence > max_confidence:
                    detected_emotion = emotion_name
                    max_confidence = confidence
                
                # Draw rectangle and emotion label
                cv2.rectangle(image, (pos[i][0],pos[i][1]),
                            (pos[i][2],pos[i][3]), emotions[emotion_idx][1], 2, lineType=cv2.LINE_AA)
                
                # Background for text
                cv2.rectangle(image, (pos[i][0],pos[i][1]-20),
                            (pos[i][2]+20,pos[i][1]), emotions[emotion_idx][1], -1, lineType=cv2.LINE_AA)
                
                # Emotion text
                cv2.putText(image, f'{emotion_name} ({confidence:.0%})', 
                        (pos[i][0],pos[i][1]-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, emotions[emotion_idx][2], 2, lineType=cv2.LINE_AA)
    
    return image, detected_emotion, max_confidence

def run_camera():
    """Run the camera feed with emotion detection"""
    # Updated emotions dictionary to only include the four emotions
    emotions = {
        0: ['Angry', (0,0,255), (255,255,255)],   # Red for Angry
        1: ['Happy', (153,0,153), (255,255,255)], # Purple for Happy
        2: ['Neutral', (160,160,160), (255,255,255)], # Gray for Neutral
        3: ['Sad', (255,0,0), (255,255,255)]      # Blue for Sad
    }
    
    num_classes = len(emotions)  # Now 4 instead of 7
    input_shape = (48, 48, 1)
    
    # Load models
    model_1 = VGGNet(input_shape, num_classes, 'saved_models/vggnet.h5')
    model_2 = VGGNet(input_shape, num_classes, 'saved_models/vggnet_up.h5')
    
    model_1.load_weights(model_1.checkpoint_path)
    model_2.load_weights(model_2.checkpoint_path)
    
    # Initialize face detection
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(
        min_detection_confidence=st.session_state.detection_sensitivity
    )
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("Could not open camera.")
        return
    
    # Create placeholders using columns for better layout
    col1, col2 = st.columns([2, 1])
    with col1:
        frame_placeholder = st.empty()
        info_placeholder = st.empty()
    
    with col2:
        playlist_placeholder = st.empty()
        emotion_placeholder = st.empty()
    
    # Initialize emotion tracker and Spotify controller
    emotion_tracker = EmotionTracker()
    spotify = SpotifyController()
    
    try:
        while cap.isOpened() and st.session_state.start_camera:
            ret, frame = cap.read()
            if ret:
                processed_frame, emotion, confidence = inference(
                    frame,
                    face_detection,
                    model_1,
                    model_2,
                    emotions
                )
                
                if emotion:
                    st.session_state.current_emotion = emotion
                    info_placeholder.info(f"Current Mood: {emotion} ({confidence:.2%})")
                    
                    should_update, stable_emotion = emotion_tracker.track_emotion(
                        emotion,
                        confidence
                    )
                    
                    if should_update:
                        spotify = SpotifyController()
                        new_tracks = spotify.update_playlist(stable_emotion)
                        if new_tracks:
                            st.session_state.current_tracks = new_tracks
                
                frame_placeholder.image(processed_frame, channels="BGR")
                
                with playlist_placeholder.container():
                    st.subheader("Current Playlist")
                    if st.session_state.current_tracks:
                        for track in st.session_state.current_tracks:
                            st.write(f"🎵 {track['name']} - {track['artists'][0]['name']}")
                    else:
                        st.write("No playlist generated yet")
                
                with emotion_placeholder.container():
                    if st.session_state.current_emotion:
                        st.subheader("Current Emotion")
                        st.info(f"Detected Mood: {st.session_state.current_emotion}")
            
            time.sleep(0.1)
            
    except Exception as e:
        st.error(f"Camera Error: {str(e)}")
        st.error(traceback.format_exc())
    
    finally:
        cap.release()
        face_detection.close()

def main():
    # Initialize session state and create sidebar
    initialize_session_state()
    create_sidebar()
    
    st.title("musikAI 🎵")
    st.markdown("Mood-Based Music Player using Facial Expression Recognition")
    
    # Main layout with columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Camera controls
        if st.button("Start Camera" if not st.session_state.start_camera else "Stop Camera"):
            st.session_state.start_camera = not st.session_state.start_camera
            if st.session_state.start_camera:
                run_camera()  # Function to run the camera feed
    
    with col2:
        # Manual controls for mood selection
        st.subheader("Manual Control")
        selected_mood = st.selectbox("Select Mood", ["Happy", "Sad", "Angry", "Neutral"])

        if st.button("Update Playlist"):
            try:
                spotify = SpotifyController()  # Ensure SpotifyController is reinitialized
                with st.spinner("Updating playlist..."):
                    tracks = spotify.update_playlist(selected_mood)
                    if tracks:
                        st.session_state.current_tracks = tracks  # Update session state with new tracks
                        st.success(f"Updated playlist for {selected_mood} mood!")
                        st.rerun()  # Refresh UI to display the playlist
            except Exception as e:
                st.error(f"Spotify Error: {str(e)}")
        
        # Display the current playlist if tracks are available
        if st.session_state.current_tracks:  # <--- Add this code block here
            st.subheader("Current Playlist")
            for track in st.session_state.current_tracks:
                st.write(f"🎵 {track['name']} - {track['artists'][0]['name']}")
        
        # Display current detected emotion if available
        if st.session_state.current_emotion:
            st.subheader("Current Emotion")
            st.info(f"Detected Mood: {st.session_state.current_emotion}")

if __name__ == "__main__":
    main()
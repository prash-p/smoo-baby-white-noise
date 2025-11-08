import math
import pyaudio
import numpy as np
import time
import threading
import wave
import platform
import csv
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from scipy import signal  # For FFT analysis
import resampy

# System-specific volume control import
if platform.system() == 'Windows':
    from ctypes import cast, POINTER
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    from tensorflow.lite.python.interpreter import Interpreter
elif platform.system() == 'Linux':
    import alsaaudio
    from tflite_runtime.interpreter import Interpreter

class SystemVolumeController:
    """Cross-platform system volume control."""
    
    def __init__(self):
        self.system = platform.system()
        if self.system == 'Windows':
            devices = AudioUtilities.GetSpeakers()
            try:
                interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
                self.volume = cast(interface, POINTER(IAudioEndpointVolume))
            except Exception as e:
                self.volume = devices.EndpointVolume.QueryInterface(IAudioEndpointVolume)
        elif self.system == 'Linux':
            try:
                self.mixer = alsaaudio.Mixer("PCM")
            except alsaaudio.ALSAAudioError:
                print("Warning: Could not initialize ALSA audio mixer")
                self.mixer = None
        
    def set_volume(self, volume_percent: float):
        """Set system volume (0-100)."""
        try:
            if self.system == 'Windows':
                # Convert to dB range (-65.25 to 0.0)
                db = 34 * math.log((volume_percent) / 100, 10)
                self.volume.SetMasterVolumeLevel(db, None)
            elif self.system == 'Linux' and self.mixer:
                self.mixer.setvolume(int(volume_percent))

        except Exception as e:
            print(f"Warning: Could not set system volume: {e}")

# --- All user settings are here! ---
CONFIG: Dict[str, Any] = {
    "white_noise_file": "sounds/snoo - level 0.wav",  # Path to the white noise WAV file
    "noise_duration_seconds": 5,        # How long the generated noise file should be
    "noise_volume": 1.0,               # Playback volume (0.0 to 1.0)            # Legacy threshold (not used with FFT)
    "sample_rate": 44100,              # Audio sample rate
    "playback_chunk_size": 2048,    # Small chunks for smooth playback
    "monitor_chunk_size": 44100,     # Large chunks for accurate YAMNet detection
    "fft_min_freq": 200,              # Minimum frequency for cry detection (Hz)
    "fft_max_freq": 1000,             # Maximum frequency for cry detection (Hz)
    "power_threshold": 0.1,           # Power threshold (normalized power per Hz)
    "level_down_time": 60,            # Time in seconds before decreasing level
    "level_up_time": 5,               # Minimum time before increasing level again
    "current_level": 0,               # Starting level
    "volume_levels": {                # System volume percentage for each level
        0: 95,
        1: 97,
        2: 98,
        3: 99,
        4: 100
    }
}

class WhiteNoisePlayer:
    """Plays a WAV audio file on a continuous loop in a background thread."""

    def __init__(self, audio_file_path: str, volume: float, pa_instance: pyaudio.PyAudio):
        self.audio_file_path = Path(audio_file_path)
        self.volume = max(0.0, min(1.0, volume))
        self.pyaudio_instance = pa_instance

        self._thread: Optional[threading.Thread] = None
        self._playing = threading.Event()
        self._stream: Optional[pyaudio.Stream] = None
        self._frames: bytes = b''
        self._wf: Optional[wave.Wave_read] = None

        self._load_file()

    def _load_file(self):
        """Loads the entire WAV file into memory for efficient playback, trims to zero crossings for seamless looping."""
        if not self.audio_file_path.exists():
            raise FileNotFoundError(f"Audio file not found at {self.audio_file_path}")

        self._wf = wave.open(str(self.audio_file_path), 'rb')
        all_frames = self._wf.readframes(self._wf.getnframes())
        audio_data = np.frombuffer(all_frames, dtype=np.int16)
        # trim the first and last 2 seconds of the recording
        trim_samples = 2 * self._wf.getframerate()  # 2 seconds worth of samples
        audio_data = audio_data[trim_samples:-trim_samples]

        # Find zero crossing at start
        start = 0
        for i in range(1, len(audio_data)):
            if audio_data[i-1] < 0 and audio_data[i] >= 0:
                start = i
                break
        # Find zero crossing at end
        end = len(audio_data)
        for i in range(len(audio_data)-1, 0, -1):
            if audio_data[i-1] < 0 and audio_data[i] >= 0:
                end = i
                break
        trimmed = audio_data[start:end]
        adjusted_data = (trimmed * self.volume).astype(np.int16)
        self._frames = adjusted_data.tobytes()
        print(f"🎵 Audio file '{self.audio_file_path}' loaded and trimmed for seamless playback.")


    def _play_loop(self):
        """Internal method to handle the audio playback stream and loop."""
        if not self._wf:
            return

        self._stream = self.pyaudio_instance.open(
            format=self.pyaudio_instance.get_format_from_width(self._wf.getsampwidth()),
            channels=self._wf.getnchannels(),
            rate=self._wf.getframerate(),
            output=True,
            frames_per_buffer=CONFIG["playback_chunk_size"]
        )

        print(f"▶️ Starting white noise playback (Volume: {self.volume:.1f})")
        
        frame_chunks = [
            self._frames[i:i + CONFIG["playback_chunk_size"]]  
            for i in range(0, len(self._frames), CONFIG["playback_chunk_size"])
        ]

        while not self._playing.is_set():
            for chunk in frame_chunks:
                if self._playing.is_set() or not self._stream.is_active():
                    break
                self._stream.write(chunk)

        # Cleanup stream
        self._stream.stop_stream()
        self._stream.close()
        print("🛑 White noise playback stopped.")

    def start(self):
        """Starts the playback in a separate thread."""
        # Stop any existing thread first
        if self._thread and self._thread.is_alive():
            self.stop()
        
        self._playing.clear()
        self._thread = threading.Thread(target=self._play_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stops the playback thread."""
        self._playing.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self._wf:
            self._wf.close()

class MicrophoneMonitor:
    """Monitors the microphone for baby crying using FFT analysis."""

    def __init__(self, threshold: float, pa_instance: pyaudio.PyAudio):
        self.power_threshold = threshold
        self.pyaudio_instance = pa_instance
        self.current_level = CONFIG["current_level"]
        self.last_cry_time = time.time()
        self.last_level_change_time = time.time()
        self._thread: Optional[threading.Thread] = None
        self._monitoring = threading.Event()
        self._stream: Optional[pyaudio.Stream] = None
        self.level_changed_callback: Optional[Callable[[int], None]] = None
        self.volume_controller = SystemVolumeController()
        self.volume_controller.set_volume(CONFIG["volume_levels"][self.current_level])

            # ADD THESE LINES - Load YAMNet model once
        self._load_yamnet_model()

    def _load_yamnet_model(self):
        """Load YAMNet model and class names once during initialization."""
        TFLITE_MODEL = "yamnet.tflite"
        CLASS_MAP = "yamnet_class_map.csv"
        
        # Load class names
        self.class_names = []
        with open(CLASS_MAP, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                self.class_names.append(row['display_name'])
        
        # Load and allocate model
        self.interpreter = Interpreter(model_path=TFLITE_MODEL)
        self.interpreter.allocate_tensors()
        self.inp_details = self.interpreter.get_input_details()
        self.out_details = self.interpreter.get_output_details()
        
        print("🤖 YAMNet model loaded successfully")

    def _analyze_sound(self, audio_data: np.ndarray, mode: str = 'yamnet') -> bool:
        # Normalize audio data to [-1, 1] range
        normalized_data = audio_data.astype(float) / 32768.0
        
        if mode == 'fft':
            # Apply Hanning window to reduce spectral leakage
            windowed = normalized_data * signal.windows.hann(len(normalized_data))
            
            # Compute FFT
            fft = np.fft.rfft(windowed)
            # Get frequency bins
            freqs = np.fft.rfftfreq(len(normalized_data), 1/CONFIG["sample_rate"])
            
            # Get power spectrum (units: power spectral density)
            power = np.abs(fft)**2 / len(normalized_data)

            # Find average power in cry frequency range (units: power per Hz)
            mask = (freqs >= CONFIG["fft_min_freq"]) & (freqs <= CONFIG["fft_max_freq"])
            cry_freq_span = CONFIG["fft_max_freq"] - CONFIG["fft_min_freq"]
            cry_power = np.sum(power[mask]) / cry_freq_span
            is_crying = cry_power > self.power_threshold
            
            return is_crying
        
        elif mode == 'yamnet':
            
            # Use pre-loaded model (remove all the loading code above)
            waveform = normalized_data.astype(np.float32)
            waveform = resampy.resample(waveform, sr_orig=44100, sr_new=16000)
            
            if len(waveform) < 15600:
                waveform = np.pad(waveform, (0, 15600 - len(waveform)), mode='wrap')
            elif len(waveform) > 15600:
                waveform = waveform[-15600:]
            
            # Use self.interpreter instead of creating new one
            self.interpreter.set_tensor(self.inp_details[0]['index'], waveform)
            self.interpreter.invoke()
            scores = self.interpreter.get_tensor(self.out_details[0]['index'])[0]
            top_idx = int(np.argmax(scores))
            top_score = float(scores[top_idx])
            label = self.class_names[top_idx]  # Use self.class_names
            print("Label: ",label," and score: ", top_score)
            is_crying = True if 'cry' in label.lower() and top_score >= 0.2 else False
            
            return is_crying
        else:
            raise ValueError("Invalid mode for sound analysis.")

    def _update_level(self, is_crying: bool):
        current_time = time.time()
        level_changed = False
        
        if is_crying and self.current_level < 4:
            # Check if enough time has passed since last level increase
            if current_time - self.last_level_change_time >= CONFIG["level_up_time"]:
                self.current_level += 1
                self.last_cry_time = current_time
                self.last_level_change_time = current_time
                level_changed = True
            
        elif not is_crying and self.current_level > 0:
            # Check if it's been quiet for level_down_time seconds
            if current_time - self.last_cry_time >= CONFIG["level_down_time"]:
                self.current_level -= 1
                self.last_cry_time = current_time
                self.last_level_change_time = current_time
                level_changed = True
        
        if level_changed:
            # Update system volume for the new level
            new_volume = CONFIG["volume_levels"][self.current_level]
            self.volume_controller.set_volume(new_volume)
            print(f"🔈 System volume set to {new_volume}%")
            
            if self.level_changed_callback:
                self.level_changed_callback(self.current_level)
        
        return level_changed

    def _monitor_loop(self):
        """Internal method to read from the mic and analyze frequencies."""
        try:
            self._stream = self.pyaudio_instance.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=CONFIG["sample_rate"],
                input=True,
                frames_per_buffer=CONFIG["monitor_chunk_size"]
            )
            print(f"🎤 Microphone monitoring started (Power threshold: {self.power_threshold})")

            while not self._monitoring.is_set():
                data = self._stream.read(CONFIG["monitor_chunk_size"], exception_on_overflow=False)  # NEW
                audio_data = np.frombuffer(data, dtype=np.int16)
                
                # Analyze the frequency content
                # is_crying = self._analyze_sound(audio_data,mode='yamnet') if platform.system() == 'Linux' else self._analyze_sound(audio_data,mode='fft')
                is_crying = self._analyze_sound(audio_data, mode='yamnet')

                self._update_level(is_crying)
                
                # Optional: Print continuous power feedback
                print(f"{'😭' if is_crying else '🔇'}, Level: {self.current_level}")
                
                # time.sleep(0.05)  # Small delay to be CPU-friendly
        
        except Exception as e:
            print(f"❌ Monitoring error: {e}")
        finally:
            if self._stream:
                self._stream.stop_stream()
                self._stream.close()
            print("🛑 Microphone monitoring stopped.")

    def start(self):
        """Starts microphone monitoring in a separate thread."""
        # Stop any existing thread first
        if self._thread and self._thread.is_alive():
            self.stop()
        
        self._monitoring.clear()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stops the monitoring thread."""
        self._monitoring.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self.current_level = 0
        self.volume_controller.set_volume(CONFIG["volume_levels"][self.current_level])


class AudioController:
    """Manages the player and monitor, handling startup and shutdown."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pyaudio_instance = pyaudio.PyAudio()
        self.current_level = config["current_level"]
        
        # Preload all WAV files into memory
        self.preloaded_players = {}
        print("🔄 Preloading all sound files...")
        for level in range(5):
            audio_file = f"sounds/snoo - level {level}.wav"
            try:
                self.preloaded_players[level] = WhiteNoisePlayer(
                    audio_file_path=audio_file,
                    volume=config["noise_volume"],
                    pa_instance=self.pyaudio_instance
                )
                print(f"  ✅ Level {level} loaded")
            except Exception as e:
                print(f"  ❌ Level {level} failed: {e}")
        
        # Set current player
        self.player = self.preloaded_players[self.current_level]
        
        try:
            self.monitor = MicrophoneMonitor(
                threshold=config["power_threshold"],
                pa_instance=self.pyaudio_instance
            )
            self.monitor.level_changed_callback = self._handle_level_change
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            self.pyaudio_instance.terminate()
            raise


    def _handle_level_change(self, new_level: int):
        """Handle level change requests from the monitor."""
        print(f"📊 Changing sound level from {self.current_level} to {new_level}")
        
        # Stop current player
        self.player.stop()
        
        # Switch to preloaded player for new level
        self.current_level = self.monitor.current_level = new_level
        self.monitor.volume_controller.set_volume(CONFIG["volume_levels"][self.current_level])
        self.player = self.preloaded_players[new_level]

        
        # Start new player
        self.player.start()

    def run(self):
        """Starts all audio processes and waits for user interruption."""
        print("\n🚀 Starting audio controller... Press Ctrl+C to stop.")
        try:
            self.player.start()
            self.monitor.start()
            # Keep the main thread alive to listen for keyboard interrupt
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n⏹️  User interrupt detected. Shutting down...")
        finally:
            self.cleanup()  # Use cleanup instead of stop
    
    def stop(self):
        """Stops audio playback and monitoring without destroying resources."""
        print("Pausing audio services...")
        self.monitor.stop()
        self.player.stop()
        print("✅ Audio controller paused (can be restarted).")

    def cleanup(self):
        """Fully cleanup and terminate all resources (call on app shutdown)."""
        print("Gracefully stopping all audio services...")
        self.stop()
        self.pyaudio_instance.terminate()
        print("✅ Audio controller cleaned up successfully.")


def generate_white_noise_file(filename: str, duration: int, sample_rate: int):
    """Generates a mono, 16-bit WAV file containing white noise if it doesn't exist."""
    if Path(filename).exists():
        print(f"👍 Found existing noise file: '{filename}'")
        return

    print(f"⌛ Generating {duration}s white noise file: '{filename}'...")
    num_samples = int(sample_rate * duration)
    # Generate random samples in the range of a 16-bit integer
    samples = np.random.randint(-32767, 32767, num_samples, dtype=np.int16)

    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)       # Mono
        wf.setsampwidth(2)       # 16-bit audio
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())
    print("✅ Noise file generated.")


if __name__ == "__main__":
    try:
        controller = AudioController(config=CONFIG)
        controller.run()
    except Exception as e:
        print(f"A critical error occurred: {e}")

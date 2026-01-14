# ================================
# MediaPipe Audio Classification Integration
# Enhanced Audio Classification using YAMNet Model
# ================================

import numpy as np
import json
import sys
import base64
import tempfile
import os
from scipy.io import wavfile
import traceback

# Try importing MediaPipe, but don't auto-install (fragile in production)
try:
    from mediapipe.tasks import python
    from mediapipe.tasks.python.components import containers
    from mediapipe.tasks.python import audio
except ImportError:
    print("❌ MediaPipe not found. Please install it using: pip install mediapipe", file=sys.stderr)
    sys.exit(1)

def log(message):
    """Log to stderr to avoid polluting stdout"""
    try:
        print(message, file=sys.stderr)
    except:
        pass

def get_yamnet_model_path():
    """Get local YAMNet model path"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'yamnet.tflite')
    
    if not os.path.exists(model_path):
        log(f"❌ YAMNet model not found at: {model_path}")
        raise FileNotFoundError(f"YAMNet model not found at {model_path}")
    
    log(f"✅ Using local YAMNet model: {model_path}")
    return model_path

def calculate_audio_metrics(wav_data_float, sample_rate, start_seconds, duration_seconds):
    """
    Calculate REAL physical metrics from the audio signal using FFT and RMS.
    """
    try:
        start_idx = int(start_seconds * sample_rate)
        end_idx = int((start_seconds + duration_seconds) * sample_rate)
        
        # Ensure indices are within bounds
        start_idx = max(0, start_idx)
        end_idx = min(len(wav_data_float), end_idx)
        
        # Extract segment
        segment = wav_data_float[start_idx:end_idx]
        
        if len(segment) == 0:
            return 440.0, -60.0
            
        # 1. Calculate RMS (Amplitude/Decibels)
        rms = np.sqrt(np.mean(segment**2))
        decibels = 20 * np.log10(rms + 1e-9)  # Avoid log(0)
        
        # 2. Calculate Dominant Frequency using FFT
        # Apply Hanning window
        windowed = segment * np.hanning(len(segment))
        spectrum = np.fft.rfft(windowed)
        frequencies = np.fft.rfftfreq(len(segment), d=1.0/sample_rate)
        
        # Find peak frequency
        magnitudes = np.abs(spectrum)
        peak_idx = np.argmax(magnitudes)
        dominant_freq = frequencies[peak_idx]
        
        return float(dominant_freq), float(decibels)
        
    except Exception as e:
        log(f"Error calculating metrics chunk: {e}")
        return 440.0, -60.0

def classify_audio_with_mediapipe(audio_data_base64, filename="uploaded_audio"):
    """
    Classify audio using MediaPipe YAMNet model
    """
    temp_path = None
    try:
        log(f"🎵 Starting MediaPipe Audio Classification: {filename}")
        
        # Decode base64 audio data
        try:
            audio_bytes = base64.b64decode(audio_data_base64)
        except Exception as e:
            return json.dumps({"error": f"Failed to decode base64 audio: {str(e)}", "analysisComplete": False})

        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            temp_file.write(audio_bytes)
            temp_path = temp_file.name
        
        # Get local model path
        model_path = get_yamnet_model_path()
        
        # Read audio file for manual signal processing
        try:
            sample_rate, wav_data = wavfile.read(temp_path)
        except Exception as e:
             return json.dumps({"error": f"Invalid WAV file format: {str(e)}", "analysisComplete": False})

        log(f"📊 Audio info: {sample_rate} Hz, {len(wav_data)} samples")
        
        # Normalize audio data
        if wav_data.dtype == np.int16:
            wav_data_float = wav_data.astype(float) / np.iinfo(np.int16).max
        elif wav_data.dtype == np.int32:
            wav_data_float = wav_data.astype(float) / np.iinfo(np.int32).max
        elif wav_data.dtype == np.uint8:
            wav_data_float = (wav_data.astype(float) - 128) / 128
        else:
            wav_data_float = wav_data.astype(float)
        
        # Handle stereo to mono
        if len(wav_data_float.shape) > 1:
            wav_data_float = np.mean(wav_data_float, axis=1)
        
        # MediaPipe Setup
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = audio.AudioClassifierOptions(
            base_options=base_options, 
            max_results=10,
            score_threshold=0.1
        )
        
        # Run Classification
        with audio.AudioClassifier.create_from_options(options) as classifier:
            audio_clip = containers.AudioData.create_from_array(wav_data_float, sample_rate)
            classification_result_list = classifier.classify(audio_clip)
            
            log(f"🔍 Found {len(classification_result_list)} audio segments")
            
            enhanced_classifications = []
            all_categories = {}
            enhanced_sound_events = []
            
            # YAMNet typically uses ~0.975s windows
            segment_duration = 0.975 
            
            for idx, classification_result in enumerate(classification_result_list):
                timestamp_ms = idx * 975 
                timestamp_s = timestamp_ms / 1000.0
                
                # --- REAL SIGNAL ANALYSIS ---
                freq_hz, db_level = calculate_audio_metrics(
                    wav_data_float, sample_rate, timestamp_s, segment_duration
                )
                
                segment_categories = []
                
                if classification_result.classifications:
                    top_category = classification_result.classifications[0].categories[0]
                    
                    forensic_category = map_to_forensic_category(top_category.category_name)
                    
                    enhanced_sound_events.append({
                        "time": round(timestamp_s, 2),
                        "duration": segment_duration,
                        "type": forensic_category,
                        "mediapipe_category": top_category.category_name,
                        "confidence": round(top_category.score, 4),
                        # Map -60dB -> 0.0, 0dB -> 1.0 amplitude approx
                        "amplitude": round(max(0, (db_level + 60) / 60), 2),
                        "frequency": round(freq_hz, 1),
                        "decibels": round(db_level, 1),
                        "classification_source": "YAMNet+FFT"
                    })

                    for category in classification_result.classifications[0].categories:
                        name = category.category_name
                        score = category.score
                        segment_categories.append({
                            "category": name,
                            "confidence": round(score, 4)
                        })
                        
                        if name not in all_categories:
                            all_categories[name] = {"total": 0, "count": 0, "max": 0}
                        all_categories[name]["total"] += score
                        all_categories[name]["count"] += 1
                        all_categories[name]["max"] = max(all_categories[name]["max"], score)
                
                enhanced_classifications.append({
                    "segment": idx,
                    "timestamp": round(timestamp_s, 2),
                    "real_frequency": round(freq_hz, 1),
                    "real_decibels": round(db_level, 1),
                    "classifications": segment_categories
                })

            # Overall Statistics
            overall_stats = []
            for name, stats in all_categories.items():
                overall_stats.append({
                    "category": name,
                    "average_confidence": round(stats["total"] / stats["count"], 4),
                    "max_confidence": round(stats["max"], 4),
                    "occurrence_count": stats["count"],
                    "coverage_percentage": round((stats["count"] / len(classification_result_list)) * 100, 1)
                })
            overall_stats.sort(key=lambda x: x["average_confidence"], reverse=True)
            
            # File-wide RMS
            rms_total = np.sqrt(np.mean(wav_data_float**2))
            
            # Dominant freq of entire file (simple peak)
            whole_spectrum = np.fft.rfft(wav_data_float[0:min(len(wav_data_float), sample_rate*10)]) # Limit to first 10s for speed
            whole_freqs = np.fft.rfftfreq(len(wav_data_float[0:min(len(wav_data_float), sample_rate*10)]), 1/sample_rate)
            dom_freq_total = whole_freqs[np.argmax(np.abs(whole_spectrum))]

            analysis_results = {
                "filename": filename,
                "duration": round(len(wav_data_float) / sample_rate, 2),
                "sampleRate": int(sample_rate),
                "segments_analyzed": len(classification_result_list),
                
                "mediapipe_classifications": {
                    "overall_statistics": overall_stats[:15],
                    "segment_classifications": enhanced_classifications
                },
                
                "enhanced_sound_events": enhanced_sound_events,
                "detectedSounds": len(enhanced_sound_events),
                
                "soundEvents": enhanced_sound_events[:50], 
                "dominantFrequency": round(float(dom_freq_total), 1),
                "maxDecibels": max([e["decibels"] for e in enhanced_sound_events]) if enhanced_sound_events else -60,
                "averageRMS": round(float(rms_total), 6),
                
                "analysisComplete": True,
                "analysisType": "mediapipe_v2_real_physics",
                "timestamp": "2024-01-01T00:00:00Z"
            }
            
            return json.dumps(analysis_results, indent=2)

    except Exception as e:
        log(f"❌ Analysis Error: {str(e)}")
        traceback.print_exc(file=sys.stderr)
        return json.dumps({
            "error": str(e), 
            "analysisComplete": False, 
            "message": "Internal processing error"
        })
        
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except:
                pass

def map_to_forensic_category(mediapipe_category):
    """Map MediaPipe categories to forensic investigation categories"""
    cat = mediapipe_category.lower()
    mapping = {
        "speech": "Human Voice", "narration": "Monologue", "conversation": "Conversation",
        "music": "Musical Content", "drum": "Percussion", "guitar": "String Inst.",
        "vehicle": "Vehicle", "car": "Automobile", "truck": "Heavy Vehicle", "siren": "Emergency Siren",
        "dog": "Canine", "cat": "Feline", "bird": "Avian",
        "gunshot": "Gunshot/Explosion", "explosion": "Gunshot/Explosion",
        "glass": "Breaking Glass", "scream": "Scream/Distress",
        "wind": "Wind Noise", "rain": "Precipitation", "water": "Water Sound", 
        "footsteps": "Footsteps"
    }
    
    for key, val in mapping.items():
        if key in cat:
            return val
    return mediapipe_category

if __name__ == "__main__":
    try:
        # Check if stdin has data
        if not sys.stdin.isatty():
            input_content = sys.stdin.read()
            if not input_content:
                log("❌ Empty input on stdin")
                print(json.dumps({"error": "Empty input"}))
                sys.exit(1)
                
            try:
                request = json.loads(input_content)
                audio_b64 = request.get("audioData")
                fname = request.get("filename", "unknown")
                
                if not audio_b64:
                    print(json.dumps({"error": "No audioData provided"}))
                    sys.exit(1)
                    
                result = classify_audio_with_mediapipe(audio_b64, fname)
                print(result) # STDOUT only receives the JSON
                sys.exit(0)
            except json.JSONDecodeError:
                log("❌ Invalid JSON on stdin")
                print(json.dumps({"error": "Invalid JSON"}))
                sys.exit(1)
        else:
            if len(sys.argv) > 1:
                log("⚠️ Warning: Legacy CLI mode.")
                result = classify_audio_with_mediapipe(sys.argv[1], sys.argv[2] if len(sys.argv)>2 else "test.wav")
                print(result)
            else:
                log("Usage: echo 'JSON' | python script.py")
    except Exception as e:
        log(f"❌ Fatal: {e}")
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

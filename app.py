import matplotlib.pyplot as plt
import librosa.display
from multiprocessing import Pool
from audio_processing import load_audio, detect_global_key, detect_tempo, detect_time_signature
from chord_analysis import analyze_segment

def main():
    audio_path = 'songs/running.mp3'
    y, sr = load_audio(audio_path)
    print('loaded', y, sr)
    librosa.display.waveshow(y, sr=sr)
    plt.title("Forma de onda del audio original")
    plt.show()
    window_size, hop_size = 5, 2
    window_samples, hop_samples = window_size * sr, hop_size * sr
    segments = range(0, len(y) - window_samples, hop_samples)
    with Pool(8) as p:
        results = p.starmap(analyze_segment, [(start, y, sr, window_samples) for start in segments])
    segment_tonality = [result[0] for result in results]
    segment_chords = [result[1] for result in results]
    print("Tonalidades detectadas en los segmentos:", segment_tonality)
    print(f"Tonalidad global de la canción: {detect_global_key(y, sr)}")
    tempo, _ = detect_tempo(y, sr)
    print(f"Tempo (BPM) de la canción: {tempo:.2f}")
    print(f"Compás de la canción: {detect_time_signature(y, sr)}")

if __name__ == '__main__':
    main()

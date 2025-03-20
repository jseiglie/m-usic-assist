from music21 import chord, pitch
import librosa
import numpy as np

chord_mapping = {
    'incomplete minor-seventh chord': 'Cm',
    'minor-seventh chord': 'Cm7',
    'major triad': 'C',
    'perfect-fourth minor tetrachord': 'F',
    'Kumoi pentachord': 'Gm',
    'dominant-eleventh': 'G7',
    'major-ninth chord': 'Cmaj9',
    'perfect-fourth major tetrachord': 'Cmaj7',
    'whole-tone trichord': 'F#',
    'phrygian trichord': 'Dm',
    'unison': 'C',
    'center-cluster pentachord': 'Gm7',
    'perfect-fourth tetramirror': 'D',
    'chromatic trimirror': 'E',
    'D all combinatorial (P6, I1, RI7)': 'D7',
}

def correct_note_name(note_name):
    return note_name.replace('♯', '#').replace('♭', 'b')

def simplify_chords(chord_name):
    return chord_mapping.get(chord_name, chord_name)

def analyze_segment(start, y, sr, window_samples):
    try:
        # Extraer el segmento de audio
        y_segment = y[start:start + window_samples]
        
        # Detectar la tonalidad en el segmento
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y_segment, fmin=librosa.note_to_hz('A0'), fmax=librosa.note_to_hz('C8')
        )
        
        # Eliminar valores nan y calcular la media de los tonos detectados
        f0_valid = f0[~np.isnan(f0)]  # Filtrar los valores nan
        if len(f0_valid) > 0:
            f0_mean = np.mean(f0_valid)
        else:
            f0_mean = 440.0  # Asignamos un valor predeterminado (A4, 440 Hz)
        
        # Convertir la frecuencia a una nota musical
        note = pitch.Pitch(frequency=f0_mean)
        note_name = note.nameWithOctave

        # Obtener el cromagrama (representación de las notas musicales)
        chroma = librosa.feature.chroma_cens(y=y_segment, sr=sr)
        
        # Filtrar las notas más prominentes (usando el valor máximo en cada "frame")
        chord_pitches = np.argmax(chroma, axis=0)
        
        notes = []
        for chord_index in chord_pitches:
            try:
                # Convertir el índice a una frecuencia de nota
                hz = librosa.midi_to_hz(chord_index + 24)  # Ajustamos para que los índices se mapeen a las notas correctas
                note_name = librosa.hz_to_note(hz)
                
                # Corregir el nombre de la nota para que music21 lo entienda
                corrected_note_name = correct_note_name(note_name)
                
                # Crear la nota con music21
                pitch_obj = pitch.Pitch(corrected_note_name)
                
                # Verificar que la nota esté en un rango musical estándar
                if pitch_obj.midi > 0 and pitch_obj.midi < 127:  # Rango de notas MIDI
                    if len(notes) == 0 or pitch_obj != notes[-1]:
                        notes.append(pitch_obj)
            except ValueError as ve:
                print(f"Error al convertir nota: {ve}")
                continue
            except Exception as e:
                print(f"Error desconocido: {e}")
                continue

        # Crear un acorde usando music21 y filtrar acordes no estándar
        if len(notes) > 0:
            chord_object = chord.Chord(notes)
            chord_name = chord_object.commonName
            
            # Simplificar el nombre del acorde
            chord_name = simplify_chords(chord_name)
        else:
            chord_name = "No chord detected"

        return note_name, chord_name, chord_object
    except librosa.util.exceptions.ParameterError as e:
        print(f"Error de parámetro en librosa: {e}")
        return "Unknown", "No chord detected", None
    except Exception as e:
        print(f"Error inesperado: {e}")
        return "Unknown", "No chord detected", None

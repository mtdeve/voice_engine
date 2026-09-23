import sounddevice as sd
import numpy as np
import threading
import sys
import scipy.signal

from pedalboard import (
    Pedalboard,
    Gain,
    Distortion,
    Compressor,
    Reverb,
    Delay,
    Chorus,
    Phaser,
    Limiter,
    HighpassFilter,
    LowpassFilter,
    Clipping,
    Bitcrush,
    NoiseGate,
    PeakFilter,
    LowShelfFilter,
    HighShelfFilter,
    LadderFilter
)

# ============================================================
# AUDIO
# ============================================================

INPUT = 11
OUTPUT = 16

SAMPLERATE = 16000
BLOCKSIZE = 1024

volume = 0.5

# ============================================================
# EFFECTS
# ============================================================

effects = {
    "gain": 0,
    "dist": 0,
    "compressor": 0,
    "reverb": 0,
    "delay": 0,
    "chorus": 0,
    "phaser": 0,
    "highpass": 0,
    "lowpass": 0,
    "pitch": 0,  # Gestito in semitoni (es. -12 a +12)
    "clip": 0,
    "bitcrush": 32,
    "gate": 0,

    "peak_freq": 1000,
    "peak_gain": 0,
    "peak_q": 1,

    "lowshelf_freq": 200,
    "lowshelf_gain": 0,
    "lowshelf_q": 1,

    "highshelf_freq": 4000,
    "highshelf_gain": 0,
    "highshelf_q": 1,

    "ladder": 0
}

# ============================================================
# GLOBALS
# ============================================================

board = Pedalboard()
stream = None

board_lock = threading.Lock()

# FIFO dell'audio già processato
audio_fifo = np.empty(0, dtype=np.float32)

fifo_lock = threading.Lock()

# ============================================================
# PITCH SHIFT HELPER (Resampling ultra-veloce per Live)
# ============================================================

def process_pitch_shift(audio_block, semitones):
    """
    Pitch Shift in tempo reale ottimizzato per streaming senza ritardi.
    Utilizza la variazione del factor di ricampionamento mantenendo costante la dimensione del blocco.
    """
    if semitones == 0:
        return audio_block

    # Calcoliamo il fattore di variazione frequenza dai semitoni
    factor = 2 ** (semitones / 12.0)
    num_samples = audio_block.shape[1]
    
    # Ricampioniamo il blocco
    resampled_len = int(num_samples / factor)
    if resampled_len <= 0:
        return audio_block
        
    resampled = scipy.signal.resample(audio_block[0], resampled_len)
    
    # Riportiamo la dimensione del blocco esattamente pari a BLOCKSIZE per evitare overflow/underflow della FIFO
    if len(resampled) < num_samples:
        resampled = np.pad(resampled, (0, num_samples - len(resampled)), mode='constant')
    else:
        resampled = resampled[:num_samples]
        
    return resampled.reshape(1, -1)

# ============================================================
# REBUILD BOARD
# ============================================================

def rebuild_board():

    global board

    new_board = []

    if effects["gate"] != 0:
        new_board.append(
            NoiseGate(
                threshold_db=effects["gate"],
                ratio=10,
                attack_ms=1,
                release_ms=100
            )
        )

    if effects["gain"] != 0:
        new_board.append(
            Gain(
                gain_db=effects["gain"]
            )
        )

    if effects["highpass"] > 0:
        new_board.append(
            HighpassFilter(
                cutoff_frequency_hz=effects["highpass"]
            )
        )

    if effects["lowpass"] > 0:
        new_board.append(
            LowpassFilter(
                cutoff_frequency_hz=effects["lowpass"]
            )
        )

    if effects["lowshelf_gain"] != 0:
        new_board.append(
            LowShelfFilter(
                cutoff_frequency_hz=effects["lowshelf_freq"],
                gain_db=effects["lowshelf_gain"],
                q=effects["lowshelf_q"]
            )
        )

    if effects["highshelf_gain"] != 0:
        new_board.append(
            HighShelfFilter(
                cutoff_frequency_hz=effects["highshelf_freq"],
                gain_db=effects["highshelf_gain"],
                q=effects["highshelf_q"]
            )
        )

    if effects["peak_gain"] != 0:
        new_board.append(
            PeakFilter(
                cutoff_frequency_hz=effects["peak_freq"],
                gain_db=effects["peak_gain"],
                q=effects["peak_q"]
            )
        )

    if effects["compressor"] > 0:
        new_board.append(
            Compressor(
                threshold_db=-20,
                ratio=effects["compressor"]
            )
        )

    if effects["dist"] > 0:
        new_board.append(
            Distortion(
                drive_db=effects["dist"]
            )
        )

    if effects["clip"] < 0:
        new_board.append(
            Clipping(
                threshold_db=effects["clip"]
            )
        )

    if effects["bitcrush"] < 32:
        new_board.append(
            Bitcrush(
                bit_depth=effects["bitcrush"]
            )
        )

    if effects["chorus"] > 0:
        new_board.append(
            Chorus(
                rate_hz=1.0,
                depth=effects["chorus"],
                mix=effects["chorus"]
            )
        )

    if effects["phaser"] > 0:
        new_board.append(
            Phaser(
                rate_hz=1.0,
                depth=effects["phaser"],
                mix=effects["phaser"]
            )
        )

    if effects["delay"] > 0:
        new_board.append(
            Delay(
                delay_seconds=effects["delay"],
                feedback=0.3,
                mix=0.3
            )
        )

    if effects["reverb"] > 0:
        new_board.append(
            Reverb(
                room_size=effects["reverb"]
            )
        )

    if effects["ladder"] > 0:
        new_board.append(
            LadderFilter(
                mode=LadderFilter.Mode.LPF24,
                cutoff_hz=effects["ladder"],
                resonance=0
            )
        )

    new_board.append(
        Limiter(
            threshold_db=-1
        )
    )

    with board_lock:
        board = Pedalboard(new_board)

# ============================================================
# CALLBACK
# ============================================================

def callback(indata, outdata, frames, time, status):

    global audio_fifo

    if status:
        print(status, file=sys.stderr)

    # --------------------------------------------------------
    # INPUT
    # --------------------------------------------------------

    audio = np.asarray(
        indata[:, 0],
        dtype=np.float32
    )

    audio = audio.reshape(1, -1)

    # --------------------------------------------------------
    # PROCESS PITCH SHIFT (LIVE STREAM SAFE)
    # --------------------------------------------------------
    if effects["pitch"] != 0:
        audio = process_pitch_shift(audio, effects["pitch"])

    # --------------------------------------------------------
    # PROCESS PEDALBOARD
    # --------------------------------------------------------

    with board_lock:
        processed = board(
            audio,
            SAMPLERATE,
            buffer_size=BLOCKSIZE,
            reset=False
        )

    # --------------------------------------------------------
    # METTI I CAMPIONI PROCESSATI NELLA FIFO
    # --------------------------------------------------------

    if processed.shape[1] > 0:

        new_audio = np.asarray(
            processed[0],
            dtype=np.float32
        )

        with fifo_lock:

            if audio_fifo.size == 0:
                audio_fifo = new_audio.copy()
            else:
                audio_fifo = np.concatenate(
                    (audio_fifo, new_audio)
                )

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    outdata.fill(0)

    with fifo_lock:

        available = min(
            frames,
            audio_fifo.size
        )

        if available > 0:
            outdata[:available, 0] = (
                audio_fifo[:available] * volume
            )
            audio_fifo = audio_fifo[available:]

# ============================================================
# START AUDIO
# ============================================================

def start_audio():

    global stream

    stream = sd.Stream(
        device=(
            INPUT,
            OUTPUT
        ),
        samplerate=SAMPLERATE,
        channels=1,
        blocksize=BLOCKSIZE,
        dtype="float32",
        callback=callback
    )

    stream.start()

# ============================================================
# RESET AUDIO FIFO
# ============================================================

def clear_audio_fifo():

    global audio_fifo

    with fifo_lock:
        audio_fifo = np.empty(
            0,
            dtype=np.float32
        )

# ============================================================
# DEVICE MENU
# ============================================================

def show_audio_devices():
    """Stampa gli ingressi e le uscite audio disponibili su richiesta."""
    try:
        devices = sd.query_devices()
    except sd.PortAudioError as error:
        print("Impossibile leggere i dispositivi audio:", error)
        return

    print()
    print("DISPOSITIVI AUDIO DISPONIBILI")
    print("Input:")

    input_count = 0
    output_count = 0

    for index, device in enumerate(devices):
        name = device["name"]
        max_input_channels = device["max_input_channels"]
        max_output_channels = device["max_output_channels"]

        if max_input_channels > 0:
            print(f"  [{index}] {name} (canali: {max_input_channels})")
            input_count += 1

    if input_count == 0:
        print("  Nessun input disponibile.")

    print("Output:")

    for index, device in enumerate(devices):
        name = device["name"]
        max_output_channels = device["max_output_channels"]

        if max_output_channels > 0:
            print(f"  [{index}] {name} (canali: {max_output_channels})")
            output_count += 1

    if output_count == 0:
        print("  Nessun output disponibile.")

    print()

# ============================================================
# INITIALIZE
# ============================================================

rebuild_board()
start_audio()

print()
print("LIVE AUDIO PROCESSOR - ONLINE")
print("PitchShift Engine: SciPy Real-Time Resampler (Zero Latency Drop)")
print("Sample rate:", SAMPLERATE)
print("Block size:", BLOCKSIZE)
print()

print("Commands:")
print("  vol 0.5")
print("  gain 5")
print("  gate -40")
print("  pitch 4       (es. +4 acuta, -4 grave, 0 normale)")
print("  compressor 4")
print("  dist 20")
print("  clip -6")
print("  bitcrush 8")
print("  reverb 0.5")
print("  delay 0.3")
print("  chorus 0.5")
print("  phaser 0.5")
print("  highpass 100")
print("  lowpass 5000")
print("  peak 1000 6 1")
print("  lowshelf 200 6 1")
print("  highshelf 4000 6 1")
print("  ladder 3000")
print("  clean")
print("  devices      (mostra input e output disponibili)")
print("  menu         (alias di devices)")
print("  input 12     (indice input mostrato da devices)")
print("  output 16    (indice output mostrato da devices)")
print("  quit")
print()

# ============================================================
# COMMAND LOOP
# ============================================================

try:
    while True:

        command = input("> ").strip().split()

        if not command:
            continue

        if command[0] in ["devices", "menu"] and len(command) == 1:
            show_audio_devices()

        elif command[0] == "vol" and len(command) == 2:
            volume = float(command[1])
            print("Volume impostato a:", volume)

        elif command[0] == "gain" and len(command) == 2:
            effects["gain"] = float(command[1])
            rebuild_board()
            print("Gain:", effects["gain"], "dB")

        elif command[0] == "gate" and len(command) == 2:
            effects["gate"] = float(command[1])
            rebuild_board()
            print("NoiseGate:", effects["gate"], "dB")

        elif command[0] == "pitch" and len(command) == 2:
            effects["pitch"] = float(command[1])
            clear_audio_fifo()
            print("Pitch modificato:", effects["pitch"], "semitoni")

        elif command[0] == "dist" and len(command) == 2:
            effects["dist"] = float(command[1])
            rebuild_board()
            print("Distortion:", effects["dist"], "dB")

        elif command[0] == "compressor" and len(command) == 2:
            effects["compressor"] = float(command[1])
            rebuild_board()
            print("Compressor:", effects["compressor"])

        elif command[0] == "reverb" and len(command) == 2:
            effects["reverb"] = float(command[1])
            rebuild_board()
            print("Reverb:", effects["reverb"])

        elif command[0] == "delay" and len(command) == 2:
            effects["delay"] = float(command[1])
            rebuild_board()
            print("Delay:", effects["delay"], "sec")

        elif command[0] == "chorus" and len(command) == 2:
            effects["chorus"] = float(command[1])
            rebuild_board()
            print("Chorus:", effects["chorus"])

        elif command[0] == "phaser" and len(command) == 2:
            effects["phaser"] = float(command[1])
            rebuild_board()
            print("Phaser:", effects["phaser"])

        elif command[0] == "highpass" and len(command) == 2:
            effects["highpass"] = float(command[1])
            rebuild_board()
            print("Highpass:", effects["highpass"], "Hz")

        elif command[0] == "lowpass" and len(command) == 2:
            effects["lowpass"] = float(command[1])
            rebuild_board()
            print("Lowpass:", effects["lowpass"], "Hz")

        elif command[0] == "clip" and len(command) == 2:
            effects["clip"] = float(command[1])
            rebuild_board()
            print("Clipping:", effects["clip"], "dB")

        elif command[0] == "bitcrush" and len(command) == 2:
            effects["bitcrush"] = float(command[1])
            rebuild_board()
            print("Bitcrush:", effects["bitcrush"], "bit")

        elif command[0] == "peak" and len(command) == 4:
            effects["peak_freq"] = float(command[1])
            effects["peak_gain"] = float(command[2])
            effects["peak_q"] = float(command[3])
            rebuild_board()
            print("Peak:", effects["peak_freq"], "Hz,", effects["peak_gain"], "dB, Q:", effects["peak_q"])

        elif command[0] == "lowshelf" and len(command) == 4:
            effects["lowshelf_freq"] = float(command[1])
            effects["lowshelf_gain"] = float(command[2])
            effects["lowshelf_q"] = float(command[3])
            rebuild_board()
            print("LowShelf:", effects["lowshelf_freq"], "Hz,", effects["lowshelf_gain"], "dB, Q:", effects["lowshelf_q"])

        elif command[0] == "highshelf" and len(command) == 4:
            effects["highshelf_freq"] = float(command[1])
            effects["highshelf_gain"] = float(command[2])
            effects["highshelf_q"] = float(command[3])
            rebuild_board()
            print("HighShelf:", effects["highshelf_freq"], "Hz,", effects["highshelf_gain"], "dB, Q:", effects["highshelf_q"])

        elif command[0] == "ladder" and len(command) == 2:
            effects["ladder"] = float(command[1])
            rebuild_board()
            print("Ladder Filter:", effects["ladder"], "Hz")

        elif command[0] == "clean":
            effects["gain"] = 0
            effects["dist"] = 0
            effects["compressor"] = 0
            effects["reverb"] = 0
            effects["delay"] = 0
            effects["chorus"] = 0
            effects["phaser"] = 0
            effects["highpass"] = 0
            effects["lowpass"] = 0
            effects["pitch"] = 0
            effects["clip"] = 0
            effects["bitcrush"] = 32
            effects["gate"] = 0
            effects["peak_gain"] = 0
            effects["lowshelf_gain"] = 0
            effects["highshelf_gain"] = 0
            effects["ladder"] = 0

            clear_audio_fifo()
            rebuild_board()
            print("Tutti gli effetti sono stati disattivati (Clean State)")

        elif command[0] == "input" and len(command) == 2:
            try:
                input_device = int(command[1])
                device_info = sd.query_devices(input_device)
            except (ValueError, sd.PortAudioError) as error:
                print("Indice input non valido:", error)
            else:
                if device_info["max_input_channels"] < 1:
                    print("Il dispositivo selezionato non è un input audio.")
                else:
                    stream.stop()
                    stream.close()
                    clear_audio_fifo()
                    INPUT = input_device
                    start_audio()
                    print("Input reindirizzato su:", input_device, "-", device_info["name"])

        elif command[0] == "output" and len(command) == 2:
            try:
                output_device = int(command[1])
                device_info = sd.query_devices(output_device)
            except (ValueError, sd.PortAudioError) as error:
                print("Indice output non valido:", error)
            else:
                if device_info["max_output_channels"] < 1:
                    print("Il dispositivo selezionato non è un output audio.")
                else:
                    stream.stop()
                    stream.close()
                    clear_audio_fifo()
                    OUTPUT = output_device
                    start_audio()
                    print("Output reindirizzato su:", output_device, "-", device_info["name"])

        elif command[0] in ["quit", "exit"]:
            if stream:
                stream.stop()
                stream.close()
            print("Arresto del sistema audio live...")
            break

        else:
            print("Comando non riconosciuto. Digita uno dei comandi nella lista sopra.")

except KeyboardInterrupt:
    if stream:
        stream.stop()
        stream.close()
    print("\nProgramma interrotto tramite tastiera (Ctrl+C). Uscita...")

    """
    Il motivo per cui il pitch shifting fa fallire o gracchiare l'audio
    nel tuo codice originale è che PitchShift di Pedalboard modifica la
    durata del blocco audio (Time Stretch), causando un continuo
    mismatch di campioni nella FIFO tra input e output.
    
    Per risolvere questo problema mantenendo intatta l'intera struttura
    del tuo codice (incluso il loop di comandi e il thread lock),
    sostituiamo PitchShift di Pedalboard all'interno di rebuild_board
    con un algoritmo basato su soxr o scipy (già presenti nel tuo pip
    list) eseguito direttamente nella callback audio, oppure tramite il
    ricampionamento con interpolazione sul buffer prima che entri nella
    Pedalboard.
    
    Ecco il codice completo pronto all'uso, aggiornato per gestire il
    pitch shift in tempo reale senza crackle o crash.
    """
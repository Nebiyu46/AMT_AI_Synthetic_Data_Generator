from pydub import AudioSegment
from pydub.playback import play

# Load the long audio file
audio = AudioSegment.from_wav("higher_masinko.wav")

# Define start and end times in milliseconds
start_time = 21300  # Start at 4 seconds
end_time = 22300   # End at 5 seconds (1-second slice)

# Extract and save the steady-state slice
steady_slice = audio[start_time:end_time]
steady_slice.export("test_output.wav", format="wav")
"""
Runs ffmpeg to convert the input file to mono 16 bit 44.1 kHz audio. 
Modifies the original file
"""

import subprocess

files = []

for input_file in files:
    """
    Run ffmpeg,
    automatically answering "yes" if asked if we want to overwite any output files,
    using the given input file,
    setting the number of audio channels to 1,
    setting the audio rate to 44100 Hz,
    setting the sample format to signed 16 bit integer,
    to the given output file.

    Then, move the temp file on top of the input file
    """
    
    temp_file = "/tmp/fix-wav-format.wav"
    
    args = []
    args += ["ffmpeg"]
    args += ["-y"]
    args += ["-i", input_file]
    args += ["-ac", "1"]
    args += ["-ar", "44100"]
    args += ["-sample_fmt", "s16"]
    args += [temp_file]

    print(" ".join(args))
    subprocess.run(args)

    args = []
    args += ["mv"]
    args += [temp_file]
    args += [input_file]

    print(" ".join(args))
    subprocess.run(args)

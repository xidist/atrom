import mido
from mido import MidiFile
import pprint
import os
import argparse

def make_pitch_interval_representation(mid):
    """
    Input: a MidiFile as opened by mido
    Output: a list whose elements are three-element tuples, where:
        the first element is the midi pitch value
        the second element is the start time of the note in seconds
        the third element is the end time of the note in seconds
    """
    
    result = []
    for track in mid.tracks:
        current_notes_to_start_times = {}
        current_time_in_seconds = 0
        current_microseconds_per_beat = 500000 # default in midi specification
        
        for msg in track:
            # update the current time...
            delta_in_beats = msg.time / mid.ticks_per_beat
            delta_in_microseconds = current_microseconds_per_beat * delta_in_beats
            delta_in_seconds = delta_in_microseconds * 1e-6
            current_time_in_seconds += delta_in_seconds

            if msg.type == "set_tempo":
                current_microseconds_per_beat = msg.tempo

            elif msg.type == "note_on" and msg.velocity > 0:
                current_notes_to_start_times[msg.note] = current_time_in_seconds

            elif (msg.type == "note_on" and msg.velocity == 0) or msg.type == "note_off":
                if msg.note in current_notes_to_start_times:
                    start_time = current_notes_to_start_times[msg.note]
                    del current_notes_to_start_times[msg.note]
                    result.append((msg.note, start_time, current_time_in_seconds))

    return result
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("inputFilePath")
    parser.add_argument("-y", "--yes", action="store_true")
    parser.add_argument("-n", "--no", action="store_true")

    args = parser.parse_args()
    
    outPath = os.path.splitext(args.inputFilePath)[0] + ".txt"

    shouldWrite = True
    if os.path.exists(outPath):
        if args.yes:
            shouldWrite = True
        elif args.no:
            shouldWrite = False
        else:
            if input(f"File {outPath} exists. Overwrite? (y/N) ").lower() != "y":
                shouldWrite = False

        if shouldWrite:
            print(f"Overwriting {outPath}")
        else:
            print(f"Not overwriting {outPath}")

    if shouldWrite:
        mid = mido.MidiFile(args.inputFilePath)
        with open(outPath, "w") as f:
            f.write("\n".join([str(x) for x in make_pitch_interval_representation(mid)]))


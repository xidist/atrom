import mido
from mido import MidiFile
import pprint
import os

def get_ticks_to_seconds_and_tempo_changes(mid):
    """
    Input: a MidiFile as opened by mido
    Output: a dictionary where keys are ticks from the start of the file,
            and values are tuples of time in seconds (adjusted for set_tempo events)
            and the set_tempo MetaMessages
    """
    
    ticks_per_beat = mid.ticks_per_beat
    
    # start with the MIDI default of 120 BPM until we get a tempo change event
    all_tempo_changes = [(0, mido.MetaMessage("set_tempo", tempo=500000, time=0))]

    # fill in all_tempo_changes...
    for track in mid.tracks:
        current_time_in_ticks = 0
        for msg in track:
            if msg.type == "set_tempo":
                all_tempo_changes.append((current_time_in_ticks, msg))
                
            current_time_in_ticks += msg.time

    # start with the MIDI default of 120 BPM until we get a tempo change event
    ticks_to_seconds_and_tempo_changes = {0 : (0, mido.MetaMessage("set_tempo", tempo=500000, time=0))}

    # fill in ticks_to_seconds_and_tempo_changes...
    for i, (ticks_from_start, tempo_change) in enumerate(all_tempo_changes):
        # we prepopulated the first entry, so skip it
        if i == 0:
            continue

        # lots of unit conversion going on.
        # we need to get the duration of the last tempo change (i.e. how long it lasted),
        # and figure out when the last tempo change happened, in seconds,
        # to figure out when the current tempo change happens, in seconds
        previous_tempo_change = all_tempo_changes[i - 1]
        previous_tempo_change_duration_in_ticks = ticks_from_start - previous_tempo_change[0]
        previous_tempo_change_duration_in_beats = previous_tempo_change_duration_in_ticks / ticks_per_beat
        previous_tempo_change_duration_in_microseconds = previous_tempo_change_duration_in_beats * previous_tempo_change[1].tempo
        previous_tempo_change_duration_in_seconds = previous_tempo_change_duration_in_microseconds / 1_000_000
        previous_tempo_change_time_in_ticks = max(ticks_to_seconds_and_tempo_changes.keys())
        previous_tempo_change_time_in_seconds = ticks_to_seconds_and_tempo_changes[previous_tempo_change_time_in_ticks][0]

        tempo_change_time_in_seconds = previous_tempo_change_time_in_seconds + previous_tempo_change_duration_in_seconds

        # finally, store it in the dictionary
        ticks_to_seconds_and_tempo_changes[ticks_from_start] = (tempo_change_time_in_seconds, tempo_change)

    return ticks_to_seconds_and_tempo_changes

def get_concurrent_note_stats(mid):
    """
    Input: a MidiFile as opened by mido
    """
    def get_stats(concurrency):
        m = max(concurrency)
        raw_avg = sum(concurrency) / (len(concurrency) or 1)
        zeros_omitted_avg = sum(concurrency) / (len([x for x in concurrency if x != 0]) or 1)

        return m, raw_avg, zeros_omitted_avg

    stats_per_track = []
    for track in mid.tracks:
        note_is_present = {(c, k) : False for k in range(128) for c in range(16)}
        concurrency = []
        for msg in track:
            if msg.type == "note_on":
                note_is_present[(msg.channel, msg.note)] = True
            elif msg.type == "note_off":
                note_is_present[(msg.channel, msg.note)] = False

            concurrency.append(sum(note_is_present.values()))
        stats_per_track.append(get_stats(concurrency))
        
    return stats_per_track

def get_concurrent_note_stats_from_filenames(filenames):
    m = 0
    raw_avg = 0
    zeros_omitted_avg = 0
    for i, filename in enumerate(filenames):
        if i % 10 == 0:
            print(i, len(filenames))
        
        mid = MidiFile(filename)
        stats = get_concurrent_note_stats(mid)
        should_print = False
        for track in stats:
            if track[0] > m:
                should_print = True
                m = track[0]
            if track[1] > raw_avg:
                should_print = True
                raw_avg = track[1]
            if track[2] > zeros_omitted_avg:
                should_print = True
                zeros_omitted_avg = track[2]

        if should_print:
            print(filename, stats)

def get_maestro_midi_filenames():
    result = []
    
    years = ["2004", "2006", "2008", "2009", "2011", "2013", "2014", "2015", "2017", "2018"]
    for year in years:
        searchDir = "/z/atrom/datasets/labeled/maestro/maestro-v3.0.0/" + year + "/"
        for file in os.listdir(searchDir):
            if os.path.splitext(file)[1] == ".mid" or os.path.splitext(file)[1] == ".midi":
                result.append(searchDir + file)
    return result


if __name__ == "__main__":
    filenames = get_maestro_midi_filenames()
    get_concurrent_note_stats_from_filenames(filenames)

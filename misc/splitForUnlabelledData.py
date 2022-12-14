import random
import json
import re

class Song:
    """
    Represents one entry in the maestro dataset
    """
    def __init__(self, canonical_composer, canonical_title,
                 split, year, midi_filename, audio_filename,
                 duration):
        self.canonical_composer = canonical_composer
        self.canonical_title = canonical_title
        self.split = split
        self.year = year
        self.midi_filename = midi_filename
        self.audio_filename = audio_filename

    def doesNameMatch(self, name):
        """
        Returns True if the name matches against self. 

        We use a fairly conservative fuzzy-string matching system. 
        """
        
        # lowercase the string...
        name = name.lower()
        # split on word boundaries...
        inputParts = re.findall(r"[a-zA-Z]+", name)

        # combine semantically relevant song information into a mask...
        mask = " ".join([self.canonical_composer, self.canonical_title]).lower()
        # split on word boundaries...
        maskParts = set(re.findall(r"[a-zA-Z]+", mask))
        # get the parts of the input word that are in the mask
        matches = [part for part in inputParts if part in maskParts]
        # get the ratio of the number of matches to the number of input parts
        matchRatio = len(matches) / len(inputParts)

        # if the match ratio is higher than a certain threshold,
        # we think the name matches against us
        if matchRatio > 0.5:
            return True
        else:
            return False

class Splitter:
    """
    Determines the split for unlabelled data, based on their file names
    """
    def __init__(self, pathToMaestroJson,
                 trainingProbability=0.7,
                 validationProbability=0.15):
        """
        pathToMaestroJson: str. the path to maestro-v3.0.0.json
        trainingProbability: float. the probability a non-maestro song will be
            put into the training set. defaults to 0.7
        validationProbability: float. the probability a non-maestro song will be
            put into the validation set. defaults to 0.15
        """
        self.trainingProbability = trainingProbability
        self.validationProbability = validationProbability
        
        jsonContents = json.loads(open(pathToMaestroJson).read())
        
        # keys are "canonical_composer", "canonical_title", "split",
        # "year", "midi_filename", "audio_filename", "duration".
        # values are dictionaries where the keys are consistent
        # across different dictionaries, and the values are strings or numbers.
        
        self.songs = []
        for key in jsonContents["canonical_composer"].keys():
            canonical_composer = jsonContents["canonical_composer"][key]
            canonical_title = jsonContents["canonical_title"][key]
            split = jsonContents["split"][key]
            year = jsonContents["year"][key]
            midi_filename = jsonContents["midi_filename"][key]
            audio_filename = jsonContents["audio_filename"][key]
            duration = jsonContents["duration"][key]
            
            newSong = Song(canonical_composer, canonical_title,
                           split, year, midi_filename, audio_filename,
                           duration)
            self.songs.append(newSong)

    def getCanonicalFormOfName(self, name):
        """
        Attempts to normalize name, so that slightly different strings
        that are likely the same are treated as the same
        """

        # lowercase the string...
        name = name.lower()
        # split on word boundaries...
        parts = re.findall(r"[a-zA-Z]+", name)
        # sort lexicographically...
        parts.sort()
        # and join with spaces
        return " ".join(parts)

    def getSplitForFile(self, name):
        """
        Returns "train", "validation", "test", or "TEST".
        If the name matches an item from the maestro dataset,
        "TEST" is returned. Otherwise, the return value is
        pseudo-random, based on the name
        """
        for song in self.songs:
            if song.doesNameMatch(name):
                return "TEST"

        # seed the random number generator for determinism
        random.seed(self.getCanonicalFormOfName(name))
        rv = random.uniform(0, 1)
        if rv < self.trainingProbability:
            return "train"
        elif rv < self.trainingProbability + self.validationProbability:
            return "validation"
        else:
            return "test"
    
def getSplitsForFiles(files):
    """
    files: list of strings

    Returns a dictionary where the keys are elements in files,
    and the values are "train", "validation", "test", or "TEST". 
    "TEST" indicates the file name matched a maestro song
    """
    splitter = Splitter("maestro-v3.0.0.json")
    return {f:splitter.getSplitForFile(f) for f in files}

import pprint
files = []
splits = getSplitsForFiles(files)
pprint.pprint(getSplitsForFiles(files))

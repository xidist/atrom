import math, itertools

class Tokenizer:
    def __init__(windowSize, timeGranularity):
        """
        windowSize: the size of each window for tokenization, in seconds. 
            audio longer than this duration is broken up into multiple windows
            and tokenized separately
        timeGranularity: the quantization level of time, in seconds. 
            note events that happen less than this distance apart may
            be tokenized to occur at the same moment in time
        """

        self.windowSize = windowSize
        self.timeGranularity = timeGranularity

        self.tokenList = []
        for i in range(1, 129):
            self.tokens.append(self.makeNoteToken(i))
        self.tokenList.append("<ON>")
        self.tokenList.append("<OFF>")
        self.tokenList.append("<EOS>")
        for i in range(1 + int(math.ceil(windowSize / timeGranularity))):
            self.tokenList.append(self.makeTimeToken(i))

    def stringToToken(self, s):
        """
        s: the human-readable version of the token. 
           may be a <NOTE_{}>, <ON>, <OFF>, <EOS>, or <TIME_{}> like string
        
        returns: a non-negative integer uniquely representing that token
        """
        return self.tokenList.index(s)

    def tokenToString(self, i):
        """
        i: a non-negative integer uniquely representing a token
        
        returns: the human-readable version of the token
        """
        return self.tokenList[i]

    def eosToken(self):
        return self.stringToInt("<EOS>")

    def makeNoteToken(self, midiPitch):
        return f"<NOTE_{midiPitch}_>"

    def makeTimeToken(self, t):
        return f"<TIME_{t}_{self.timeGranularity}>"

    def detokenize(self, windows):
        """
        windows: a list of windows, where each window is a list of integer tokens

        returns: a list of three-element tuples in pitch-interval format
        """
        noteStartTimes = {}
        result = []
        time = None
        for i in range(len(windows)):
            isOn = None
            foundEos = False
            for event in windows[i]:
                s = self.tokenToString(event)
                if event.startswith("<TIME_"):
                    newTime = int(event.split("_")[1])
                    newTime *= i * (self.windowSize / self.timeGranularity)
                    if newTime <= time:
                        print("detokenizing error: newTime less than time")
                    else:
                        time = newTime
                elif event == "<ON>":
                    isOn = True
                elif event == "<OFF>":
                    isOn = False
                elif event.startswith("<NOTE_"):
                    if time is None or isOn is None:
                        print("detokenizing error: note without time/on-off")
                    
                    note = int(event.split("_")[1])
                    if note in noteStartTimes:
                        if isOn:
                            print("detokenizing error: reiterating on note")
                        else:
                            result.append((note, noteStartTimes[note], time))
                            del noteStartTimes[note]
                    else:
                        if not isOn:
                            print("detokenizing error: ending non-started note")
                        else:
                            noteStartTimes[note] = time
                elif event == "<EOS>":
                    if foundEos:
                        print("detokenizing error: found multiple eos")
                    foundEos = True
                    
            # end of for event in windows[1]
            if not foundEos:
                print("detokenizing error: no eos")

        # end of for i in range(len(windows))
        if len(noteStartTimes) != 0:
            print("detokenizing error: unterminated notes")
        return result

    def tokenize(self, pitchIntervals):
        """
        pitchIntervals: list of three-element tuples. 
            the first element is the midi pitch (integer from 1-128)
            the second element is the start time of the note, in seconds
            the third element is the end time of the note, in seconds

        returns: list of windows, where each window is a list of integer tokens
        """
        ungroupedEvents = []
        for x in pitchIntervals:
            pitch = x[0]
            startTime = int(math.round(x[1] / self.timeGranularity))
            endTime = int(math.round(x[2] / self.timeGranularity))
            ungroupedEvents.append((startTime, True, pitch))
            ungroupedEvents.append((endTime, False, pitch))
        ungroupedEvents.sort()

        windows = itertools.groupby(ungroupedEvents, lambda x: int(x / (self.windowSize / self.timeGranularity)))
        for i in range(len(windows)):
            window = windows[i]
            time = -1
            isOn = None

            windows[i] = []
            for event in window:
                if event[0] != time:
                    time = event[0]
                    windows[i].append(self.makeTimeToken(time))
                if event[1] != isOn:
                    isOn = event[1]
                    windows[i].append(self.stringToToken("<ON>" if isOn else "<OFF>"))
                windows[i].append(self.makeNoteToken(event[2]))
        return windows
        
        

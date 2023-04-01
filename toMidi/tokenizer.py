import math, itertools

class Tokenizer:
    def __init__(self, windowSize, timeGranularity):
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
            self.tokenList.append(self.makeNoteString(i))
        self.tokenList.append(self.onString())
        self.tokenList.append(self.offString())
        self.tokenList.append(self.sosString())
        self.tokenList.append(self.eosString())
        self.tokenList.append(self.padString())
        for i in range(1 + int(math.ceil(windowSize / timeGranularity))):
            self.tokenList.append(self.makeTimeString(i))

    def vocab_size(self) -> int:
        return len(self.tokenList)

    def stringToToken(self, s):
        """
        s: the human-readable version of the token. 
           may be a <NOTE_{}>, <ON>, <OFF>, <SOS>, <EOS>, <PAD>, or <TIME_{}> like string
        
        returns: a non-negative integer uniquely representing that token
        """
        return self.tokenList.index(s)

    def tokenToString(self, i):
        """
        i: a non-negative integer uniquely representing a token
        
        returns: the human-readable version of the token
        """
        return self.tokenList[i]

    def sosString(self):
        return "<SOS>"

    def sosIndex(self):
        return self.stringToToken(self.sosString())

    def eosString(self):
        return "<EOS>"

    def eosIndex(self):
        return self.stringToToken(self.eosString())

    def padString(self):
        return "<PAD>"

    def padIndex(self):
        return self.stringToToken(self.padString())

    def onString(self):
        return "<ON>"

    def offString(self):
        return "<OFF>"

    def makeNoteString(self, midiPitch):
        return f"<NOTE_{midiPitch}_>"

    def makeTimeString(self, t):
        return f"<TIME_{t}_{self.timeGranularity}>"

    def detokenize(self, windows):
        """
        windows: a list of windows, where each window is a list of string or integer tokens

        returns: a list of three-element tuples in pitch-interval format
        """
        if windows and windows[0] and type(windows[0][0]) == int:
            windows = [[self.tokenToString(x) for x in w] for w in windows]
        
        noteStartTimes = {}
        result = []
        time = None
        for i in range(len(windows)):
            isOn = None
            foundEos = False
            for eventIndex, event in enumerate(windows[i]):
                if event.startswith("<TIME_"):
                    newTime = int(event.split("_")[1]) * self.timeGranularity
                    newTime += i * self.windowSize
                    if time is not None and newTime <= time:
                        print("detokenizing error: newTime less than time")
                    else:
                        time = newTime
                elif event == self.onString():
                    isOn = True
                elif event == self.offString():
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
                elif event == self.eosString():
                    if foundEos:
                        print("detokenizing error: found multiple eos")
                    foundEos = True

                elif event == self.sosString():
                    if eventIndex != 0:
                        print("detokenizing error: found extra sos")

                elif event == self.padString():
                    if not foundEos:
                        print("detokenizing error: padding before eos")
                    
            # end of for event in windows[1]
            if not foundEos:
                print("detokenizing error: no eos")

        # end of for i in range(len(windows))
        if len(noteStartTimes) != 0:
            print("detokenizing error: unterminated notes")
        return result

    def makeWindows(self, pitchIntervals):
        """
        pitchIntervals: list of three-element tuples. 
            the first element is the midi pitch (integer from 1-128)
            the second element is the start time of the note, in seconds
            the third element is the end time of the note, in seconds

        returns: a list of windows, where each window is a list of three-element tuples. 
            the first element is the quantized (integer) time of the event, measured from the start of the window
            the second element is True iff the event is a note-on event
            the third element is the midi pitch (integer from 1-128)
        """
        ungroupedEvents = []
        for x in pitchIntervals:
            pitch = x[0]
            startTime = int(round(x[1] / self.timeGranularity))
            endTime = int(round(x[2] / self.timeGranularity))
            ungroupedEvents.append((startTime, True, pitch))
            ungroupedEvents.append((endTime, False, pitch))
        ungroupedEvents.sort()

        grouped = itertools.groupby(ungroupedEvents, lambda x: int(x[0] / (self.windowSize / self.timeGranularity)))
        windows = []

        for (key, v) in grouped:
            while(len(windows)) <= key:
                windows.append([])
            timeCorrectedV = [(
                x[0] % int(self.windowSize / self.timeGranularity),
                x[1],
                x[2]
                ) for x in v]
            windows[-1] = timeCorrectedV
        return windows

    def tokenize(self, pitchIntervals, batchSize: int, toInts=True, padToLength: int=100):
        """
        pitchIntervals: list of three-element tuples. 
            the first element is the midi pitch (integer from 1-128)
            the second element is the start time of the note, in seconds
            the third element is the end time of the note, in seconds

        returns: list of windows, where each window is a list of string or int tokens
        """

        windows = self.makeWindows(pitchIntervals)
        result = []
        for window in windows:
            result.append([self.sosString()])
            time = -1
            isOn = None

            for event in window:
                if event[0] != time:
                    time = event[0]
                    s = self.makeTimeString(time)
                    result[-1].append(s)
                if event[1] != isOn:
                    isOn = event[1]
                    s = self.onString() if isOn else self.offString()
                    result[-1].append(s)
                s = self.makeNoteString(event[2])
                result[-1].append(s)

            result[-1].append(self.eosString())

            if len(result[-1]) < padToLength:
                result[-1] += [self.padString() for _ in range(padToLength - len(result[-1]))]

        while len(result) < batchSize:
            result.append([self.padString() for _ in range(padToLength)])

        if toInts:
            result = [[self.stringToToken(s) for s in w] for w in result]
        return result

if __name__ == "__main__":
    pitchIntervals = (
        [
         (60, 0, 0.5),
         (62, 0.6, 1.2),
         (64, 0.3, 4.5101)
         ]
    )
    
    tokenizer = Tokenizer(1, 0.05)
    tokens = tokenizer.tokenize(pitchIntervals, toInts=False)
    print(tokens)
    detokens = tokenizer.detokenize(tokens)
    print(detokens)

import mir_eval
import numpy as np

def _toHz(x):
    if not isinstance(x, list):
        return 440 * 2 ** ((x - 69) / 12)
    return [_toHz(y) for y in x]

def getAllMetrics(actual, estimated):
    """
    actual: list of three tuples, where:
            the first element is the midi pitch,
            the second element is the onset time in seconds,
            the third element is the offset time in seconds

    estimated: same type as actual

    returns: dictionary with string keys and float values corresponding to the 
             values of different metrics. keys are: 
             Precision, Recall, F-measure, Average_Overlap_Ratio, 
             Precision_no_offset, Recall_no_offset, F-measure_no_offset, Average_Overlap_Ratio_no_offset, 
             Onset_Precision, Onset_Recall, Onset_F-measure, 
             Offset_Precision, Offset_Recall, Offset_F-measure
    """
    actualPitches = np.array(_toHz([x[0] for x in actual]))
    actualIntervals = np.array([(x[1], x[2]) for x in actual])
    estimatedPitches = np.array(_toHz([x[0] for x in estimated]))
    estimatedIntervals = np.array([(x[1], x[2]) for x in estimated])
    
    mir_eval.transcription.validate(actualIntervals, actualPitches,
                                    estimatedIntervals, estimatedPitches)
    allMetrics = mir_eval.transcription.evaluate(actualIntervals, actualPitches, estimatedIntervals, estimatedPitches)
    return allMetrics

def getFMeasure(actual, estimated):
    """
    actual: list of three tuples, where:
            the first element is the midi pitch,
            the second element is the onset time in seconds,
            the third element is the offset time in seconds

    estimated: same type as actual

    returns: the F-measure including the offset of notes
    """
    return getAllMetrics(actual, estimated)["F-measure"]

def getFMeasure_no_offset(actual, estimated):
    """
    actual: list of three tuples, where:
            the first element is the midi pitch,
            the second element is the onset time in seconds,
            the third element is the offset time in seconds

    estimated: same type as actual

    returns: the F-measure excluding the offset of notes
    """

    return getAllMetrics(actual, estimated)["F-measure_no_offset"]


if __name__ == "__main__":
    actual = [(60, 0, 0.5),
              (62, 0.6, 1.2),
              (64, 0.3, 4.5101)]
    estimated = [(60.1, 0, 0.2),
                 (62, 0.6, 1.2),
                 (64, 0.3, 4.5101)]
    
    print(getFMeasure(actual, estimated))
    print(getFMeasure_no_offset(actual, estimated))

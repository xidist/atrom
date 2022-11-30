import matplotlib.pyplot as plt

def printLogFile(logFile, everyN=100, alwaysPrintLast=True):
    digitCtr = 0

    with open(logFile) as f:
        for l in f.readlines():
            if l[0].isdigit():
                digitCtr += 1

            if not l[0].isdigit() or (digitCtr % everyN == 0):
                print(l, end="")

    if alwaysPrintLast:
        with open(logFile) as f:
            print(f.readlines()[-1])

def graphLoss(logFile, everyN=100):
    digitCtr = 0

    batches = []
    losses = []

    maxBatchN = -1
    with open(logFile) as f:
        for l in f.readlines():
            if l[0].isdigit():
                batch = int(l.split("|")[1].strip())
                if batch > maxBatchN:
                    maxBatchN = batch
                else:
                    break

    with open(logFile) as f:
        for l in f.readlines():
            if l[0].isdigit():
                digitCtr += 1

            if l[0].isdigit() and (digitCtr % everyN == 0):
                epoch = int(l.split("|")[0].strip())
                batch = int(l.split("|")[1].strip()) / 3600
                loss = float(l.split("|")[2].strip())
                batches.append((epoch * maxBatchN / 3600) + batch)
                losses.append(loss)

    plt.plot(batches, losses)
    plt.show()

printLogFile("nohup.out")
graphLoss("nohup.out")

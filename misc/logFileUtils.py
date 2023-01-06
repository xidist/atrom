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

def graphLoss(logFile, everyN=1):
    digitCtr = 0

    batches = []
    train_losses = []
    val_losses = []

    maxBatchN = -1
    with open(logFile) as f:
        for l in f.readlines():
            if l[0].isdigit():
                batch = int(l.split("|")[1].strip())
                if batch > maxBatchN:
                    maxBatchN = batch
                else:
                    break

    batchesPerEpoch = maxBatchN + 1

    maxEpochN = 0
    with open(logFile) as f:
        for l in f.readlines():
            if l[0].isdigit():
                maxEpochN = max(maxEpochN, int(l.split("|")[0].strip()))

    nEpochs = maxEpochN + 1

    endOfEpochs = []

    with open(logFile) as f:
        for l in f.readlines():
            if l[0].isdigit():
                digitCtr += 1

            if l[0].isdigit() and (digitCtr % everyN == 0):
                epoch = int(l.split("|")[0].strip())
                batch = int(l.split("|")[1].strip()) / 3600
                train_loss = float(l.split("|")[2].strip())
                batches.append((epoch * batchesPerEpoch / 3600) + batch)
                train_losses.append(train_loss)

                if len(l.split("|")) > 3:
                    val_loss = float(l.split("|")[3].strip())
                    val_losses.append(((epoch * batchesPerEpoch / 3600) + batch, val_loss))

                if epoch >= len(endOfEpochs):
                    endOfEpochs.append((epoch, batch, train_loss))
                else:
                    if endOfEpochs[-1][1] < batch:
                        endOfEpochs[-1] = ((epoch, batch, train_loss))

        
    plt.plot(batches, train_losses)

    plt.plot([(e * batchesPerEpoch / 3600) + b for e, b, _ in endOfEpochs],
             [l for _, _, l in endOfEpochs],
             color="red")

    print("training losses at end of epochs:")
    for e, b, l in endOfEpochs:
        print(f"{e} ({b}): {l}")

    for i in range(nEpochs):
        plt.axvline(i * batchesPerEpoch / 3600, color="black")

    plt.plot([x[0] for x in val_losses], [x[1] for x in val_losses], color="green")

    plt.show()

printLogFile("nohup.out")
graphLoss("nohup.out")

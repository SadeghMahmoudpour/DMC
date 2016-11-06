def convert_to_csv(path, inputFileName, outputFileName):
    inputPath = '{path}/{inputFileName}'.format(path=path, inputFileName=inputFileName)
    outputPath = '{path}/{outputFileName}'.format(path=path, outputFileName=outputFileName)

    outputFile = open(outputPath, 'w', encoding='utf-8')

    with open(inputPath, 'r', encoding='utf-8') as inputFile:
        for line in inputFile:
            csvline = line.replace(';', ',')
            outputFile.write(csvline)

        outputFile.close()

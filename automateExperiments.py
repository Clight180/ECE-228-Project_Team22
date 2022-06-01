import config
from main import Experiment

SpecList = [(8,000,000), (16,000,000)]

for spec in SpecList:
    config.numAngles = spec[0]
    config.modelNum = spec[1]
    config.datasetID = spec[2]
    Experiment()
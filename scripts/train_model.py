modelName = "log_transform"  # original configs

args = {}
args["outputDir"] = "/home/necl/data/brain2speech/logs/speech_logs/" + modelName
args["datasetPath"] = "/home/necl/data/brain2speech/ptDecoder_ctc"
args["seqLen"] = 150
args["maxTimeSeriesLen"] = 1200
args["batchSize"] = 128
args["eps"] = 0.1
args["l2_decay"] = 0.00001
args["lrStart"] = 0.05
args["lrEnd"] = 0.02
args["nUnits"] = 256
args["nBatch"] = 10000
args["nLayers"] = 5
args["seed"] = 0
args["nClasses"] = 40
args["nInputFeatures"] = 256
args["dropout"] = 0.4 
args["whiteNoiseSD"] = 0.8
args["constantOffsetSD"] = 0.2
args["gaussianSmoothWidth"] = 2.0
args["strideLen"] = 4
args["kernelLen"] = 33
args["bidirectional"] = False  
args["timeMask_maxWidth"] = 0
args["timeMask_nMasks"] = 0
args["timeMask_p"] = 0

from neural_decoder.neural_decoder_trainer import trainModel

trainModel(args)


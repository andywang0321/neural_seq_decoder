modelName = "speechBaseline4"  # original configs

args = {}
args["outputDir"] = "/home3/matthew/data/brain2speech/logs/speech_logs/" + modelName
args["datasetPath"] = "/home3/matthew/data/brain2speech/ptDecoder_ctc"
args["seqLen"] = 150
args["maxTimeSeriesLen"] = 1200
args["batchSize"] = 128 # 64
args["lrStart"] = 0.05  # 0.02
args["lrEnd"] = 0.02
args["nUnits"] = 1024
args["nBatch"] = 5000  # 10000
args["nLayers"] = 5
args["seed"] = 0
args["nClasses"] = 40
args["nInputFeatures"] = 256
args["dropout"] = 0.2  # 0.4
args["whiteNoiseSD"] = 0.8
args["constantOffsetSD"] = 0.2
args["gaussianSmoothWidth"] = 2.0
args["strideLen"] = 4
args["kernelLen"] = 33
args["bidirectional"] = False  # True
args["l2_decay"] = 1e-5
args["timeMask_maxWidth"] = 300
args["timeMask_nMasks"] = 2
args["timeMask_p"] = 0.5

from neural_decoder.neural_decoder_trainer import trainModel

trainModel(args)


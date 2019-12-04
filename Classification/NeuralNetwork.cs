using Microsoft.ML;
using System;
using System.IO;
using static Microsoft.ML.DataOperationsCatalog;

namespace Classification
{
    public static class NeuralNetwork
    {
        private static string _appPath => Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);
        private static string _modelPath => Path.Combine(_appPath, "Models", "model.zip");

        private static MLContext _mlContext;
        private static PredictionEngine<Diseases, PredictionDiseases> _predEngine;
        private static ITransformer _trainedModel;

        private static IDataView trainData;
        private static IDataView testData;

        public static IDataView LoadData(string pathDataSet)
        {
            _mlContext = new MLContext(seed: 0);
            return _mlContext.Data.LoadFromTextFile<Diseases>(pathDataSet, separatorChar: ';', hasHeader: true);
        }

        public static IEstimator<ITransformer> ProcessData()
        {
            var pipeline = _mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: "Disease", outputColumnName: "Label")
                .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Sym1", outputColumnName: "Sym1Featurized"))
                .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Sym2", outputColumnName: "Sym2Featurized"))
                .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Sym3", outputColumnName: "Sym3Featurized"))
                .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Sym4", outputColumnName: "Sym4Featurized"))
                .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Sym5", outputColumnName: "Sym5Featurized"))
                .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Sym6", outputColumnName: "Sym6Featurized"))
                .Append(_mlContext.Transforms.Concatenate("Features", "Sym1Featurized", "Sym2Featurized", "Sym3Featurized", "Sym4Featurized", "Sym5Featurized", "Sym6Featurized"))
                .AppendCacheCheckpoint(_mlContext);

            return pipeline;
        }
        public static IEstimator<ITransformer> BuildAndTrainModel(IDataView trainingDataView, IEstimator<ITransformer> pipeline, double testFraction)
        {
            var trainingPipeline = pipeline.Append(_mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features")
                .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel")));

            TrainTestData dataSplit = _mlContext.Data.TrainTestSplit(trainingDataView, testFraction: testFraction);
            trainData = dataSplit.TrainSet;
            testData = dataSplit.TestSet;

            _trainedModel = trainingPipeline.Fit(trainData);

            _predEngine = _mlContext.Model.CreatePredictionEngine<Diseases, PredictionDiseases>(_trainedModel);

            return trainingPipeline;
        }
        public static void Evaluate(DataViewSchema trainingDataViewSchema)
        {
            var testMetrics = _mlContext.MulticlassClassification.Evaluate(_trainedModel.Transform(testData));

            Console.WriteLine($"=============== Evaluating to get model's accuracy metrics - Ending time: {DateTime.Now.ToString()} ===============");

            Console.WriteLine($"*************************************************************************************************************");
            Console.WriteLine($"*       Metrics for Multi-class Classification model - Test Data     ");
            Console.WriteLine($"*------------------------------------------------------------------------------------------------------------");
            Console.WriteLine($"*       MicroAccuracy:    {testMetrics.MicroAccuracy:0.###}");
            Console.WriteLine($"*       MacroAccuracy:    {testMetrics.MacroAccuracy:0.###}");
            Console.WriteLine($"*       LogLoss:          {testMetrics.LogLoss:#.###}");
            Console.WriteLine($"*       LogLossReduction: {testMetrics.LogLossReduction:#.###}");
            Console.WriteLine($"*************************************************************************************************************");

            SaveModelAsFile(_mlContext, trainingDataViewSchema, _trainedModel);
        }
        public static void PredictDisease(Diseases diseases)
        {
            ITransformer loadedModel = _mlContext.Model.Load(_modelPath, out var modelInputSchema);

            _predEngine = _mlContext.Model.CreatePredictionEngine<Diseases, PredictionDiseases>(loadedModel);

            var prediction = _predEngine.Predict(diseases);

            Console.WriteLine($"=============== Single Prediction - Result: {prediction.Disease} ===============");
        }
        private static void SaveModelAsFile(MLContext mlContext, DataViewSchema trainingDataViewSchema, ITransformer trainedModel)
        {
            mlContext.Model.Save(trainedModel, trainingDataViewSchema, _modelPath);

            Console.WriteLine("The model is saved to {0}", _modelPath);
        }

        public static void PredictDiseases()
        {
            ITransformer loadedModel = _mlContext.Model.Load(_modelPath, out var modelInputSchema);
            _predEngine = _mlContext.Model.CreatePredictionEngine<Diseases, PredictionDiseases>(loadedModel);

            Diseases[] diseases = new Diseases[]
            {
                new Diseases()
                {
                    Sym1="coughing",
                    Sym2="headache",
                    Sym3="hurt glatat",
                    Sym4="temperature",
                    Sym5="malaise"
                },
                new Diseases()
                {
                    Sym1="seeing double;",
                    Sym2="the inability to straighten legs",
                    Sym4="stiff neck",
                    Sym5="skin rash"
                }
            };

            IDataView batchDiseases = _mlContext.Data.LoadFromEnumerable<Diseases>(diseases);

            var predictedResults = _mlContext.Data.CreateEnumerable<Diseases>(batchDiseases, reuseRowObject: false);

            foreach (Diseases prediction in predictedResults)
            {
                Console.WriteLine($"Prediction: {_predEngine.Predict(prediction).Disease}");
            }
        }
    }
}

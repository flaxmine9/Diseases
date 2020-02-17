using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using static Microsoft.ML.DataOperationsCatalog;

namespace Classification
{
    public static class NeuralNetwork
    {
        private static string _appPath => Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);
        private static string _modelPath => Path.Combine(_appPath, "Models", "SdcaLogisticRegression.zip");

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
                .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Sex", outputColumnName: "SexFeaturized"))
                .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Age", outputColumnName: "AgeFeaturized"))
                .Append(_mlContext.Transforms.Concatenate("Features", "Sym1Featurized", "Sym2Featurized", "Sym3Featurized", "Sym4Featurized", "Sym5Featurized", "Sym6Featurized", "SexFeaturized", "AgeFeaturized"))
                .AppendCacheCheckpoint(_mlContext);

            return pipeline;
        }
        public static IEstimator<ITransformer> BuildAndTrainModel(IDataView trainingDataView, IEstimator<ITransformer> pipeline, double testFraction)
        {
            var trainingPipeline = pipeline.Append(_mlContext.MulticlassClassification.Trainers.OneVersusAll(_mlContext.BinaryClassification.Trainers.SdcaLogisticRegression("Label", "Features"), "Label")
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
        public static Dictionary<string, float> PredictDisease(Diseases diseases)
        {
            Encoding.RegisterProvider(CodePagesEncodingProvider.Instance);
            Encoding.GetEncoding(1251);

            _mlContext = new MLContext(seed: 0);
            ITransformer loadedModel = _mlContext.Model.Load(_modelPath, out var modelInputSchema);
            _predEngine = _mlContext.Model.CreatePredictionEngine<Diseases, PredictionDiseases>(loadedModel);

            var prediction = _predEngine.Predict(diseases);

            return GetScoresWithLabelsSorted(_predEngine.OutputSchema, "Score", prediction.Score)
                .Take(10)
                .Where(x => x.Value >= 0.2)
                .ToDictionary(i => i.Key, i => i.Value);
        }
        private static void SaveModelAsFile(MLContext mlContext, DataViewSchema trainingDataViewSchema, ITransformer trainedModel)
        {
            mlContext.Model.Save(trainedModel, trainingDataViewSchema, _modelPath);
        }

        public static List<string> PredictDiseases(IEnumerable<Diseases> list)
        {
            ITransformer loadedModel = _mlContext.Model.Load(_modelPath, out var modelInputSchema);
            _predEngine = _mlContext.Model.CreatePredictionEngine<Diseases, PredictionDiseases>(loadedModel);

            IDataView batchDiseases = _mlContext.Data.LoadFromEnumerable<Diseases>(list);

            var predictedResults = _mlContext.Data.CreateEnumerable<Diseases>(batchDiseases, reuseRowObject: false);

            List<string> translatedDiseases = new List<string>();

            foreach (Diseases prediction in predictedResults)
            {
                translatedDiseases.Add(_predEngine.Predict(prediction).Disease);
            }

            return translatedDiseases;
        }

        private static Dictionary<string, float> GetScoresWithLabelsSorted(DataViewSchema schema, string name, float[] scores)
        {
            Dictionary<string, float> result = new Dictionary<string, float>();

            var column = schema.GetColumnOrNull(name);

            var slotNames = new VBuffer<ReadOnlyMemory<char>>();
            column.Value.GetSlotNames(ref slotNames);
            var names = new string[slotNames.Length];
            var num = 0;

            foreach (var denseValue in slotNames.DenseValues())
            {
                result.Add(denseValue.ToString(), scores[num++]);
            }

            return result.OrderByDescending(c => c.Value).ToDictionary(i => i.Key, i => i.Value);
        }

    }
}

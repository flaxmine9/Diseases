using Classification;
using CSV;
using System;
using System.Collections.Generic;
using System.IO;
using TranslateYandex;

namespace Disease
{
    class Program
    {
        private static string trainPath = Path.Combine(Environment.CurrentDirectory, "DataSet", "TrainDataSet.csv");
        private static string pathDataSetRussian = Path.Combine(Environment.CurrentDirectory, "DataSet", "DatasetRussian.csv");
        static void Main(string[] args)
        {
            Csv csv = new Csv();
            Translate translate = new Translate();

            //var csvDataSet = csv.ReadCsv(pathDataSetRussian);

            //var translatedText = translate.TranslateDataSet(csvDataSet);
            //var resultShuffle = csv.GetShuffleSymptoms(csvDataSet);

            //csv.Write(pathDataSetRussian, resultShuffle);

            var dataSet = NeuralNetwork.LoadData(trainPath);
            var pipeLine = NeuralNetwork.ProcessData();
            NeuralNetwork.BuildAndTrainModel(dataSet, pipeLine, 0.2);
            NeuralNetwork.Evaluate(dataSet.Schema);

            Diseases dis = new Diseases()
            {
                Sym1 = translate.TranslateText("головная боль", "ru-en"),
                Sym2 = translate.TranslateText("боль в плечах", "ru-en"),
                Sym3 = translate.TranslateText("потеря сознания", "ru-en"),
                Sym4 = translate.TranslateText("повышенное давление", "ru-en"),
                Sym5 = translate.TranslateText("потливость", "ru-en"),
                Sym6 = translate.TranslateText("горечь во рту", "ru-en")
            };

            Dictionary<string, float> result = NeuralNetwork.PredictDisease(dis);

            foreach (KeyValuePair<string, float> item in result)
            {
                Console.WriteLine($"Disease: {translate.TranslateText(item.Key)} Score: {item.Value * 100}%");
            }
        }
    }
}

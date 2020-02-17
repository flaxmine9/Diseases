using Classification;
using CSV;
using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.IO;
using Yandex;

namespace Disease
{
    class Program
    {
        private static string trainPath = Path.Combine(Environment.CurrentDirectory, "DataSet", "TrainDataSet2.csv");
        private static string pathDataSetRussian = Path.Combine(Environment.CurrentDirectory, "DataSet", "DatasetRussian2.csv");
        static void Main(string[] args)
        {
            Csv csv = new Csv();
            Translate translate = new Translate();

            //var csvDataSet = csv.ReadCsv(pathDataSetRussian);

            //var translatedText = translate.TranslateDataSet(csvDataSet);
            //var resultShuffle = csv.GetShuffleSymptoms(translatedText);

            //csv.Write(@"C:\Users\Dima\Desktop\TrainDataSet2.csv", resultShuffle);

            IDataView dataSet = NeuralNetwork.LoadData(trainPath);
            IEstimator<ITransformer> pipeLine = NeuralNetwork.ProcessData();

            NeuralNetwork.BuildAndTrainModel(dataSet, pipeLine, 0.2);
            NeuralNetwork.Evaluate(dataSet.Schema);

            Diseases dis = new Diseases()
            {
                Sym1 = translate.TranslateText("жгучая боль во время мочеиспускания", "ru-en"),
                Sym2 = translate.TranslateText("учащенное сердцебиение", "ru-en"),
                Sym3 = translate.TranslateText("повышение артериального давления", "ru-en"),
                Sym4 = translate.TranslateText("повышенное давление", "ru-en"),
                Sym5 = translate.TranslateText("Обезвоживание", "ru-en"),
                Sym6 = translate.TranslateText("Лихорадка", "ru-en"),
                Sex = translate.TranslateText("мужской", "ru-en"),
                Age = "3"
            };

            Dictionary<string, float> result = NeuralNetwork.PredictDisease(dis);

            foreach (KeyValuePair<string, float> item in result)
            {
                Console.WriteLine($"Disease: {translate.TranslateText(item.Key)} Score: {item.Value * 100}%");
            }
        }
    }
}

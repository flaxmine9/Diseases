using Classification;
using CSV;
using System;
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
            Console.WriteLine(DateTime.Now.ToString());
            Console.WriteLine($"*************************************************************************************************************");

            Csv csv = new Csv();
            Translate translate = new Translate();

            //var csvDataSet = csv.ReadCsv(pathDataSetRussian);

            //var translatedText = translate.TranslateDataSet(csvDataSet);
            //var resultShuffle = csv.GetShuffleSymptoms(translatedText);

            //csv.Write(trainPath, resultShuffle);

            var dataSet = NeuralNetwork.LoadData(trainPath);
            //var pipeLine = NeuralNetwork.ProcessData();
            //NeuralNetwork.BuildAndTrainModel(dataSet, pipeLine, 0.2);
            //NeuralNetwork.Evaluate(dataSet.Schema);

            NeuralNetwork.PredictDiseases();


            Console.WriteLine($"*************************************************************************************************************");

            NeuralNetwork.PredictDisease(new Diseases()
            {
                Sym1 = "numbness of fingers",
                Sym2 = "shoulder pain",
                Sym3 = "pain at sharp movements",
                Sym4 = "Pain behind the sternum",
                Sym5 = "diarrhea",
                Sym6 = "constipation"
            });

            Console.WriteLine(DateTime.Now.ToString());
            Console.WriteLine($"*************************************************************************************************************");
        }
    }
}

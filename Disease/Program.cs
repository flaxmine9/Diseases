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
            var pipeLine = NeuralNetwork.ProcessData();
            NeuralNetwork.BuildAndTrainModel(dataSet, pipeLine, 0.3);
            NeuralNetwork.Evaluate(dataSet.Schema);

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

            var predictions = NeuralNetwork.PredictDiseases(diseases);
            for (int i = 0; i < predictions.Count; i++)
            {
                Console.WriteLine($"{i + 1}) " + translate.TranslateText(predictions[i], "en-ru"));
            }
            
            Console.WriteLine(DateTime.Now.ToString());
            Console.WriteLine($"*************************************************************************************************************");
        }
    }
}

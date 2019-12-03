using Classification;
using CsvHelper;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace CSV
{
    public class Csv
    {
        public Csv() { }

        public List<Diseases> ReadCsv(string path)
        {
            Encoding.RegisterProvider(CodePagesEncodingProvider.Instance);

            using (StreamReader streamReader = new StreamReader(path, Encoding.GetEncoding(1251)))
            {             
                using (CsvReader csvReader = new CsvReader(streamReader))
                {
                    csvReader.Configuration.Delimiter = ";";
                    return csvReader.GetRecords<Diseases>().ToList();
                }
            }
        }
        public void Write(string path, List<Diseases> dataSet)
        {
            using (StreamWriter streamWriter = new StreamWriter(new FileStream(path, FileMode.Create), Encoding.Default))
            {
                using (CsvWriter csvWriter = new CsvWriter(streamWriter))
                {
                    csvWriter.Configuration.Delimiter = ";";
                    csvWriter.WriteRecords(dataSet);
                }
            }
        }
        public List<Diseases> GetShuffleSymptoms(List<Diseases> list)
        {
            Random random = new Random();

            List<Diseases> shuffleDiseases = new List<Diseases>();

            for (int i = 0; i < list.Count; i++)
            {
                List<string> lst = new List<string>()
                {
                    list[i].Disease,
                    list[i].Sym1,
                    list[i].Sym2,
                    list[i].Sym3,
                    list[i].Sym4,
                    list[i].Sym5,
                    list[i].Sym6,
                };

                for (int j = 0; j < 8; j++)
                {
                    var resultShuffle = Shuffle<string>(lst.Skip(1).ToList(), random);
                    resultShuffle.Insert(0, list[i].Disease);

                    Diseases diseases = new Diseases()
                    {
                        Disease = resultShuffle[0],
                        Sym1 = resultShuffle[1],
                        Sym2 = resultShuffle[2],
                        Sym3 = resultShuffle[3],
                        Sym4 = resultShuffle[4],
                        Sym5 = resultShuffle[5],
                        Sym6 = resultShuffle[6],
                    };
                    shuffleDiseases.Add(diseases);
                }
            }
            return shuffleDiseases;
        }
        private List<T> Shuffle<T>(List<T> list, Random random)
        {
            List<T> lst = list;

            for (int i = lst.Count - 1; i >= 1; i--)
            {
                int j = random.Next(i + 1);

                T tmp = lst[j];
                lst[j] = lst[i];
                lst[i] = tmp;
            }
            return lst;
        }
    }
}

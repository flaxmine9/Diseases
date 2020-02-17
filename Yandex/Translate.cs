using Classification;
using Newtonsoft.Json;
using System.Collections.Generic;
using System.IO;
using System.Net;

namespace Yandex
{
    public class Translate
    {
        private string URL { get; set; } = "https://translate.yandex.net/api/v1.5/tr.json/translate";
        private string Key { get; set; } = "trnsl.1.1.20191130T203841Z.4dbd550bf91ef3ae.d9f39c64fa35f4141e265ee33781bdacb9e16973";

        public Translate() { }

        public string TranslateText(string text, string lang = "en-ru")
        {
            if (text != "")
            {
                string requestString = $"{URL}?key={Key}&text={text}&lang={lang}&format=plain&options=1";

                WebRequest request = WebRequest.Create(requestString);
                WebResponse response = request.GetResponse();

                Stream stream = response.GetResponseStream();
                StreamReader streamReader = new StreamReader(stream);

                string resultTranslate = streamReader.ReadToEnd();

                Text text1 = JsonConvert.DeserializeObject<Text>(resultTranslate);
                return text1.text[0];
            }
            return "";
        }
        public List<Diseases> TranslateDataSet(List<Diseases> list)
        {
            List<Diseases> translatedDiseases = new List<Diseases>();

            for (int i = 0; i < list.Count; i++)
            {
                Diseases diseases = new Diseases()
                {
                    Disease = TranslateText(list[i].Disease, "ru-en"),
                    Sym1 = TranslateText(list[i].Sym1, "ru-en"),
                    Sym2 = TranslateText(list[i].Sym2, "ru-en"),
                    Sym3 = TranslateText(list[i].Sym3, "ru-en"),
                    Sym4 = TranslateText(list[i].Sym4, "ru-en"),
                    Sym5 = TranslateText(list[i].Sym5, "ru-en"),
                    Sym6 = TranslateText(list[i].Sym6, "ru-en"),
                    Sex = TranslateText(list[i].Sex, "ru-en"),
                    Age = list[i].Age
                };
                translatedDiseases.Add(diseases);
            }
            return translatedDiseases;
        }
    }
}

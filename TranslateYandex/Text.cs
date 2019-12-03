using System.Collections.Generic;

namespace TranslateYandex
{
    public class Text
    {
        public int Code { get; set; }
        public Detected Detected { get; set; }
        public string Lang { get; set; }
        public List<string> text { get; set; }

        public Text() { }
    }
}

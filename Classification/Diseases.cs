using Microsoft.ML.Data;

namespace Classification
{
    public class Diseases
    {
        [LoadColumn(0)]
        public string Disease { get; set; }

        [LoadColumn(1)]
        public string Sym1 { get; set; }

        [LoadColumn(2)]
        public string Sym2 { get; set; }

        [LoadColumn(3)]
        public string Sym3 { get; set; }

        [LoadColumn(4)]
        public string Sym4 { get; set; }

        [LoadColumn(5)]
        public string Sym5 { get; set; }

        [LoadColumn(6)]
        public string Sym6 { get; set; }

        [LoadColumn(7)]
        public string Sex { get; set; }

        [LoadColumn(8)]
        public float Age { get; set; }
    }
}

using Microsoft.ML.Data;

namespace Classification
{
    public class PredictionDiseases
    {
        [ColumnName("PredictedLabel")]
        public string Disease { get; set; }
    }
}

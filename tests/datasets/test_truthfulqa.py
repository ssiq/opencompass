import unittest
from unittest.mock import patch

from opencompass.datasets.truthfulqa import TruthfulQAEvaluator


class TestTruthfulQAEvaluator(unittest.TestCase):

    def test_normalize_string_metric(self):
        self.assertEqual(TruthfulQAEvaluator._normalize_metrics('bleu'),
                         ['bleu'])

    def test_init_supports_tuple_basic_metrics(self):
        evaluator = TruthfulQAEvaluator(metrics=('bleu', 'rouge', 'bleurt'))

        self.assertEqual(evaluator.metrics, ['bleu', 'rouge', 'bleurt'])
        self.assertEqual(evaluator.api_metrics, [])

    @patch('opencompass.datasets.truthfulqa.evaluate.load')
    def test_bleurt_load_error_is_actionable(self, mock_load):
        mock_load.side_effect = FileNotFoundError('bleurt is unavailable')

        with self.assertRaisesRegex(ImportError, 'Google BLEURT package'):
            TruthfulQAEvaluator._load_basic_metric('bleurt')


if __name__ == '__main__':
    unittest.main()

import unittest

from opencompass.datasets.gsm8k import gsm8k_postprocess


class TestGsm8kPostprocess(unittest.TestCase):

    def test_prefers_answer_line_before_code_noise(self):
        text = (
            'The profit is $200,000 - $130,000 = $70,000.\n'
            'The answer is 70000\n'
            '```python\n'
            'profit = 70000\n'
            '```\n'
            '```output\n'
            '7')

        self.assertEqual(gsm8k_postprocess(text), '70000')

    def test_normalizes_comma_separated_answer(self):
        self.assertEqual(gsm8k_postprocess('Final answer: $70,000.'),
                         '70000')

    def test_keeps_last_number_fallback(self):
        self.assertEqual(gsm8k_postprocess('compute 3.75 then round up to 4'),
                         '4')


if __name__ == '__main__':
    unittest.main()

import unittest

from opencompass.datasets.gpqa import GPQA_Simple_Eval_postprocess


class TestGPQAPostprocess(unittest.TestCase):

    def test_gpqa_simple_eval_postprocess_uses_last_answer_match(self):
        text = 'I first thought ANSWER: A, but the final answer is ANSWER: C'
        self.assertEqual(GPQA_Simple_Eval_postprocess(text), 'C')

    def test_gpqa_simple_eval_postprocess_avoids_partial_answer_prefix(self):
        self.assertEqual(GPQA_Simple_Eval_postprocess("I'll answer: ANSWER: C"),
                         'C')


if __name__ == '__main__':
    unittest.main()

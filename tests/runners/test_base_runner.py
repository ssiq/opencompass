import unittest

from opencompass.runners.base import BaseRunner


class DummyRunner(BaseRunner):

    def __init__(self, status):
        super().__init__(task=dict(type='DummyTask'))
        self.status = status

    def launch(self, tasks):
        return self.status


class TestBaseRunner(unittest.TestCase):

    def test_runner_call_returns_failed_task_count(self):
        runner = DummyRunner([('task_a', 0), ('task_b', 1), ('task_c', 2)])

        self.assertEqual(runner([]), 2)

    def test_runner_call_returns_zero_when_all_tasks_succeed(self):
        runner = DummyRunner([('task_a', 0), ('task_b', 0)])

        self.assertEqual(runner([]), 0)


if __name__ == '__main__':
    unittest.main()

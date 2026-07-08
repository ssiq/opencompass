"""Tests for MedBench dataset configs."""


def test_medbench_config_imports_smdoc_evaluator():
    """Test MedBench config can resolve the SMDoc evaluator."""
    from opencompass.configs.datasets.MedBench.medbench_gen_0b4fff import (
        medbench_datasets,
    )

    smdoc_datasets = [
        dataset for dataset in medbench_datasets if dataset['name'] == 'SMDoc'
    ]

    assert len(medbench_datasets) == 20
    assert len(smdoc_datasets) == 1
    assert smdoc_datasets[0]['eval_cfg']['evaluator']['type'].__name__ == (
        'MedBenchEvaluator_SMDoc'
    )

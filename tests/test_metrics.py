from assetmind.evaluation.metrics import rmse, nasa_score


def test_perfect_prediction_scores_zero():
    assert rmse([10, 20], [10, 20]) == 0
    assert nasa_score([10, 20], [10, 20]) == 0


def test_late_prediction_penalized_more_than_early():
    late = nasa_score([50], [60])    # predicted 10 cycles too many
    early = nasa_score([50], [40])   # predicted 10 cycles too few
    assert late > early
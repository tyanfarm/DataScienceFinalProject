import numpy as np 
import bentoml
from bentoml.io import NumpyNdarray

universities_rank_neural_network_runner = bentoml.sklearn.get("universities_rank_regression:latest").to_runner()
svc = bentoml.Service("university_rank",runners=[universities_rank_neural_network_runner])

@svc.api(input=NumpyNdarray(),output=NumpyNdarray())
def predict_rank(input_series: np.ndarray) -> np.ndarray:
    result = universities_rank_neural_network_runner.predict.run(input_series)
    return result
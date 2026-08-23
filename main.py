from predictor import Predictor


p = Predictor.from_model_json(
    "model_params.json"
)


for peak in [6,8,10,12,14]:

    prob = p.predict_probability(
        peak,
        214
    )

    print(
        peak,
        prob
    )
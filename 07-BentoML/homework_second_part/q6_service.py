import bentoml
from bentoml.io import NumpyNdarray

model_ref = bentoml.sklearn.get("mlzoomcamp_homework:jsi67fslz6txydu5")
#dv = model_ref.custom_objects['dictVectorizer']

model_runner = model_ref.to_runner()

svc = bentoml.Service("mlzoomcamp_homework", runners=[model_runner])


@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
async def regress(application_data):
    #vector = dv.transform(application_data)
    prediction = await model_runner.predict.async_run(application_data)
    print(prediction)
    
    return prediction
        
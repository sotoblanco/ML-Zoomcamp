import requests


#url = 'http://localhost:8080/2015-03-31/functions/function/invocations'
#url = "https://w9oqgv678c.execute-api.us-west-2.amazonaws.com/test/predict"
url = 'https://tpxtwbbm2oqkfpyqpj64orwspa0iwwnc.lambda-url.us-west-2.on.aws/predict'

data = {"url": "https://upload.wikimedia.org/wikipedia/en/e/e9/GodzillaEncounterModel.jpg"}

result = requests.post(url, json=data).json()

print(result)

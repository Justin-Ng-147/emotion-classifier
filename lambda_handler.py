import json
import logging
from inference_onnx import ONNXPredictor

print(f"Loading the model")
inferencing_instance = ONNXPredictor("./models/model.onnx")


def lambda_handler(event, context):
	"""
	Lambda function handler for predicting linguistic acceptability of the given sentence
	"""
	
	if "resource" in event.keys():
		body = event["body"]
		body = json.loads(body)

		response = inferencing_instance.predict(body)
		return {
			"statusCode": 200,
			"headers": {},
			"body": json.dumps(response)
		}
	else:
		response = inferencing_instance.predict(event)
		return response

if __name__ == "__main__":
	test = {"sentence": "this is a sample sentence"}
	lambda_handler(test, None)
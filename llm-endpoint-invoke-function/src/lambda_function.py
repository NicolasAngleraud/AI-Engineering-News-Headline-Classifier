import json
import boto3
from botocore.exceptions import BotoCoreError, ClientError

def lambda_handler(event, context):
    try:
        
        sagemaker_runtime = boto3.client('sagemaker-runtime')
        
        try:
            body = json.loads(event['body'])
        except (json.JSONDecodeError, KeyError):
            return {
                'statusCode': 400,
                'body': json.dumps({"error": "Invalid JSON format or missing 'body' in request."})
            }

        try:
            headline = body['query']['headline']
        except KeyError:
            return {
                'statusCode': 400,
                'body': json.dumps({"error": "Missing 'query' or 'headline' in request body."})
            }

        endpoint_name = 'multiclass-text-classification-endpoint-v1'

        payload = json.dumps({"inputs": headline})

        try:
            response = sagemaker_runtime.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType="application/json",
                Body=payload
            )
        except (BotoCoreError, ClientError) as e:
            return {
                'statusCode': 500,
                'body': json.dumps({"error": f"AWS SageMaker error: {str(e)}"})
            }

        try:
            result = json.loads(response['Body'].read().decode())
        except json.JSONDecodeError:
            return {
                'statusCode': 500,
                'body': json.dumps({"error": "Failed to decode response from SageMaker."})
            }

        return {
            'statusCode': 200,
            'body': json.dumps(result)
        }

    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({"error": f"An unexpected error occurred: {str(e)}"})
        }

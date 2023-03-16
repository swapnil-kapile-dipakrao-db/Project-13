# Project-13

## Call an Amazon SageMaker model endpoint using Amazon API Gateway and AWS Lambda

**By Archana Sawant, Swapnil Kapile, Sachin Pharande, Snehal Chaudhary, Rohit Patange** | on 17 March 2023 | in Amazon SageMaker, Machin Learning | Permalink |  Comments |  Share

At AWS Machine Learning (ML) workshops, customers often ask, “After I deploy an endpoint, where do I go from there?” You can deploy an **Amazon SageMaker** trained and validated ML model as an online endpoint in production. Alternatively, you can choose which SageMaker functionality to use. For example, you can choose just to train a model or to host one. Whether you choose one SageMaker function.

The following diagram shows how the deployed model is called using serverless architecture. Starting from the client side, a client script calls an **Amazon API** Gateway API action and passes parameter values. API Gateway is a layer that provides the API to the client. In addition, it seals the backend so that  **AWS Lambda** stays and runs in a protected private network. API Gateway passes the parameter values to the Lambda function. The Lambda function parses the value and sends it to the SageMaker model endpoint. The model performs the prediction and returns the predicted value to Lambda. The Lambda function parses the returned value and sends it back to API Gateway. API Gateway responds to the client with that value.

<img src="https://user-images.githubusercontent.com/117730243/225567787-497e9a77-42bb-44fb-96d3-bb1dd5075414.png">

In this post, I show you how to invoke a model endpoint deployed by SageMaker using API Gateway and Lambda. For testing purposes, we use Postman.


### Breast cancer model notebook

We use a sample notebook provided by SageMaker called Breast Cancer Prediction.ipynb. You have access to this notebook on the SageMaker Examples tab.

Choosing Use creates a folder and loads the notebook. The breast cancer prediction model predicts whether a breast mass is a malignant tumor or benign by looking at features computed from a digitized image of a fine needle aspirate of a breast mass. The data used to train the model consists of the diagnosis as well as the 10 real-valued features that are computed for each cell nucleus (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension). The prediction returned by the model is either 0 or 1; 0 being benign and 1 being malignant tumor. The Lambda function converts this value to be either B for benign or M for malignant tumor.

SageMaker has managed built-in Jupyter notebooks that allow you to write code in Python or R to explore, analyze, and do some modeling with small set of data. The sample Jupyter notebooks get loaded onto a notebook instance when the notebook instance boots up. Each sample notebook consists of markdown-based comments that explain each step, from downloading training data, performing the training, to deploying a model endpoint. After the model is trained and deployed, you can invoke the model endpoint using the SageMaker runtime API. To make it free from server and infrastructure management, we encapsulate this invocation using API Gateway and Lambda.

### Create a SageMaker model endpoint

To create your model endpoint, complete the following steps:

1.	Open the __Breast Cancer Prediction.ipynb__ sample notebook.

2.	Comment out the last cell by inserting #, because it deletes the endpoint created in the previous cell.

<img src="https://user-images.githubusercontent.com/117730243/225568683-37fc5e52-701d-404d-be55-08790c0867c1.png">

3.	Run the entire notebook by choosing Run All on the Cell

Alternatively, you can run each cell one by one by pressing Shift + Enter. If you run each cell, you can learn what each step is doing. For the purpose of this post, I chose Run All to deploy the model as an endpoint after the model training is complete.

<img src="https://user-images.githubusercontent.com/117730243/225569334-54f3ad56-aa77-4281-bc12-f2ad0a0193da.png">

Upon creation, you can view this endpoint on the SageMaker console. The default endpoint name looks like linear-endpoint-201803211721, but you can make it more meaningful. I called mine __linear-learner-breast-cancer-prediction-endpoint.__

#### Create a Lambda function that calls the SageMaker runtime invoke_endpoint

Now we have a SageMaker model endpoint. Let’s look at how we call it from Lambda. We use the SageMaker runtime API action and the Boto3 **sagemaker-runtime.invoke_endpoint().**

1.	On the Lambda console, on the Functions page, choose Create function.
2.	For Function name, enter a name.
3.	For Runtime¸ choose your runtime.
4.	For Execution role¸ select Create a new role or Use an existing role.

<img src="https://user-images.githubusercontent.com/117730243/225570095-ff7a29e8-e902-48ba-86b1-952352b9a8d2.png">

If you chose Create a new role, after the Lambda function is created, go to the Configuration tab and find the name of the IAM role created. Click on the role name which will take you to IAM console.

<img src="https://user-images.githubusercontent.com/117730243/225570429-52ceb8c7-45f1-42c9-828e-757573b7fd40.png">

5.	Whether you created a new role or using the existing role, make sure to include the following policy, which gives your function permission to invoke a model endpoint:

{
    "Version": "2012-10-17",
    
    "Statement": [
    
        {
            "Sid": "VisualEditor0",
            
            "Effect": "Allow",
            
            "Action": "sagemaker:InvokeEndpoint",
            
            "Resource": "*"
            
        }
        
    ]
}

<img src="https://user-images.githubusercontent.com/117730243/225570935-2ffc977a-0fc7-49d6-a361-f1ad09aa0601.png">

The following is the sample Lambda function code:

import os
import io
import boto3
import json
import csv

# grab environment variables
ENDPOINT_NAME = os.environ['ENDPOINT_NAME']
runtime= boto3.client('runtime.sagemaker')

def lambda_handler(event, context):
    print("Received event: " + json.dumps(event, indent=2))
    
    data = json.loads(json.dumps(event))
    payload = data['data']
    print(payload)
    
    response = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME,
                                       ContentType='text/csv',
                                       Body=payload)
    print(response)
    result = json.loads(response['Body'].read().decode())
    print(result)
    pred = int(result['predictions'][0]['score'])
    predicted_label = 'M' if pred == 1 else 'B'
    
    return predicted_label

**ENDPOINT_NAME** is an environment variable that holds the name of the SageMaker model endpoint you just deployed using the sample notebook. Go to the SageMaker console to find the end point name generated by SageMaker. Enter the name as the environment variable value. It would look like DEMO-linear-endpoint-xxxxxxxxx.

<img src="https://user-images.githubusercontent.com/117730243/225571489-c8ff0431-2322-4f9b-ab1e-2a83f6138121.png">

<img src="https://user-images.githubusercontent.com/117730243/225571825-d9b6a044-e688-4a5f-9c9e-a419ffcd9473.png">

The event that invokes the Lambda function is triggered by API Gateway. API Gateway simply passes the test data through an event.

#### Create a REST API: Integration request setup

You can create an API by following these steps:

1.	On the API Gateway console, choose the REST API
2.	Choose Build.
<img src="https://user-images.githubusercontent.com/117730243/225572323-309cded1-427a-426b-8536-50ca7b4b4b0e.png">

3.	Select New API.
4.	For API name¸ enter a name (for example, BreastCancerPredition).
5.	Leave Endpoint Type as Regional.
6.	Choose Create API.

<img src="https://user-images.githubusercontent.com/117730243/225572609-f8020a00-846a-4b65-bc91-f34cd799c1dc.png">

<img src="https://user-images.githubusercontent.com/117730243/225572865-64991d97-8bda-4d57-975c-64bce84a8ec8.png">

7.	On the Actions menu, choose Create resource.

8.	Enter a name for the resource (for example, predictbreastcancer).

9.	After the resource is created, on the Actions menu, choose Create Method to create a POST method.

<img src="https://user-images.githubusercontent.com/117730243/225573261-427025b1-b560-4289-b0ab-b7c1912599e2.png">

10.	For Integration type, select Lambda Function.

<img src="https://user-images.githubusercontent.com/117730243/225573888-9ebaf98a-2eb5-400c-82e4-2ed085911f2e.png">

11.	For Lambda function, enter the function you created.
When the setup is complete, you can deploy the API to a stage.
12.	On the Actions menu, choose Deploy API.
13.	Create a new stage called test.
14.	Choose Deploy.

<img src="https://user-images.githubusercontent.com/117730243/225574538-5e2d5648-8abd-4bc4-9aad-7ba5c3adafae.png>

This step gives you the invoke URL.

For more information on creating an API with API Gateway, see Creating a REST API in Amazon API Gateway. In addition, you can make the API more secure using various methods.

Now that you have an API and a Lambda function in place, let’s look at the test data.

#### Test data

When the sample notebook loads the dataset to the __Amazon Simple Storage Service___ (Amazon S3) bucket that you specified, CSV files with data are loaded. The sample notebook separates the dataset into two files: training data and validation data. Each file is saved in its respective folder. The Amazon S3 path to the file looks like <your bucket name>/sagemaker/DEMO-breast-cancer-prediction/validation.

<img src="https://user-images.githubusercontent.com/117730243/225574989-d653dc53-267f-4d82-908f-a7fa649cd827.png">

The following code is one row of the validation data from the file linear_validation.data in the validation folder:

{“data”:”13.49,22.3,86.91,561.0,0.08752,0.07697999999999999,0.047510000000000004,0.033839999999999995,0.1809,0.057179999999999995,0.2338,1.3530000000000002,1.735,20.2,0.004455,0.013819999999999999,0.02095,0.01184,0.01641,0.001956,15.15,31.82,99.0,698.8,0.1162,0.1711,0.2282,0.1282,0.2871,0.06917000000000001″}

#### Test with Postman

Now that we have the Lambda function, REST API, and test data, let’s test it using Postman, which is an HTTP client for testing web services. Make sure to download the latest version.

When you deployed your API, it provided the invoke URL, which looks like
https://{restapi_id}.execute-api.us-west-2.amazonaws.com/prod/predictbreastcancer.
It follows the format

https://{restapi_id}.execute-api.{region}.amazonaws.com/{stage_name}/{resource_name}.

For more information about invoking an API in API Gateway, see **Invoking a REST API** in __Amazon API Gateway.___

1.	Enter the invoke URL into Postman.
2.	Choose POST as method.
3.	On the Body tab, enter the test data.
4.	Choose Send to see the returned result as B for the row of test data we looked at earlier.

<img src="https://user-images.githubusercontent.com/117730243/225575644-ca9ba76d-68ff-4f33-872f-edb07959d4c4.png">

### Resources

To learn more about SageMaker Inference, please refer to the following resources:

•	SageMaker Inference documentation
•	SageMaker Inference recommender
•	SageMaker Serverless Inference
•	SageMaker Asynchronous Inference
•	Inference endpoint testing from studio
•	Roundup of re:Invent 2021 Amazon SageMaker announcements

### Conclusion

In this post, you created a model endpoint deployed and hosted by SageMaker. Then you created serverless components (a REST API and Lambda function) that invoke the endpoint. Now you know how to call an ML model endpoint hosted by SageMaker using serverless technology.
If you have feedback about this post, please leave it in the comments. If you have questions about implementing the example used in this post, you can also open a thread on the Developer Tools forum.
________________________________________


















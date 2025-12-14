# Deploying codellama-7b-qlora-finetune to AWS SageMaker Endpoint

**Model:** https://huggingface.co/HeyHey12Hey/codellama-7b-qlora-finetune

## Prerequisites

1. **AWS Account** with SageMaker access and appropriate IAM permissions
2. **HuggingFace Account** (optional - model is public)
3. **AWS CLI** configured with credentials
4. **Python 3.8+** with boto3 and sagemaker SDK installed

---

## Step-by-Step Process

### Step 1: Install Required Dependencies

```bash
pip install sagemaker boto3 huggingface_hub
```

### Step 2: Verify Model Access

1. Visit https://huggingface.co/HeyHey12Hey/codellama-7b-qlora-finetune
2. Verify the model files are accessible (this is a public model, no license required)
3. (Optional) Generate an access token at https://huggingface.co/settings/tokens if you encounter rate limits

### Step 3: Set Up AWS SageMaker Session

```python
import sagemaker
import boto3
from sagemaker.huggingface import HuggingFaceModel, get_huggingface_llm_image_uri

# Initialize session
sess = sagemaker.Session()
role = sagemaker.get_execution_role()  # Or specify your IAM role ARN
region = sess.boto_region_name
```

### Step 4: Configure the HuggingFace LLM Model

```python
# Get the HuggingFace LLM Deep Learning Container URI
llm_image = get_huggingface_llm_image_uri(
    backend="huggingface",  # or "tgi" for Text Generation Inference
    region=region
)

# Model configuration
hub_config = {
    'HF_MODEL_ID': 'HeyHey12Hey/codellama-7b-qlora-finetune',
    'SM_NUM_GPUS': '1',  # Number of GPUs
    'MAX_INPUT_LENGTH': '2048',
    'MAX_TOTAL_TOKENS': '4096',
    # 'HF_TOKEN': '<YOUR_HUGGINGFACE_TOKEN>',  # Optional - only if rate limited
}

# Create HuggingFace Model
huggingface_model = HuggingFaceModel(
    image_uri=llm_image,
    env=hub_config,
    role=role,
)
```

### Step 5: Deploy the Model to an Endpoint

```python
# Deploy to SageMaker endpoint
# CodeLlama-7b can run on ml.g5.xlarge (1 GPU, 24GB VRAM) or ml.g5.2xlarge
# 7B models are smaller and more cost-effective than 13B

predictor = huggingface_model.deploy(
    initial_instance_count=1,
    instance_type="ml.g5.xlarge",  # GPU instance - sufficient for 7B model
    endpoint_name="codellama-7b-qlora-endpoint",  # Optional custom name
)
```

**Note:** Deployment typically takes 5-10 minutes as SageMaker downloads the model weights (~14GB for 7B model).

### Step 6: Test the Endpoint

```python
# Test inference
response = predictor.predict({
    "inputs": "def fibonacci(n):",
    "parameters": {
        "max_new_tokens": 256,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True,
    }
})

print(response)
```

### Step 7: Invoke from External Applications (Optional)

```python
import boto3
import json

# Using boto3 runtime client
runtime = boto3.client('sagemaker-runtime')

response = runtime.invoke_endpoint(
    EndpointName='codellama-7b-qlora-endpoint',
    ContentType='application/json',
    Body=json.dumps({
        "inputs": "def quicksort(arr):",
        "parameters": {
            "max_new_tokens": 512,
            "temperature": 0.7,
        }
    })
)

result = json.loads(response['Body'].read().decode())
print(result)
```

### Step 8: Clean Up (When Done)

```python
# Delete endpoint to stop incurring charges
predictor.delete_endpoint()

# Optionally delete the model
predictor.delete_model()
```

---

## Instance Recommendations (for 7B model)

| Instance Type | GPUs | GPU Memory | Cost (approx) | Notes |
|---------------|------|------------|---------------|-------|
| ml.g5.xlarge | 1 | 24 GB | ~$1.00/hr | Recommended for 7B model |
| ml.g5.2xlarge | 1 | 24 GB | ~$1.50/hr | Better CPU/memory headroom |
| ml.g5.4xlarge | 1 | 24 GB | ~$2.00/hr | Higher throughput |

---

## Alternative: Using Text Generation Inference (TGI)

For better performance, use the TGI backend:

```python
llm_image = get_huggingface_llm_image_uri(
    backend="tgi",
    region=region,
    version="1.4.0"  # Specify TGI version
)

hub_config = {
    'HF_MODEL_ID': 'HeyHey12Hey/codellama-7b-qlora-finetune',
    'SM_NUM_GPUS': '1',
    'MAX_INPUT_LENGTH': '2048',
    'MAX_TOTAL_TOKENS': '4096',
    # 'HF_TOKEN': '<YOUR_HUGGINGFACE_TOKEN>',  # Optional
}
```

---

## Troubleshooting

1. **Model download timeout**: Increase `container_startup_health_check_timeout` in deploy()
2. **Out of memory**: Use a larger instance (ml.g5.2xlarge) or enable quantization
3. **Rate limited**: Add HF_TOKEN to environment config
4. **Slow inference**: Consider TGI backend or larger instance

---

## Cost Considerations

- **Endpoint costs**: Charged per hour while running (~$1.00/hr for g5.xlarge)
- **Storage**: Model artifacts stored in S3 (minimal cost)
- **Data transfer**: Ingress free, egress charged

**Tip**: Use SageMaker Serverless Inference or Asynchronous Inference for cost optimization if real-time isn't required.

## Note on QLora Fine-tuned Models

This model (`HeyHey12Hey/codellama-7b-qlora-finetune`) is a QLora fine-tuned version of CodeLlama-7B. The adapter weights have been merged into the base model, so it deploys like a standard model. If you encounter issues with merged weights, you may need to use the `peft` library to load adapter weights separately.
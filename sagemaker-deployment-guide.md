
**Model:** https://huggingface.co/HeyHey12Hey/codellama-7b-qlora-finetune

## Prerequisites

1. **AWS Account** with SageMaker access and appropriate IAM permissions
2. **HuggingFace Account** (optional - model is public)
3. **AWS CLI** configured with credentials
4. **Python 3.8+** with boto3 and sagemaker SDK installed

---

## Deployment Timing in SageMaker Notebook

| Step | Duration | Notes |
|------|----------|-------|
| Import & Setup | ~1-2 seconds | Instant |
| Configure Model | ~1-2 seconds | Instant |
| **Deploy Endpoint** | **5-15 minutes** | Downloads model, provisions instance |
| Test Inference | ~2-10 seconds | First call may be slower (cold start) |

**Total: ~5-15 minutes** (mostly waiting for deployment)

---

## Step-by-Step Process with Error Handling

### Step 1: Install Required Dependencies

```bash
pip install sagemaker boto3 huggingface_hub
```

Note: If running in a SageMaker notebook, these are already pre-installed.

### Step 2: Verify Model Access

1. Visit https://huggingface.co/HeyHey12Hey/codellama-7b-qlora-finetune
2. Verify the model files are accessible (this is a public model, no license required)
3. (Optional) Generate an access token at https://huggingface.co/settings/tokens if you encounter rate limits

### Step 3: Set Up AWS SageMaker Session

```python
import sagemaker
import boto3
import json
import time
from botocore.exceptions import ClientError, EndpointConnectionError

# Setup with Error Handling
try:
    sess = sagemaker.Session()
    role = sagemaker.get_execution_role()
    region = sess.boto_region_name
    print(f"[SUCCESS] Session initialized in region: {region}")
except Exception as e:
    print(f"[ERROR] Failed to initialize session: {str(e)}")
    print("  -> Ensure you're running in a SageMaker notebook with proper IAM role")
    raise
```

### Step 3.5: Set Up ECR in GovCloud (Required for GovCloud)

**Note:** GovCloud cannot access public AWS ECR. You must copy the HuggingFace container to your GovCloud ECR.

#### Option A: Run from a machine with access to BOTH commercial AWS and GovCloud

**Prerequisites:**
- Docker installed
- AWS CLI configured with both commercial and GovCloud profiles

```bash
# ============================================
# Run these commands in a terminal (not notebook)
# ============================================

# Variables - UPDATE THESE
GOVCLOUD_ACCOUNT_ID="your-govcloud-account-id"
GOVCLOUD_REGION="us-gov-west-1"
COMMERCIAL_REGION="us-east-1"
ECR_REPO_NAME="huggingface-llm-tgi"

# Source image from commercial AWS (HuggingFace TGI container)
SOURCE_IMAGE="763104351884.dkr.ecr.${COMMERCIAL_REGION}.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"

# Target image in GovCloud ECR
TARGET_IMAGE="${GOVCLOUD_ACCOUNT_ID}.dkr.ecr.${GOVCLOUD_REGION}.amazonaws.com/${ECR_REPO_NAME}:latest"

# Step 1: Login to commercial AWS ECR
aws ecr get-login-password --region ${COMMERCIAL_REGION} --profile commercial | \
    docker login --username AWS --password-stdin 763104351884.dkr.ecr.${COMMERCIAL_REGION}.amazonaws.com

# Step 2: Pull the HuggingFace container
docker pull ${SOURCE_IMAGE}

# Step 3: Create ECR repository in GovCloud (if not exists)
aws ecr create-repository \
    --repository-name ${ECR_REPO_NAME} \
    --region ${GOVCLOUD_REGION} \
    --profile govcloud \
    2>/dev/null || echo "Repository may already exist"

# Step 4: Login to GovCloud ECR
aws ecr get-login-password --region ${GOVCLOUD_REGION} --profile govcloud | \
    docker login --username AWS --password-stdin ${GOVCLOUD_ACCOUNT_ID}.dkr.ecr.${GOVCLOUD_REGION}.amazonaws.com

# Step 5: Tag and push to GovCloud ECR
docker tag ${SOURCE_IMAGE} ${TARGET_IMAGE}
docker push ${TARGET_IMAGE}

echo "[SUCCESS] Container pushed to GovCloud ECR: ${TARGET_IMAGE}"
```

#### Option B: If containers are pre-staged in GovCloud ECR

If your organization has already copied HuggingFace containers to GovCloud, get the ECR URI from your admin.

---

### Step 4: Configure the HuggingFace LLM Model

**Note:** This model is in a private HuggingFace repository. You must provide a HuggingFace access token.

To generate a token:
1. Go to https://huggingface.co/settings/tokens
2. Click "New token"
3. Select "Read" access
4. Copy the token (starts with `hf_...`)

```python
from sagemaker.huggingface import HuggingFaceModel

# ============================================
# GovCloud ECR Configuration
# ============================================
# UPDATE with your GovCloud account ID
GOVCLOUD_ACCOUNT_ID = "your-govcloud-account-id"
GOVCLOUD_REGION = "us-gov-west-1"
ECR_REPO_NAME = "huggingface-llm-tgi"

# Use the container image from your GovCloud ECR
llm_image = f"{GOVCLOUD_ACCOUNT_ID}.dkr.ecr.{GOVCLOUD_REGION}.amazonaws.com/{ECR_REPO_NAME}:latest"

print(f"Using GovCloud ECR image: {llm_image}")

# IMPORTANT: Replace with your HuggingFace token for private repo access
HF_TOKEN = 'hf_xxxxxxxxxxxxxxxxxxxx'  # Your HuggingFace access token

hub_config = {
    'HF_MODEL_ID': 'HeyHey12Hey/codellama-7b-qlora-finetune',
    'HF_TOKEN': HF_TOKEN,  # Required for private repository access
    'SM_NUM_GPUS': '1',
    'MAX_INPUT_LENGTH': '2048',
    'MAX_TOTAL_TOKENS': '4096',
}

# Validate token is set
if HF_TOKEN == 'hf_xxxxxxxxxxxxxxxxxxxx' or not HF_TOKEN.startswith('hf_'):
    print("[ERROR] Invalid HuggingFace token")
    print("  -> Replace HF_TOKEN with your actual token from https://huggingface.co/settings/tokens")
    raise ValueError("HuggingFace token not configured")

try:
    huggingface_model = HuggingFaceModel(
        image_uri=llm_image,
        env=hub_config,
        role=role,
    )
    print("[SUCCESS] HuggingFace model configured")
except Exception as e:
    print(f"[ERROR] Failed to configure model: {str(e)}")
    raise
```

### Step 5: Deploy the Model to an Endpoint

```python
ENDPOINT_NAME = "codellama-7b-qlora-endpoint"

try:
    print(f"Deploying endpoint '{ENDPOINT_NAME}'...")
    print("  This typically takes 5-15 minutes. Please wait...")

    start_time = time.time()

    predictor = huggingface_model.deploy(
        initial_instance_count=1,
        instance_type="ml.g5.xlarge",
        endpoint_name=ENDPOINT_NAME,
        container_startup_health_check_timeout=600,  # 10 min timeout for model loading
    )

    elapsed = (time.time() - start_time) / 60
    print(f"[SUCCESS] Endpoint deployed successfully in {elapsed:.1f} minutes")

except ClientError as e:
    error_code = e.response['Error']['Code']
    error_msg = e.response['Error']['Message']

    if error_code == 'ResourceLimitExceeded':
        print(f"[ERROR] Resource limit exceeded: {error_msg}")
        print("  -> Request a quota increase for ml.g5.xlarge in Service Quotas")
    elif error_code == 'ResourceInUse':
        print(f"[ERROR] Endpoint name already exists: {ENDPOINT_NAME}")
        print("  -> Delete existing endpoint or use a different name")
    elif error_code == 'ValidationException':
        print(f"[ERROR] Validation error: {error_msg}")
        print("  -> Check instance type availability in your region")
    else:
        print(f"[ERROR] AWS error ({error_code}): {error_msg}")
    raise

except Exception as e:
    print(f"[ERROR] Deployment failed: {str(e)}")
    raise
```

### Step 6: Test the Endpoint

```python
def invoke_endpoint(prompt, max_new_tokens=256, temperature=0.7):
    """Invoke the endpoint with error handling."""
    try:
        response = predictor.predict({
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": 0.9,
                "do_sample": True,
            }
        })
        return response

    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_msg = e.response['Error']['Message']

        if error_code == 'ModelError':
            print(f"[ERROR] Model error: {error_msg}")
            print("  -> The model may have failed to load or crashed")
        elif error_code == 'ValidationError':
            print(f"[ERROR] Invalid input: {error_msg}")
            print("  -> Check your prompt format and parameters")
        elif error_code == 'ThrottlingException':
            print(f"[ERROR] Rate limited: {error_msg}")
            print("  -> Wait and retry, or increase instance count")
        else:
            print(f"[ERROR] Inference error ({error_code}): {error_msg}")
        raise

    except EndpointConnectionError:
        print("[ERROR] Cannot connect to endpoint")
        print("  -> Check if endpoint is still running")
        raise

    except json.JSONDecodeError:
        print("[ERROR] Invalid response format from model")
        raise

# Test the endpoint
try:
    print("Testing endpoint...")
    result = invoke_endpoint("def fibonacci(n):")
    print("[SUCCESS] Inference successful!")
    print(f"Response: {result}")
except Exception as e:
    print(f"[ERROR] Test failed: {str(e)}")
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
def cleanup_endpoint(predictor):
    """Delete endpoint and model with error handling."""
    try:
        print("Deleting endpoint...")
        predictor.delete_endpoint()
        print("[SUCCESS] Endpoint deleted")
    except ClientError as e:
        if e.response['Error']['Code'] == 'ResourceNotFound':
            print("  Endpoint already deleted")
        else:
            print(f"[ERROR] Failed to delete endpoint: {str(e)}")

    try:
        print("Deleting model...")
        predictor.delete_model()
        print("[SUCCESS] Model deleted")
    except ClientError as e:
        if e.response['Error']['Code'] == 'ResourceNotFound':
            print("  Model already deleted")
        else:
            print(f"[ERROR] Failed to delete model: {str(e)}")

# Run cleanup when done (uncomment when ready):
# cleanup_endpoint(predictor)
```

---

## Common Errors and Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| `ResourceLimitExceeded` | No quota for instance type | Request quota increase in AWS console |
| `ResourceInUse` | Endpoint name exists | Delete existing or use new name |
| `ModelError` | Model failed to load | Check CloudWatch logs, try larger instance |
| `ThrottlingException` | Too many requests | Add retry logic with backoff |
| `Timeout` | Model too slow to load | Increase `container_startup_health_check_timeout` |
| `ValidationException` | Invalid instance type | Check instance availability in region |

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

## Cost Considerations

- **Endpoint costs**: Charged per hour while running (~$1.00/hr for g5.xlarge)
- **Storage**: Model artifacts stored in S3 (minimal cost)
- **Data transfer**: Ingress free, egress charged

**Tip**: Use SageMaker Serverless Inference or Asynchronous Inference for cost optimization if real-time isn't required.

---

## Note on QLora Fine-tuned Models

This model (`HeyHey12Hey/codellama-7b-qlora-finetune`) is a QLora fine-tuned version of CodeLlama-7B. The adapter weights have been merged into the base model, so it deploys like a standard model. If you encounter issues with merged weights, you may need to use the `peft` library to load adapter weights separately.

# AWS Setup Guide — Start from Zero

This guide walks you through everything needed to run Pitwall AI on AWS:
S3 for data storage, ECR for container images, and EKS for GPU training.

> **Cost estimate:** Phase 1 (S3 + ECR only) costs < $5/month.
> Phase 2 (EKS) adds ~$150/month while the cluster is running — spin it down
> when you are not training.

---

## Prerequisites — install these first

### 1. AWS CLI v2

**Windows (PowerShell, run as Administrator):**
```powershell
msiexec.exe /i https://awscli.amazonaws.com/AWSCLIV2.msi /quiet
# Restart terminal, then verify:
aws --version
```

**macOS / Linux:**
```bash
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o awscliv2.zip
unzip awscliv2.zip && sudo ./aws/install
```

### 2. Terraform ≥ 1.7

**Windows:**
```powershell
# Install via winget (easiest):
winget install HashiCorp.Terraform
# Or download from https://developer.hashicorp.com/terraform/downloads
```

**macOS / Linux:**
```bash
# Via tfenv (lets you switch versions):
git clone https://github.com/tfutils/tfenv.git ~/.tfenv
echo 'export PATH="$HOME/.tfenv/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc
tfenv install 1.8.5 && tfenv use 1.8.5
```

### 3. kubectl

```powershell
# Windows (PowerShell):
winget install Kubernetes.kubectl

# macOS:
brew install kubectl
```

### 4. Helm (needed for Week 7 deployment)

```powershell
winget install Helm.Helm      # Windows
brew install helm             # macOS
```

---

## Step 1 — Create an AWS account

1. Go to <https://aws.amazon.com> → **Create an AWS Account**
2. Choose **Personal** account type, enter payment info (required even for Free Tier)
3. Select the **Free** support plan

> **Billing alert:** Set a budget alarm immediately after sign-up so you are
> notified if you approach $50/month.
> IAM → Billing → Budgets → Create budget → Zero-spend or $50 threshold.

---

## Step 2 — Create an IAM user for CLI access

Never use your root account for day-to-day work.

1. Sign in to the [AWS Console](https://console.aws.amazon.com)
2. Go to **IAM → Users → Create user**
3. Username: `pitwall-admin`
4. Attach policy: **AdministratorAccess** (narrow this down before production)
5. **Create access key** → use case: *CLI* → copy **Access Key ID** and **Secret Access Key**

---

## Step 3 — Configure the AWS CLI

```bash
aws configure
# AWS Access Key ID:     <paste your key>
# AWS Secret Access Key: <paste your secret>
# Default region name:   us-east-1
# Default output format: json
```

Verify it works:
```bash
aws sts get-caller-identity
# Should print your Account ID, UserId, and ARN
```

---

## Step 4 — Bootstrap Terraform state storage

Terraform needs an S3 bucket to store its own state file, and a DynamoDB
table to prevent concurrent runs from corrupting it.  We create these once
with a tiny bootstrap module (its own local state is fine — the bucket rarely
changes).

```powershell
cd infra/terraform/aws/bootstrap

terraform init
terraform apply -auto-approve

# Note the outputs — you'll need them in Step 5:
#   state_bucket = "pitwall-ai-tfstate-<your-account-id>"
#   lock_table   = "pitwall-ai-tflock"
```

---

## Step 5 — Configure the remote backend

```bash
cd ../   # back to infra/terraform/aws/

# Copy the backend example and fill in your values:
cp backend.tf.example backend.tf
```

Edit `backend.tf` — replace `<account-id>` with your 12-digit AWS account ID:
```hcl
terraform {
  backend "s3" {
    bucket         = "pitwall-ai-tfstate-<account-id>"
    key            = "pitwall-ai/terraform.tfstate"
    region         = "us-east-1"
    dynamodb_table = "pitwall-ai-tflock"
    encrypt        = true
  }
}
```

---

## Step 6 — Phase 1: provision S3 + ECR

This is all you need to start ingesting data.  EKS comes in Phase 2.

```powershell
# Copy and review variable defaults:
Copy-Item terraform.tfvars.example terraform.tfvars

terraform init
```

Then apply Phase 1 via the Makefile (handles line continuation correctly on all platforms):

```powershell
# Run from the repo root:
make tf-apply-phase1
```

<details>
<summary>Raw terraform command (PowerShell — note the backtick line continuation)</summary>

```powershell
terraform apply `
  -target=aws_s3_bucket.data_lake `
  -target=aws_s3_bucket_versioning.data_lake `
  -target=aws_s3_bucket_server_side_encryption_configuration.data_lake `
  -target=aws_s3_bucket_public_access_block.data_lake `
  -target=aws_s3_bucket_lifecycle_configuration.data_lake `
  -target=aws_ecr_repository.training `
  -target=aws_ecr_lifecycle_policy.training
```
</details>

When it completes, note the outputs:
```
s3_bucket_name = "pitwall-ai-prod"
ecr_registry   = "<account-id>.dkr.ecr.us-east-1.amazonaws.com"
```

---

## Step 7 — Run the ingestion pipeline into S3

Set your environment variables in PowerShell (or add them to `.env`):
```powershell
$env:PITWALL_STORAGE_BACKEND = "s3"
$env:PITWALL_AWS_REGION      = "us-east-1"
$env:PITWALL_AWS_S3_BUCKET   = "pitwall-ai-prod"   # from terraform output s3_bucket_name
```

Run ingestion for one race to verify the pipeline works end-to-end:
```powershell
make ingest SEASON=2024 ROUND=1 SESSION=R
```

Check the data landed in S3:
```powershell
aws s3 ls s3://pitwall-ai-prod/bronze/ --recursive
aws s3 ls s3://pitwall-ai-prod/silver/ --recursive
aws s3 ls s3://pitwall-ai-prod/gold/   --recursive
```

Once verified, ingest a full season:
```powershell
make ingest-season SEASON=2024
```

---

## Step 8 — Phase 2: provision VPC + EKS + GPU nodes (for training)

Do this only when you are ready to train models.  The EKS control plane costs
~$0.10/hour regardless of whether pods are running.

```bash
cd infra/terraform/aws/

terraform apply   # applies everything not yet applied (VPC + EKS + IRSA)
```

This takes **10–15 minutes**.  When done:

```bash
# Update your local kubeconfig:
aws eks update-kubeconfig --name pitwall-prod --region us-east-1

# Verify the cluster:
kubectl get nodes
```

GPU nodes start at 0 replicas — they scale up automatically when an Argo
Workflow requests `nvidia.com/gpu: 1` and scale back to 0 when the job ends.

---

## Step 9 — Submit training jobs

```bash
# Install Argo CLI (Windows):
winget install Argo.Argo

# Port-forward the Argo server UI:
kubectl port-forward svc/argo-server 2746:2746 -n argo

# Submit tire degradation training:
make submit-train-tire EKS_CLUSTER=pitwall-prod

# Submit GPU strategy policy training:
make submit-train-policy EKS_CLUSTER=pitwall-prod
```

---

## Teardown (stop incurring EKS charges)

```powershell
# Destroy only EKS (keeps S3 data intact) — use the Makefile target:
make tf-destroy-eks

# Full teardown (deletes S3 data too — careful!):
make tf-destroy
```

---

## Cost summary

| Resource | Always-on cost | Notes |
|---|---|---|
| S3 (data lake) | ~$1–5/month | Depends on data volume |
| ECR | ~$0.10/GB/month | Minimal for this project |
| EKS control plane | $0.10/hr (~$73/mo) | Phase 2 only; destroy when idle |
| General nodes (t3.large ×2) | ~$0.08/hr | Phase 2 only |
| GPU node (g4dn.xlarge) | $0.53/hr | Zero-scaled; only active during training |

**Tip:** Use [AWS Cost Explorer](https://console.aws.amazon.com/cost-management/home)
and set a $50 budget alert.

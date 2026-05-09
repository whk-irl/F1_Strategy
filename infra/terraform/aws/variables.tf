variable "aws_region" {
  description = "AWS region for all resources."
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Short identifier used as a prefix in all resource names."
  type        = string
  default     = "pitwall-ai"
}

variable "environment" {
  description = "Deployment environment (prod | staging)."
  type        = string
  default     = "prod"
  validation {
    condition     = contains(["prod", "staging"], var.environment)
    error_message = "environment must be 'prod' or 'staging'."
  }
}

# ---------------------------------------------------------------------------
# S3
# ---------------------------------------------------------------------------

variable "bronze_retention_days" {
  description = "Days before bronze Parquet files are transitioned to Glacier."
  type        = number
  default     = 90
}

# ---------------------------------------------------------------------------
# EKS
# ---------------------------------------------------------------------------

variable "eks_cluster_version" {
  description = "Kubernetes version for the EKS control plane."
  type        = string
  default     = "1.29"
}

variable "general_node_instance_type" {
  description = "EC2 instance type for the general-purpose node group."
  type        = string
  default     = "t3.large"
}

variable "general_node_desired" {
  description = "Desired number of general-purpose nodes."
  type        = number
  default     = 2
}

variable "general_node_min" {
  type    = number
  default = 1
}

variable "general_node_max" {
  type    = number
  default = 4
}

variable "gpu_node_instance_type" {
  description = "EC2 instance type for the GPU training node group."
  type        = string
  default     = "g4dn.xlarge"   # 1× T4 GPU, 16 GB VRAM, ~$0.53/hr on-demand
}

variable "gpu_node_max" {
  description = "Maximum GPU nodes (0 = effectively disabled)."
  type        = number
  default     = 2
}

# ---------------------------------------------------------------------------
# Tags
# ---------------------------------------------------------------------------

variable "tags" {
  description = "Additional tags applied to all resources."
  type        = map(string)
  default     = {}
}

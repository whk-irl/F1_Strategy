output "s3_bucket_name" {
  description = "Name of the S3 data lake bucket."
  value       = aws_s3_bucket.data_lake.bucket
}

output "s3_bucket_arn" {
  description = "ARN of the S3 data lake bucket."
  value       = aws_s3_bucket.data_lake.arn
}

output "ecr_registry" {
  description = "ECR registry base URL (prefix every image name with this)."
  value       = "${data.aws_caller_identity.current.account_id}.dkr.ecr.${var.aws_region}.amazonaws.com"
}

output "ecr_repositories" {
  description = "Map of ECR repository names to their URLs."
  value       = { for k, v in aws_ecr_repository.training : k => v.repository_url }
}

output "eks_cluster_name" {
  description = "EKS cluster name."
  value       = module.eks.cluster_name
}

output "eks_cluster_endpoint" {
  description = "EKS API server endpoint."
  value       = module.eks.cluster_endpoint
  sensitive   = true
}

output "training_role_arn" {
  description = "IAM role ARN for the pitwall-training service account (IRSA)."
  value       = module.pitwall_training_irsa.iam_role_arn
}

output "kubeconfig_command" {
  description = "Run this command to update your local kubeconfig."
  value       = "aws eks update-kubeconfig --name ${module.eks.cluster_name} --region ${var.aws_region}"
}

output "env_vars_snippet" {
  description = "Add these to your .env to point the pipeline at S3."
  value       = <<-EOT
    PITWALL_STORAGE_BACKEND=s3
    PITWALL_AWS_REGION=${var.aws_region}
    PITWALL_AWS_S3_BUCKET=${aws_s3_bucket.data_lake.bucket}
  EOT
}

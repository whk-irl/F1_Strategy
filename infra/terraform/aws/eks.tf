# EKS cluster with two managed node groups:
#   - general:      t3.large × 2 — runs Argo, MLflow, Prefect, API, frontend
#   - gpu-training: g4dn.xlarge, min=0 — zero-scaled; spins up only for RL training

module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 20.13"

  cluster_name    = local.cluster_name
  cluster_version = var.eks_cluster_version

  vpc_id                   = module.vpc.vpc_id
  subnet_ids               = module.vpc.private_subnets
  control_plane_subnet_ids = module.vpc.private_subnets

  # Allow your local machine to reach the API server.
  cluster_endpoint_public_access = true

  # Grant the Terraform caller admin permissions on the cluster so subsequent
  # kubectl / helm / argo commands work out of the box.
  enable_cluster_creator_admin_permissions = true

  # ---------------------------------------------------------------------------
  # Add-ons
  # ---------------------------------------------------------------------------
  cluster_addons = {
    coredns = {
      most_recent = true
    }
    kube-proxy = {
      most_recent = true
    }
    vpc-cni = {
      most_recent    = true
      before_compute = true   # must be ready before nodes join
    }
    aws-ebs-csi-driver = {
      most_recent              = true
      service_account_role_arn = module.ebs_csi_irsa.iam_role_arn
    }
  }

  # ---------------------------------------------------------------------------
  # Node groups
  # ---------------------------------------------------------------------------
  eks_managed_node_groups = {

    # General-purpose nodes — always on, run platform workloads.
    general = {
      name           = "general"
      instance_types = [var.general_node_instance_type]
      min_size       = var.general_node_min
      max_size       = var.general_node_max
      desired_size   = var.general_node_desired
      disk_size      = 50   # GB

      # Spot instances for ~60-70% savings on non-critical workloads.
      capacity_type = "SPOT"

      labels = {
        role = "general"
      }
    }

    # GPU training nodes — zero-scaled by default, scale up per training job.
    gpu-training = {
      name           = "gpu-training"
      instance_types = [var.gpu_node_instance_type]
      ami_type       = "AL2_x86_64_GPU"   # Amazon Linux 2 with NVIDIA drivers

      min_size     = 0   # cost control: no idle GPU nodes
      max_size     = var.gpu_node_max
      desired_size = 0

      disk_size = 100   # GB — PyTorch + CUDA layer cache is large

      # On-demand for GPU (Spot availability is inconsistent for g4dn).
      capacity_type = "ON_DEMAND"

      labels = {
        role                               = "gpu-training"
        "eks.amazonaws.com/nodegroup"      = "gpu-training"
      }

      # Taint prevents non-GPU workloads from landing on expensive GPU nodes.
      taints = [
        {
          key    = "nvidia.com/gpu"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
      ]
    }
  }
}

# ---------------------------------------------------------------------------
# EBS CSI driver IRSA (needed for PersistentVolume support)
# ---------------------------------------------------------------------------

module "ebs_csi_irsa" {
  source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.39"

  role_name             = "${local.cluster_name}-ebs-csi"
  attach_ebs_csi_policy = true

  oidc_providers = {
    main = {
      provider_arn               = module.eks.oidc_provider_arn
      namespace_service_accounts = ["kube-system:ebs-csi-controller-sa"]
    }
  }
}

# ---------------------------------------------------------------------------
# Cluster autoscaler IRSA
# The autoscaler scales the gpu-training node group from 0 → N when a pod
# with nvidia.com/gpu requests lands in Pending state.
# ---------------------------------------------------------------------------

module "cluster_autoscaler_irsa" {
  source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.39"

  role_name                        = "${local.cluster_name}-cluster-autoscaler"
  attach_cluster_autoscaler_policy = true
  cluster_autoscaler_cluster_names = [module.eks.cluster_name]

  oidc_providers = {
    main = {
      provider_arn               = module.eks.oidc_provider_arn
      namespace_service_accounts = ["kube-system:cluster-autoscaler"]
    }
  }
}

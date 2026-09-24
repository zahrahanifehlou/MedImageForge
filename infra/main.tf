# MedImageForge — AWS architecture sketch (Step 17)
#
# STATUS: sketch, not deployment-ready. It encodes the SHAPE of the target
# architecture — the buckets, roles, and task boundaries a cloud engineer
# would implement — with the same zone discipline the local pipeline uses.
# Not validated with `terraform plan`; no provider credentials configured.
#
# The mapping rule throughout: every local path in configs/default.yaml is
# a "zone" — in the cloud a zone is a bucket (or a DB) plus an IAM policy
# that decides who may touch it. The privacy boundary that on localhost is
# a code convention ("the API speaks pseudonyms") becomes an IAM fact.

terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.0" }
  }
}

provider "aws" {
  region = var.region
}

variable "region" {
  default = "eu-west-1"
}

variable "project" {
  default = "medimageforge"
}

locals {
  tags = { Project = var.project, ManagedBy = "terraform-sketch" }
}

"""Official documentation crawl sources (AWS services, Docker, Google Drive API, …)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DocSourceSpec:
    source_id: str
    output_subdir: str
    allowed_netlocs: tuple[str, ...]
    start_urls: tuple[str, ...]
    allowed_path_prefixes: tuple[str, ...] = ()
    default_title: str = "Documentation"


def _aws(service: str, path_prefix: str, start_url: str) -> DocSourceSpec:
    return DocSourceSpec(
        source_id=f"aws_{service}",
        output_subdir=f"aws/{service}",
        allowed_netlocs=("docs.aws.amazon.com",),
        allowed_path_prefixes=(path_prefix,),
        start_urls=(start_url,),
        default_title=f"AWS {service.upper()} documentation",
    )


AWS_SERVICE_SOURCES: dict[str, DocSourceSpec] = {
    "ec2": _aws("ec2", "/AWSEC2/", "https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/concepts.html"),
    "ec2_alt": _aws(
        "ec2_alt",
        "/ec2/",
        "https://docs.aws.amazon.com/ec2/latest/instancetypes/instance-types.html",
    ),
    "s3": _aws("s3", "/AmazonS3/", "https://docs.aws.amazon.com/AmazonS3/latest/userguide/Welcome.html"),
    "iam": _aws("iam", "/IAM/", "https://docs.aws.amazon.com/IAM/latest/UserGuide/introduction.html"),
    "lambda": _aws(
        "lambda",
        "/lambda/",
        "https://docs.aws.amazon.com/lambda/latest/dg/welcome.html",
    ),
    "rds": _aws("rds", "/AmazonRDS/", "https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/Welcome.html"),
    "dynamodb": _aws(
        "dynamodb",
        "/amazondynamodb/",
        "https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/Introduction.html",
    ),
    "vpc": _aws("vpc", "/vpc/", "https://docs.aws.amazon.com/vpc/latest/userguide/what-is-amazon-vpc.html"),
    "cloudwatch": _aws(
        "cloudwatch",
        "/AmazonCloudWatch/",
        "https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/WhatIsCloudWatch.html",
    ),
    "sqs": _aws("sqs", "/AWSSimpleQueueService/", "https://docs.aws.amazon.com/AWSSimpleQueueService/latest/SQSWelcome.html"),
    "sns": _aws("sns", "/sns/", "https://docs.aws.amazon.com/sns/latest/dg/welcome.html"),
    "eks": _aws("eks", "/eks/", "https://docs.aws.amazon.com/eks/latest/userguide/what-is-eks.html"),
    "ecs": _aws("ecs", "/AmazonECS/", "https://docs.aws.amazon.com/AmazonECS/latest/developerguide/Welcome.html"),
    "cloudformation": _aws(
        "cloudformation",
        "/AWSCloudFormation/",
        "https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/Welcome.html",
    ),
    "route53": _aws(
        "route53",
        "/Route53/",
        "https://docs.aws.amazon.com/Route53/latest/DeveloperGuide/Welcome.html",
    ),
    "ebs": _aws("ebs", "/ebs/", "https://docs.aws.amazon.com/ebs/latest/userguide/what-is-ebs.html"),
    "elb": _aws(
        "elb",
        "/elasticloadbalancing/",
        "https://docs.aws.amazon.com/elasticloadbalancing/latest/userguide/what-is-load-balancing.html",
    ),
    "kms": _aws("kms", "/kms/", "https://docs.aws.amazon.com/kms/latest/developerguide/overview.html"),
    "secretsmanager": _aws(
        "secretsmanager",
        "/secretsmanager/",
        "https://docs.aws.amazon.com/secretsmanager/latest/userguide/intro.html",
    ),
}

OTHER_SOURCES: dict[str, DocSourceSpec] = {
    "docker": DocSourceSpec(
        source_id="docker",
        output_subdir="docker",
        allowed_netlocs=("docs.docker.com",),
        allowed_path_prefixes=("/",),
        start_urls=(
            "https://docs.docker.com/get-started/",
            "https://docs.docker.com/engine/",
            "https://docs.docker.com/compose/",
            "https://docs.docker.com/desktop/",
            "https://docs.docker.com/build/",
            "https://docs.docker.com/reference/",
        ),
        default_title="Docker documentation",
    ),
    "google_drive": DocSourceSpec(
        source_id="google_drive",
        output_subdir="google_drive",
        allowed_netlocs=("developers.google.com",),
        allowed_path_prefixes=(
            "/workspace/drive",
            "/drive",
            "/workspace/docs",
        ),
        start_urls=(
            "https://developers.google.com/workspace/drive/api/guides/about-sdk",
            "https://developers.google.com/drive/api/guides/about-sdk",
            "https://developers.google.com/workspace/drive/api/reference/rest/v3",
            "https://developers.google.com/workspace/docs",
        ),
        default_title="Google Workspace / Drive documentation",
    ),
}

ALL_AWS_SERVICE_KEYS: tuple[str, ...] = tuple(sorted(AWS_SERVICE_SOURCES.keys()))
ALL_SOURCE_IDS: tuple[str, ...] = ALL_AWS_SERVICE_KEYS + tuple(sorted(OTHER_SOURCES.keys()))

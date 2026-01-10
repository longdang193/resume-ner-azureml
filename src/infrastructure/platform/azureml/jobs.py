from __future__ import annotations

"""
@meta
name: azureml_jobs
type: utility
domain: azureml
responsibility:
  - Submit Azure ML jobs and wait for completion
  - Handle job status monitoring
inputs:
  - Azure ML job objects
outputs:
  - Completed job instances
tags:
  - utility
  - azureml
  - jobs
ci:
  runnable: false
  needs_gpu: false
  needs_cloud: true
lifecycle:
  status: active
"""
from typing import Any

from azure.ai.ml import MLClient
from azure.ai.ml.entities import Job

def submit_and_wait_for_job(ml_client: MLClient, job: Any) -> Job:
    """
    Submit a job and block until it completes or fails.

    Args:
        ml_client: Azure ML client used for job operations.
        job: Job object (command or sweep) to submit.

    Returns:
        Completed :class:`Job` instance.

    Raises:
        RuntimeError: If the job terminates with a non-``Completed`` status.
    """
    submitted = ml_client.jobs.create_or_update(job)
    ml_client.jobs.stream(submitted.name)
    completed = ml_client.jobs.get(submitted.name)
    if completed.status != "Completed":
        raise RuntimeError(
            f"Job {completed.name} failed with status: {completed.status}"
        )
    return completed


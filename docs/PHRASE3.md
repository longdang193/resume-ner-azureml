Once the system is live, the primary risks shift from *building the model* to *operating it safely*.

This phase ensures the inference pipeline remains **reliable, observable, and controllable** over time, even as data, traffic, and models change.

## Step P3-1: Monitoring (System & Model Observability)

### Context

A production ML system must be observable at two levels:

* **System health** (latency, throughput, failures)
* **Model behavior** (input characteristics, output signals)

Both are required to detect failures and degradation early.

### Action

1. Instrument the **Azure Function** and **Managed Online Endpoint** with Azure Application Insights.
2. Log **system-level metrics**:

* `inference_latency_ms`
* `request_count`
* `error_rate`

3. Log **model-level signals**:

* `entities_detected_count`
* `empty_prediction_rate`
* input size statistics (e.g., text length)

4. Create an Azure Dashboard showing:

* average inference latency
* number of resumes processed
* error rate over time

5. Configure alerts for:

* sustained latency increase
* elevated error rate
* sudden drop in extracted entities

### Rationale

* System metrics ensure availability and performance.
* Model-level signals act as **proxies for model quality** when ground truth is unavailable.
* Alerts enable proactive intervention instead of reactive debugging.

### Important

* Application-level and model-level metrics must be logged **separately**.
* Monitoring should be lightweight to avoid inference overhead.
* Metrics should be tagged with **model version** and **endpoint deployment name**.

### Deliverable

* Application Insights workspace with custom metrics.
* Dashboards and alerts covering system health and model behavior.

## Step P3-2: CI/CD (Training, Validation & Deployment Automation)

### Context

Model updates must be automated **without sacrificing safety**.

CI/CD in ML systems must include **validation and deployment gates**, not just automation.

### Action

1. Configure GitHub Actions pipelines:

* **Training pipeline**
* Triggered only on changes to:
* training code
* model configs
* dataset versions
* Submits an Azure ML training job.
* **Evaluation & gating**
* Compare new model metrics against:
* previous production model
* minimum quality thresholds
* **Deployment pipeline**
* If checks pass, update the Managed Online Endpoint to a new deployment.

2. Use staged rollout:

* deploy new model as secondary deployment
* shift traffic gradually (canary or blue/green)

### Rationale

* Conditional triggers prevent unnecessary retraining.
* Quality gates prevent silent regressions.
* Staged deployment reduces blast radius of failures.

### Important

* Not every code change should trigger retraining.
* Automatic deployment must be blocked if validation fails.
* Rollback to the previous model must be possible without retraining.

### Deliverable

* GitHub Actions workflows for training, validation, and deployment.
* Managed Online Endpoint with controlled rollout capability.

## Step P3-3: Governance & Lineage (Model Safety Controls)

### Context

As models evolve, teams must be able to answer:

* which model is running?
* how was it trained?
* what data did it use?

### Action

1. Record lineage metadata:

* model version
* dataset version
* training config hash
* MLflow run ID

2. Enforce promotion rules:

* only models tagged `stage=prod` may be deployed

3. Maintain an audit trail of:

* deployments
* rollbacks
* performance changes

### Rationale

* Enables reproducibility and accountability.
* Supports debugging and compliance needs.
* Prevents accidental deployment of experimental models.

### Important

* Model registry is the source of truth.
* Production deployments must reference registry versions, not artifacts.
* Lineage must be immutable once recorded.

### Deliverable

* Traceable model lifecycle from training to production.
* Registry-backed deployment history.

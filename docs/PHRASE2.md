This phase implements the **production, always-on inference system**.

It automatically reacts to new user uploads, runs model inference using the **registered production ONNX model**, and persists structured results for downstream applications.

## Step P2-1: Ingestion (The Trigger)

### Context

User data must be ingested in a way that is:

* automatic
* event-driven
* decoupled from model logic

File uploads are the natural trigger for inference.

### Action

1. A user or frontend application uploads a **new PDF resume** to the `incoming-resumes` Blob Storage container.
2. Azure Event Grid listens for `BlobCreated` events on that container.
3. Event Grid immediately emits a notification indicating a new file is available.

### Rationale

* Blob Storage is a natural, durable ingestion point for files.
* Event Grid provides low-latency, push-based notifications.
* This design avoids polling and enables near–real-time processing.

### Important

* Only **new uploads** should trigger inference.
* Event payloads must include blob URI and metadata.
* The ingestion step must remain stateless.

### Deliverable

* Event Grid subscription bound to the `incoming-resumes` container.
* Verified `BlobCreated` events triggering downstream processing.

## Step P2-2: Processing & Serving (The Brain)

### Context

Once ingestion occurs, the system must:

* extract text
* run model inference
* return structured predictions without coupling application logic to model implementation.

### Action

Two supported execution patterns:

**Option A – Azure Functions (Simpler / Cheaper)**

1. Event Grid triggers a Python Azure Function.
2. The Function:

* downloads the PDF
* extracts text
* loads the model from the registry
* runs inference
* produces JSON output

**Option B – Azure ML Managed Online Endpoint (Recommended)**

1. Event Grid triggers a Python Azure Function.
2. The Function:

* extracts text from the PDF
* sends the text to a Managed Online Endpoint

3. The Endpoint:

* loads the production ONNX model
* performs inference
* returns structured JSON

### Rationale

* Separating application logic (Functions) from model inference (Endpoint) improves:
  * scalability
  * model update safety
  * operational clarity
* Managed Online Endpoints provide:
  * autoscaling
  * health checks
  * versioned deployments

### Important

* The Azure Function must remain stateless.
* The model must be loaded **only from the Model Registry**.
* Model updates must not require Function redeployment.
* Option A is acceptable for prototypes; Option B is preferred for production.

### Deliverable

* Deployed Azure Function wired to Event Grid.
* Active Managed Online Endpoint serving inference requests.
* Successful end-to-end inference returning JSON.

## Step P2-3: Storage of Results

### Context

Inference outputs must be stored in a way that supports:

* variable schema
* partial results
* future enrichment

Resume data is inherently heterogeneous.

### Action

1. The Azure Function persists the inference output (JSON) into Azure Cosmos DB.
2. Each document includes:

* resume identifier
* extracted entities (e.g., name, email, skills)
* processing metadata (timestamp, model version)

### Rationale

* Cosmos DB (NoSQL) supports flexible schemas.
* Different resumes may produce different entity sets.
* Enables fast queries and downstream integrations.

### Important

* Each record must include model version and inference timestamp.
* Writes should be idempotent to avoid duplicates.
* PII access controls must be enforced.

### Deliverable

* Cosmos DB container populated with structured inference results.
* Queryable documents per processed resume.

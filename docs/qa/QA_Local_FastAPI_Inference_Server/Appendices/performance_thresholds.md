# Performance Thresholds

Performance thresholds must align with the defined SLAs and be enforceable in automated tests.

## Latency Thresholds

| Endpoint | P50 | P95 | Max |
| --------------------- | ------- | --------- | ---------- |
| `/predict` | ≤200 ms | ≤500 ms | ≤1,000 ms |
| `/predict/batch` | ≤500 ms | ≤1,500 ms | ≤3,000 ms |
| `/predict/file` | — | ≤3,000 ms | ≤5,000 ms |
| `/predict/file/batch` | — | ≤8,000 ms | ≤12,000 ms |

## Throughput Thresholds

| Endpoint | Threshold |
| --------------------- | ---------------- |
| `/predict` | ≥5 req/sec |
| `/predict/batch` | ≥2 batches/sec |
| `/predict/file` | ≥1 file/sec |
| `/predict/file/batch` | ≥0.5 batches/sec |

## Startup & Timeout Thresholds

| Scenario | Threshold |
| ----------------------- | ----------- |
| Model load | ≤30 seconds |
| Port availability check | ≤5 seconds |
| Server startup | ≤45 seconds |
| Text request timeout | ≤5 seconds |
| File request timeout | ≤15 seconds |
| Batch file timeout | ≤20 seconds |

## Performance Consistency Thresholds

| Metric | Threshold |
| ---------------------- | --------------------- |
| Latency variance (P95) | ≤ ±20% |
| Timeout rate | 0% |
| Error rate | 0% |
| Memory growth | No unbounded increase |

## Reporting & Traceability

* All tests must log:
  * Endpoint name
  * Input type
  * Latency metrics
  * SLA pass/fail status

* Performance results must be comparable across runs

* Historical baselines should be preserved when possible

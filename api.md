# AutoPatternChecker API Documentation

This document provides comprehensive API documentation for the AutoPatternChecker service.

## Base URL

- **Development**: `http://localhost:8000`
- **Production**: `https://your-domain.com`

## Authentication

Currently, the API does not require authentication. For production deployments, implement API key authentication or OAuth2.

## Content Type

All requests and responses use `application/json`.

## Error Handling

The API uses standard HTTP status codes:

- `200 OK`: Request successful
- `400 Bad Request`: Invalid request parameters
- `404 Not Found`: Resource not found
- `422 Unprocessable Entity`: Validation error
- `500 Internal Server Error`: Server error

Error responses include a JSON object with error details:

```json
{
  "detail": "Error message describing what went wrong"
}
```

## Endpoints

### Health Check

Check if the service is running and healthy.

**Endpoint**: `GET /health`

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00.000Z",
  "version": "1.0.0"
}
```

### Validate Value

Validate a value against learned patterns for a given composite key.

**Endpoint**: `POST /validate`

**Request Body**:
```json
{
  "key_parts": ["string", "string", "string"],
  "value": "string",
  "metadata": {
    "optional": "object"
  }
}
```

**Parameters**:
- `key_parts` (required): Array of exactly 3 strings forming the composite key
- `value` (required): The value to validate
- `metadata` (optional): Additional metadata for the validation request

**Response**:
```json
{
  "verdict": "accepted|reformatted|duplicate|anomaly|needs_review",
  "issues": ["string"],
  "suggested_fix": "string|null",
  "nearest_examples": ["string"],
  "confidence": 0.92,
  "rules_applied": ["string"],
  "processing_time_ms": 15.3
}
```

**Verdict Types**:
- `accepted`: Value matches expected patterns
- `reformatted`: Value was normalized and accepted
- `duplicate`: Value is semantically similar to existing values
- `anomaly`: Value doesn't match any known patterns
- `needs_review`: Value requires human review

**Example Request**:
```bash
curl -X POST "http://localhost:8000/validate" \
  -H "Content-Type: application/json" \
  -d '{
    "key_parts": ["Tools", "Accessories", "Accessory Type (184)"],
    "value": "SMB Mini Jack, Right Angle",
    "metadata": {"source": "user_input"}
  }'
```

**Example Response**:
```json
{
  "verdict": "accepted",
  "issues": [],
  "suggested_fix": null,
  "nearest_examples": ["SMB Mini Jack Right Angle"],
  "confidence": 0.92,
  "rules_applied": ["trim_whitespace"],
  "processing_time_ms": 15.3
}
```

### Get Key Profile

Retrieve detailed profile information for a specific composite key.

**Endpoint**: `GET /profile/{composite_key}`

**Parameters**:
- `composite_key` (path): The composite key to get profile for

**Response**:
```json
{
  "composite_key": "Tools||Accessories||Accessory Type (184)",
  "count": 1381,
  "unique_values": 132,
  "unique_ratio": 0.096,
  "is_free_text": false,
  "is_numeric_unit": false,
  "top_signatures": [
    {
      "sig": "L_L",
      "count": 1200,
      "pct": 86.9,
      "examples": ["SMB Mini Jack, Right Angle", "SC Plug"]
    }
  ],
  "candidate_regex": "[A-Za-z\\u0600-\\u06FF]+\\s*[A-Za-z\\u0600-\\u06FF]+",
  "normalization_rules": [
    {
      "name": "trim_whitespace",
      "priority": 1,
      "params": {}
    }
  ],
  "clusters": [
    {
      "cluster_id": 0,
      "size": 1200,
      "pattern_signature": "L_L",
      "cluster_regex": "[A-Za-z\\u0600-\\u06FF].+",
      "example_values": ["SMB Mini Jack, Right Angle"],
      "coverage_pct": 86.9
    }
  ],
  "similarity_threshold": 0.82
}
```

**Example Request**:
```bash
curl "http://localhost:8000/profile/Tools||Accessories||Accessory%20Type%20(184)"
```

### Get Review Items

Retrieve items that need human review.

**Endpoint**: `GET /review/next`

**Query Parameters**:
- `limit` (optional): Maximum number of items to return (default: 10)

**Response**:
```json
[
  {
    "id": "review_20240115_103000_001",
    "composite_key": "Tools||Accessories||Accessory Type (184)",
    "original_value": "Unknown Component",
    "suggested_fix": "SMB Mini Jack, Right Angle",
    "nearest_examples": ["SMB Mini Jack, Right Angle", "SC Plug"],
    "confidence": 0.45,
    "timestamp": "2024-01-15T10:30:00.000Z",
    "issues": ["format_mismatch", "low_confidence"]
  }
]
```

**Example Request**:
```bash
curl "http://localhost:8000/review/next?limit=5"
```

### Submit Review Decision

Submit a human review decision for a flagged item.

**Endpoint**: `POST /review/submit`

**Request Body**:
```json
{
  "item_id": "string",
  "decision": "accept|edit|reject",
  "corrected_value": "string|null",
  "notes": "string|null"
}
```

**Parameters**:
- `item_id` (required): ID of the review item
- `decision` (required): Human decision on the item
- `corrected_value` (optional): Corrected value if decision is "edit"
- `notes` (optional): Additional notes about the decision

**Response**:
```json
{
  "status": "accepted",
  "message": "Review decision processed"
}
```

**Example Request**:
```bash
curl -X POST "http://localhost:8000/review/submit" \
  -H "Content-Type: application/json" \
  -d '{
    "item_id": "review_20240115_103000_001",
    "decision": "edit",
    "corrected_value": "SMB Mini Jack, Right Angle",
    "notes": "Fixed formatting and spacing"
  }'
```

### Get Service Statistics

Retrieve statistics about the service and data.

**Endpoint**: `GET /stats`

**Response**:
```json
{
  "total_keys": 150,
  "review_queue_size": 23,
  "indices_loaded": 150,
  "normalization_rules_loaded": 150,
  "uptime_seconds": 3600,
  "total_validations": 1250,
  "average_processing_time_ms": 12.5
}
```

**Example Request**:
```bash
curl "http://localhost:8000/stats"
```

## Data Models

### ValidationRequest

```json
{
  "key_parts": ["string", "string", "string"],
  "value": "string",
  "metadata": {
    "optional": "object"
  }
}
```

### ValidationResponse

```json
{
  "verdict": "string",
  "issues": ["string"],
  "suggested_fix": "string|null",
  "nearest_examples": ["string"],
  "confidence": "number",
  "rules_applied": ["string"],
  "processing_time_ms": "number"
}
```

### ReviewItem

```json
{
  "id": "string",
  "composite_key": "string",
  "original_value": "string",
  "suggested_fix": "string|null",
  "nearest_examples": ["string"],
  "confidence": "number",
  "timestamp": "string",
  "issues": ["string"]
}
```

### ReviewDecision

```json
{
  "item_id": "string",
  "decision": "string",
  "corrected_value": "string|null",
  "notes": "string|null"
}
```

### ProfileResponse

```json
{
  "composite_key": "string",
  "count": "number",
  "unique_values": "number",
  "unique_ratio": "number",
  "is_free_text": "boolean",
  "is_numeric_unit": "boolean",
  "top_signatures": [
    {
      "sig": "string",
      "count": "number",
      "pct": "number",
      "examples": ["string"]
    }
  ],
  "candidate_regex": "string|null",
  "normalization_rules": [
    {
      "name": "string",
      "priority": "number",
      "params": "object"
    }
  ],
  "clusters": [
    {
      "cluster_id": "number",
      "size": "number",
      "pattern_signature": "string",
      "cluster_regex": "string",
      "example_values": ["string"],
      "coverage_pct": "number"
    }
  ],
  "similarity_threshold": "number"
}
```

## Rate Limiting

The API implements rate limiting to prevent abuse:

- **Default**: 100 requests per minute per IP
- **Burst**: 200 requests per minute for short bursts
- **Headers**: Rate limit information is included in response headers

Rate limit headers:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1642248000
```

## CORS

The API supports Cross-Origin Resource Sharing (CORS) for web applications:

- **Allowed Origins**: Configurable (default: all)
- **Allowed Methods**: GET, POST, OPTIONS
- **Allowed Headers**: Content-Type, Authorization

## WebSocket Support

For real-time updates, WebSocket support can be added:

```javascript
const ws = new WebSocket('ws://localhost:8000/ws');
ws.onmessage = function(event) {
  const data = JSON.parse(event.data);
  console.log('Update:', data);
};
```

## SDK Examples

### Python

```python
import requests

class AutoPatternCheckerClient:
    def __init__(self, base_url):
        self.base_url = base_url
    
    def validate(self, key_parts, value, metadata=None):
        response = requests.post(
            f"{self.base_url}/validate",
            json={
                "key_parts": key_parts,
                "value": value,
                "metadata": metadata or {}
            }
        )
        return response.json()
    
    def get_profile(self, composite_key):
        response = requests.get(f"{self.base_url}/profile/{composite_key}")
        return response.json()

# Usage
client = AutoPatternCheckerClient("http://localhost:8000")
result = client.validate(
    ["Tools", "Accessories", "Accessory Type (184)"],
    "SMB Mini Jack, Right Angle"
)
print(result)
```

### JavaScript

```javascript
class AutoPatternCheckerClient {
  constructor(baseUrl) {
    this.baseUrl = baseUrl;
  }
  
  async validate(keyParts, value, metadata = {}) {
    const response = await fetch(`${this.baseUrl}/validate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        key_parts: keyParts,
        value: value,
        metadata: metadata
      })
    });
    return await response.json();
  }
  
  async getProfile(compositeKey) {
    const response = await fetch(`${this.baseUrl}/profile/${compositeKey}`);
    return await response.json();
  }
}

// Usage
const client = new AutoPatternCheckerClient('http://localhost:8000');
const result = await client.validate(
  ['Tools', 'Accessories', 'Accessory Type (184)'],
  'SMB Mini Jack, Right Angle'
);
console.log(result);
```

## Error Codes

| Code | Description |
|------|-------------|
| `VALIDATION_ERROR` | Input validation failed |
| `KEY_NOT_FOUND` | Composite key not found in profiles |
| `EMBEDDING_ERROR` | Error generating embeddings |
| `INDEX_ERROR` | Error searching FAISS index |
| `NORMALIZATION_ERROR` | Error applying normalization rules |
| `REVIEW_ITEM_NOT_FOUND` | Review item not found |
| `INVALID_DECISION` | Invalid review decision |

## Changelog

### Version 1.0.0
- Initial release
- Basic validation API
- Pattern learning and profiling
- Active learning system
- FAISS indexing support
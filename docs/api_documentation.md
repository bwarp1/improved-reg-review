# API Documentation

## Overview

The Regulatory Compliance Analysis Tool provides a Python API for integration into custom workflows and applications. This document details the available classes and methods.

## Core Components

### RegulationClient

The `RegulationClient` class handles communication with the regulations.gov API.

```python
from compliance_poc.clients import RegulationClient

# Initialize the client
reg_client = RegulationClient(api_key="your-api-key")

# Fetch a regulation document
regulation = reg_client.get_regulation("EPA-HQ-OAR-2021-0257")

# Search for regulations by keyword
results = reg_client.search_regulations(keyword="emissions")
```

#### Methods

| Method | Parameters | Return Type | Description |
|--------|------------|-------------|-------------|
| `get_regulation(docket_id)` | docket_id (str) | Regulation | Retrieves a specific regulation by docket ID |
| `search_regulations(keyword, start_date, end_date)` | keyword (str), start_date (str, optional), end_date (str, optional) | List[Regulation] | Searches for regulations matching criteria |

### DocumentParser

The `DocumentParser` class extracts text and structured information from documents.

```python
from compliance_poc.parsers import DocumentParser

# Initialize the parser
parser = DocumentParser()

# Parse a document
document = parser.parse("path/to/document.pdf")

# Extract requirements
requirements = parser.extract_requirements(document)
```

#### Methods

| Method | Parameters | Return Type | Description |
|--------|------------|-------------|-------------|
| `parse(file_path)` | file_path (str) | Document | Parses a document and returns structured content |
| `extract_requirements(document)` | document (Document) | List[Requirement] | Identifies and extracts regulatory requirements |

### ComplianceAnalyzer

The `ComplianceAnalyzer` class compares regulations against policy documents.

```python
from compliance_poc.analyzers import ComplianceAnalyzer

# Initialize the analyzer
analyzer = ComplianceAnalyzer()

# Analyze compliance
results = analyzer.analyze(regulation, policy_documents)

# Generate report
report = analyzer.generate_report(results, format="html")
```

## Data Models

### Regulation

Represents a regulatory document with extracted requirements.

```python
class Regulation:
    id: str                   # Docket ID
    title: str                # Regulation title
    agency: str               # Issuing agency
    publication_date: str     # Date published
    effective_date: str       # Date effective
    requirements: List[Requirement]  # Extracted requirements
```

### Requirement

Represents a specific regulatory requirement.

```python
class Requirement:
    id: str                   # Unique identifier
    text: str                 # Requirement text
    section: str              # Document section
    category: str             # Requirement category
    importance: float         # Importance score (0.0-1.0)
```

## Integration Examples

### Batch Processing Example

```python
from compliance_poc.clients import RegulationClient
from compliance_poc.parsers import DocumentParser
from compliance_poc.analyzers import ComplianceAnalyzer

def batch_analyze(dockets, policy_dir):
    client = RegulationClient()
    parser = DocumentParser()
    analyzer = ComplianceAnalyzer()
    
    # Load policies
    policies = []
    for policy_file in os.listdir(policy_dir):
        policy = parser.parse(os.path.join(policy_dir, policy_file))
        policies.append(policy)
    
    # Process each docket
    results = {}
    for docket_id in dockets:
        regulation = client.get_regulation(docket_id)
        results[docket_id] = analyzer.analyze(regulation, policies)
    
    return results
```

## Error Handling

The API uses custom exceptions for better error handling:

```python
from compliance_poc.exceptions import RegulationNotFound, APIRateLimitExceeded

try:
    regulation = client.get_regulation("INVALID-ID")
except RegulationNotFound:
    print("Regulation not found")
except APIRateLimitExceeded:
    print("API rate limit exceeded. Try again later.")
```

# User Guide

## Overview

The Regulatory Compliance Analysis Tool helps organizations ensure their internal policies align with regulatory requirements. This guide will walk you through all aspects of using the tool effectively.

## Getting Started

### Prerequisites
- Python 3.8 or higher
- API key from api.data.gov
- Your organization's policy documents in text, PDF, or DOCX format

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-org/improved-reg-review.git
   cd improved-reg-review
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

## Basic Usage

### Command Line Interface

The basic command structure is:
```bash
python run.py --docket [DOCKET_ID] --policy-dir [POLICY_DIRECTORY]
```

#### Required Parameters:
- `--docket`: The regulations.gov docket ID to analyze (e.g., EPA-HQ-OAR-2021-0257)
- `--policy-dir`: Directory containing your company policy documents

#### Optional Parameters:
- `--output-format`: Specify output format (csv, html, json, all). Default: all
- `--output-dir`: Directory where reports will be saved. Default: ./reports
- `--threshold`: Similarity threshold (0.0-1.0). Default: 0.75

### Web Interface

For a more user-friendly experience:
```bash
streamlit run app.py
```

The web interface allows you to:
1. Upload policy documents directly
2. Search for regulations by keyword or docket ID
3. Visualize compliance gaps with interactive charts
4. Export results in various formats

## Advanced Features

### Custom Analyzers

Create specialized analyzers for specific regulatory domains:

```python
from compliance_poc.analyzers import BaseAnalyzer

class FinancialAnalyzer(BaseAnalyzer):
    def __init__(self):
        super().__init__(domain="financial")
        
    # Override analysis methods
```

### Batch Processing

Process multiple regulations against your policies:

```bash
python batch_run.py --config batch_config.yaml
```

## Troubleshooting

### Common Issues

1. **API Rate Limits**: If you encounter rate limits, adjust the `request_delay` in config.yaml
2. **Document Parsing Errors**: Ensure your documents follow standard formatting
3. **Memory Issues**: For large regulations, use the `--chunk-size` parameter

### Getting Help

For additional assistance:
1. Check the [FAQ](/docs/faq.md)
2. File an issue on our GitHub repository

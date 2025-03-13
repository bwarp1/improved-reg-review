# Regulatory Compliance Analysis Tool

This tool automatically extracts regulatory requirements from Regulations.gov 
and compares them against your organization's internal policy documents to 
identify compliance gaps.

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

1. Get an API key from api.data.gov
2. Update `compliance_poc/config/config.yaml` with your API key
3. Organize your policy documents in the `company_policies` directory

## Usage

### Command Line

```bash
python run.py --docket EPA-HQ-OAR-2021-0257 --policy-dir ./company_policies
```

### Web Interface

```bash
streamlit run app.py
```

## Output

The tool generates reports in multiple formats:
- CSV for spreadsheet analysis
- HTML for interactive viewing
- JSON for integration with other systems

## Documentation

Comprehensive documentation is available in the `/docs` directory:

### User Documentation
- [User Guide](/docs/user_guide.md) - Complete instructions for using the tool
- [Configuration Guide](/docs/configuration.md) - Detailed configuration options
- [FAQ](/docs/faq.md) - Frequently asked questions

### Developer Documentation
- [API Documentation](/docs/api_documentation.md) - Reference for integrating with the tool
- [Contributing Guidelines](/docs/contributing.md) - How to contribute to the project
- [Architecture](/docs/architecture.md) - System design and components

### Regulatory Knowledge
- [Regulatory Interpretation Guide](/docs/regulatory_interpretation.md) - Understanding compliance requirements
- [Best Practices](/docs/compliance_best_practices.md) - Industry best practices

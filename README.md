# Regulatory Compliance Analysis Tool

This tool automatically extracts regulatory requirements from Regulations.gov 
and compares them against your organization's internal policy documents to 
identify compliance gaps.

## Installation

```bash
# Clone the repository (if not already done)
git clone https://github.com/your-org/improved-reg-review.git
cd improved-reg-review

# Create and activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Configuration

1. Obtain an API key from api.data.gov.
2. Update the `api_key` field in the `config.yaml` file (located in the project's root directory) with your key.
3. Organize your organization's policy documents (text, PDF, DOCX) into a directory. We recommend naming this directory `organization_policies`. You may need to create it. The `sample_data/policies` directory provides examples of how to structure this.

## Usage

### Command Line

```bash
python run.py --docket [DOCKET_ID] --policy-dir ./organization_policies
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
- [User Guide](/docs/user_guide.md) - For complete usage instructions.
- [Installation and Configuration Guide](/docs/installation_guide.md) - For detailed setup.

### Developer Documentation
- [API Documentation](/docs/api_documentation.md) - For integration.
- [Contributing Guidelines](/docs/contributing.md) - For development contributions.

### Regulatory Knowledge
- [Regulatory Interpretation Guide](/docs/regulatory_interpretation.md) - For understanding compliance.

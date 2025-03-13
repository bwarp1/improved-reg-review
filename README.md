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

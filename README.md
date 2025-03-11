# AI-Driven Regulatory Compliance Analysis PoC

This proof-of-concept (PoC) tool automatically extracts regulatory requirements from Regulations.gov and compares them against internal policy documents to identify compliance gaps.

## Features

- Automatically fetches regulatory documents from Regulations.gov API
- Extracts regulatory obligations using NLP techniques
- Compares regulatory requirements against internal policies
- Generates structured compliance gap analysis reports

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Configure your API key in `compliance_poc/config/config.yaml`
4. Run the application:
   ```
   python run.py --regulation "data privacy" --policy-dir "./policies"
   ```

## Configuration

Update `compliance_poc/config/config.yaml` with your API key and other settings:

```yaml
api:
  key: "YOUR_API_KEY_HERE"
  base_url: "https://api.regulations.gov/v4"
  rate_limit: 1000  # requests per hour

nlp:
  model: "en_core_web_sm"
  obligation_keywords: ["must", "shall", "required", "requirement", "mandate", "obligation"]
  threshold: 0.75  # similarity threshold for matching
```

## Usage

```bash
# Run the complete pipeline
python run.py --regulation "privacy" --policy-dir "./policies"

# Specify output format
python run.py --regulation "2023-12345" --policy-dir "./policies" --output-format html

# Use a specific docket ID
python run.py --docket-id "EPA-HQ-OAR-2021-0317" --policy-dir "./policies"
```

## Project Structure

- `compliance_poc/`: Main package directory
  - `config/`: Configuration files
  - `src/`: Source code
    - `api/`: Regulations.gov API integration
    - `nlp/`: Natural language processing for obligation extraction
    - `policy/`: Internal policy document handling
    - `matching/`: Comparison and gap analysis
    - `reporting/`: Report generation
  - `tests/`: Unit and integration tests

## License

MIT

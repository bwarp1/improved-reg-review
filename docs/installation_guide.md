# Installation Guide

This guide walks you through installing the Regulatory Compliance Analyzer and configuring it to work with your organization's policy library.

## Prerequisites

- Python 3.9+ installed
- pip package manager
- Git (for cloning the repository)
- 4GB+ RAM recommended for processing large documents
- Internet access (for API calls to Regulations.gov)

## Step 1: Install the Application

### Option A: Direct Installation

```bash
# Clone the repository
git clone https://github.com/your-org/improved-reg-review.git
cd improved-reg-review

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download required NLP models
python -m spacy download en_core_web_sm
```

### Option B: Docker Installation

```bash
# Clone the repository
git clone https://github.com/your-org/improved-reg-review.git
cd improved-reg-review

# Build Docker image
docker build -t reg-compliance-analyzer .

# Run the container
docker run -p 8501:8501 -v /path/to/your/policies:/app/company_policies reg-compliance-analyzer
```

## Step 2: Configure API Access

1. Register for an API key at [api.data.gov](https://api.data.gov/)
2. Create a copy of the example configuration:
   ```bash
   cp compliance_poc/config/config.example.yaml compliance_poc/config/config.yaml
   ```
3. Edit the configuration file:
   ```yaml
   api:
     base_url: "https://api.regulations.gov/v4"
     api_key: "YOUR_API_KEY_HERE"  # Replace with your key
     request_delay: 1.0  # Time between requests in seconds
   ```

## Step 3: Set Up Your Policy Library

### Directory Structure

Create a directory structure for your policies:

```bash
mkdir -p company_policies/{general,security,privacy,compliance}
```

### Supported File Formats

The system supports these policy document formats:
- PDF (.pdf)
- Microsoft Word (.docx)
- Plain Text (.txt)
- HTML (.html)

### Policy Library Configuration

Edit the `compliance_poc/config/config.yaml` file to specify your policy library location:

```yaml
policy_library:
  location: "company_policies"  # Default location
  index_file: "policy_index.json"  # Metadata index
  formats: ["pdf", "docx", "txt", "html"]
```

### Optional: Add Policy Metadata

Create a JSON file with metadata about your policies for better matching:

```bash
cat > company_policies/policy_metadata.json << EOF
{
  "data-retention-policy.pdf": {
    "title": "Data Retention Policy",
    "effective_date": "2024-01-01",
    "owner": "Legal Department",
    "tags": ["data", "retention", "privacy"]
  },
  "security-training.docx": {
    "title": "Security Training Policy",
    "effective_date": "2023-11-15",
    "owner": "Information Security",
    "tags": ["security", "training", "awareness"]
  }
}
EOF
```

## Step 4: Configure the Scheduled Checks

If you want to run automated checks for new regulations:

1. Edit the scheduler configuration in `compliance_poc/config/config.yaml`:
   ```yaml
   scheduler:
     agencies: ["EPA", "HHS", "SEC"]  # Agencies to monitor
     document_types: ["Rule", "Notice"]  # Document types to check
     check_frequency: "daily"  # Options: daily, weekly
     notification:
       email_enabled: false
       email_recipients: []
   ```

2. Set up a scheduled task:
   - **Linux/Mac**: Add to crontab
     ```bash
     crontab -e
     # Add: 0 8 * * * cd /path/to/improved-reg-review && ./scheduled_check.py
     ```
   - **Windows**: Use Task Scheduler
     ```
     Create a task that runs scheduled_check.py daily
     ```

## Step 5: Verify Installation

Run the verification script to check that everything is working:

```bash
python verify_installation.py
```

This will:
- Test API connectivity
- Validate your policy library setup
- Verify NLP components are working
- Check output directory permissions

## Using Your Own Policy Library

### Method 1: Direct Copy

Place your policy files directly in the appropriate subdirectories of `company_policies/`:

```bash
cp /path/to/your/policies/*.pdf company_policies/
```

### Method 2: Symbolic Link

Link to your existing policy repository:

```bash
# Remove the default directory
rm -rf company_policies

# Create symbolic link to your policy repository
ln -s /path/to/your/policy/repository company_policies
```

### Method 3: Volume Mount (Docker)

When using Docker, mount your policy directory as a volume:

```bash
docker run -p 8501:8501 -v /path/to/your/policies:/app/company_policies reg-compliance-analyzer
```

## Troubleshooting

### Common Installation Issues

1. **Missing Dependencies**:
   ```bash
   pip install -r requirements.txt --no-cache-dir
   ```

2. **spaCy Model Issues**:
   ```bash
   python -m spacy validate
   ```

3. **Permission Errors**:
   ```bash
   chmod -R 755 company_policies
   ```

4. **API Connection Failures**:
   Check your API key and internet connection:
   ```bash
   python -m compliance_poc.src.api.test_connection
   ```

## Next Steps

Once installed, proceed to the [User Guide](user_guide.md) for instructions on:
- Running compliance analyses
- Interpreting results
- Customizing matching algorithms
- Setting up automated monitoring

## Support

For additional assistance:
- Submit issues via GitHub
- Contact support@example.com
- Join our Slack community at example.slack.com
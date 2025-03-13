import streamlit as st
import pandas as pd
import numpy as np
import os
import logging
import yaml
from datetime import datetime
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

from compliance_poc.src.main import process_regulation

# Configure page
st.set_page_config(
    page_title="Regulatory Compliance Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from YAML file."""
    config_path = Path("compliance_poc/config/config.yaml")
    if config_path.exists():
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}

# Load configuration
config = load_config()

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
    }
    .compliance-high {
        color: #4CAF50;
        font-weight: bold;
    }
    .compliance-medium {
        color: #FF9800;
        font-weight: bold;
    }
    .compliance-low {
        color: #F44336;
        font-weight: bold;
    }
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">Regulatory Compliance Analyzer</p>', unsafe_allow_html=True)
st.markdown("""
This tool analyzes regulatory documents from Regulations.gov and compares them 
against your organization's internal policies to identify compliance gaps.
""")

# Check for API key in environment
api_key = os.environ.get("REGULATIONS_API_KEY")
if not api_key:
    api_key = config.get("api", {}).get("key")

# Show API key status
with st.sidebar:
    st.sidebar.header("Configuration")
    
    if api_key:
        st.success("✅ API Key detected")
        # Set up the mode selection option
        use_demo_data = st.checkbox("Use sample data", value=False)
        if use_demo_data:
            st.info("Using sample data instead of real API calls")
    else:
        st.warning("⚠️ No API Key found - will use sample data")
        use_demo_data = True
        st.info("""
        To use real regulations, set the REGULATIONS_API_KEY environment variable 
        or update the config.yaml file.
        """)
    
    # Input fields
    st.subheader("Regulation Source")
    docket_id = st.text_input("Docket ID", value="EPA-HQ-OAR-2021-0257")
    document_id = st.text_input("Document ID (Optional)")
    
    st.subheader("Company Policies")
    policy_dir = st.text_input("Policy Directory", value="sample_data/policies")
    
    # Run analysis button
    run_analysis = st.button("Run Analysis", type="primary")
    
    # About section
    st.sidebar.markdown("---")
    st.sidebar.header("About")
    st.sidebar.info("""
    This tool uses NLP to extract obligations from regulations and 
    match them against company policies.
    
    For help or issues, contact your administrator.
    """)

# Main content area - show this when no analysis has been run yet
if "results" not in st.session_state and not run_analysis:
    st.markdown('<p class="sub-header">How It Works</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 1️⃣ Extract")
        st.markdown("""
        The tool extracts regulatory requirements from official documents using NLP.
        It identifies obligations (must, shall, required) and their context.
        """)
    
    with col2:
        st.markdown("### 2️⃣ Compare")
        st.markdown("""
        The extracted requirements are compared against your internal policies
        using semantic similarity and keyword matching.
        """)
    
    with col3:
        st.markdown("### 3️⃣ Report")
        st.markdown("""
        Results are displayed showing compliance status, gaps, and 
        recommendations for policy updates.
        """)

# Process regulations when requested
if run_analysis:
    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    
    with st.spinner("Processing regulation..."):
        status_placeholder.text("Initializing...")
        progress_bar = progress_placeholder.progress(0)
        
        # Processing stages
        stages = ["Fetching regulation", "Extracting obligations", "Loading policies", 
                  "Comparing against policies", "Generating report"]
        
        for i, stage in enumerate(stages):
            # Update progress
            progress_value = (i / len(stages))
            progress_bar.progress(progress_value)
            status_placeholder.text(f"Step {i+1}/{len(stages)}: {stage}")
            
            # Add a slight delay to show progress visually
            import time
            time.sleep(0.5)
        
        # Update the config to use demo data if selected
        if "api" not in config:
            config["api"] = {}
        config["api"]["use_demo_data"] = use_demo_data
        
        # Process the regulation
        results = process_regulation(docket_id, document_id, policy_dir)
        
        # Complete the progress bar
        progress_bar.progress(1.0)
        status_placeholder.text("Analysis complete!")
        st.session_state.results = results
        st.session_state.processed_time = datetime.now()
        
        # Remove progress elements after completion
        time.sleep(1)
        progress_placeholder.empty()
        status_placeholder.empty()

# Display results if available
if "results" in st.session_state:
    results = st.session_state.results
    
    # Check for errors
    if "error" in results:
        st.error(f"Analysis error: {results['error']}")
    else:
        # Show summary statistics
        obligations = results.get("obligations", [])
        policies = results.get("policies", {})
        matches = results.get("matches", [])
        
        st.markdown('<p class="sub-header">Analysis Results</p>', unsafe_allow_html=True)
        
        # Date and time of analysis
        st.markdown(f"Analysis completed on: {st.session_state.processed_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Top-level metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Obligations Analyzed", len(obligations))
        
        with col2:
            st.metric("Policies Reviewed", len(policies))
        
        # Calculate compliance metrics
        if matches:
            compliant = sum(1 for m in matches if m.get("status") == "Compliant")
            partial = sum(1 for m in matches if m.get("status") == "Partial")
            non_compliant = sum(1 for m in matches if m.get("status") == "Non-Compliant")
            
            with col3:
                overall_compliance = (compliant + (partial * 0.5)) / len(matches) * 100 if matches else 0
                st.metric("Overall Compliance", f"{overall_compliance:.1f}%")
        
            # Visualization
            st.markdown("### Compliance Distribution")
            
            # Create compliance chart
            fig = px.pie(
                values=[compliant, partial, non_compliant],
                names=["Compliant", "Partial", "Non-Compliant"],
                color=["Compliant", "Partial", "Non-Compliant"],
                color_discrete_map={"Compliant": "#4CAF50", "Partial": "#FF9800", "Non-Compliant": "#F44336"},
                hole=0.4
            )
            
            fig.update_layout(
                margin=dict(l=20, r=20, t=30, b=20),
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed results
            st.markdown("### Detailed Results")
            
            # Create a DataFrame for the results
            results_data = []
            for match in matches:
                results_data.append({
                    "Obligation ID": match.get("obligation_id", ""),
                    "Requirement Text": match.get("requirement", "")[:200] + "...",
                    "Status": match.get("status", "Unknown"),
                    "Compliance Score": match.get("score", 0),
                    "Matching Policy": match.get("policy", "None"),
                    "Policy Text": match.get("policy_text", "None")[:200] + "..." if match.get("policy_text") else "None"
                })
            
            df = pd.DataFrame(results_data)
            
            # Add color to the Status column
            def color_status(val):
                if val == "Compliant":
                    return 'background-color: #C8E6C9; color: #2E7D32'
                elif val == "Partial":
                    return 'background-color: #FFE0B2; color: #E65100'
                else:
                    return 'background-color: #FFCDD2; color: #B71C1C'
                
            # Filter options
            st.markdown("#### Filter Results")
            status_filter = st.multiselect(
                "Filter by Status:",
                ["Compliant", "Partial", "Non-Compliant"],
                default=["Compliant", "Partial", "Non-Compliant"]
            )
            
            # Apply filters
            filtered_df = df[df["Status"].isin(status_filter)]
            
            # Sort options
            sort_by = st.selectbox(
                "Sort by:",
                ["Compliance Score", "Obligation ID", "Status"],
                index=0
            )
            
            if sort_by == "Compliance Score":
                filtered_df = filtered_df.sort_values(by="Compliance Score", ascending=False)
            else:
                filtered_df = filtered_df.sort_values(by=sort_by)
            
            # Display the styled dataframe
            st.dataframe(
                filtered_df.style.applymap(color_status, subset=["Status"]), 
                height=500,
                use_container_width=True
            )
            
            # Download options
            col1, col2 = st.columns(2)
            
            with col1:
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download as CSV",
                    data=csv,
                    file_name=f"compliance_results_{docket_id}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            
            # Gap analysis
            st.markdown("### Compliance Gaps")
            st.markdown("The following obligations have the lowest compliance scores and require attention:")
            
            # Get the bottom 5 compliance scores
            gap_df = df.sort_values(by="Compliance Score").head(5)
            
            for _, row in gap_df.iterrows():
                with st.expander(f"{row['Obligation ID']}: {row['Requirement Text']}"):
                    st.markdown(f"**Status:** {row['Status']}")
                    st.markdown(f"**Compliance Score:** {row['Compliance Score']:.2f}")
                    st.markdown("**Recommendation:**")
                    
                    if row['Status'] == "Non-Compliant":
                        st.markdown("⚠️ Create a new policy to address this obligation")
                    else:
                        st.markdown("⚠️ Update existing policy to better address this obligation")
                        
                    st.markdown(f"**Closest matching policy:** {row['Matching Policy']}")
                    if row['Policy Text'] != "None":
                        st.markdown(f"**Policy text:** {row['Policy Text']}")

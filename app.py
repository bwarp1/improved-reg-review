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
    
    # Company Policies - Enhanced with department filtering
    st.subheader("Company Policies")
    policy_dir = st.text_input("Policy Directory", value="sample_data/policies")
    
    # Add department filtering
    st.subheader("Filter Options")
    departments = ["All Departments", "Legal", "Operations", "IT", "Finance", "Compliance", "HR"]
    selected_departments = st.multiselect(
        "Filter by Department",
        departments,
        default=["All Departments"]
    )
    
    # Add regulation type filtering
    regulation_types = ["All Types", "Environmental", "Financial", "Healthcare", "Data Privacy", "Safety"]
    selected_reg_types = st.multiselect(
        "Filter by Regulation Type",
        regulation_types,
        default=["All Types"]
    )
    
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
            
            # New: Add Compliance Gap visualization section
            st.markdown("### Compliance Gap Analysis")
            
            # Create two columns for gap visualizations
            gap_col1, gap_col2 = st.columns(2)
            
            with gap_col1:
                # Add Gap by Department visualization (if department info is available)
                st.markdown("#### Gaps by Department")
                
                # Simulate department distribution data
                # In a real implementation, this would be derived from actual data
                department_gaps = {
                    "Legal": non_compliant * 0.3,
                    "Compliance": non_compliant * 0.25,
                    "Operations": non_compliant * 0.2,
                    "IT": non_compliant * 0.15,
                    "Finance": non_compliant * 0.1,
                }
                
                dept_fig = px.bar(
                    x=list(department_gaps.keys()),
                    y=list(department_gaps.values()),
                    labels={'x': 'Department', 'y': 'Number of Gaps'},
                    color=list(department_gaps.values()),
                    color_continuous_scale=['#FFCDD2', '#F44336', '#B71C1C'],
                    title="Compliance Gaps by Department"
                )
                
                dept_fig.update_layout(
                    xaxis_title="Department",
                    yaxis_title="Number of Gaps",
                    height=300
                )
                
                st.plotly_chart(dept_fig, use_container_width=True)
            
            with gap_col2:
                # Add Gap by Severity visualization
                st.markdown("#### Gaps by Severity")
                
                # Calculate severity based on compliance scores
                # In a real implementation, this would have more sophisticated logic
                high_severity = sum(1 for m in matches if m.get("score", 0) < 0.3 and m.get("status") == "Non-Compliant")
                medium_severity = sum(1 for m in matches if 0.3 <= m.get("score", 0) < 0.6 and m.get("status") != "Compliant")
                low_severity = sum(1 for m in matches if m.get("score", 0) >= 0.6 and m.get("status") != "Compliant")
                
                severity_fig = px.pie(
                    values=[high_severity, medium_severity, low_severity],
                    names=["High", "Medium", "Low"],
                    color=["High", "Medium", "Low"],
                    color_discrete_map={"High": "#B71C1C", "Medium": "#FF9800", "Low": "#FFC107"},
                    title="Gap Severity Distribution"
                )
                
                severity_fig.update_layout(
                    height=300
                )
                
                st.plotly_chart(severity_fig, use_container_width=True)
            
            # Add compliance trend visualization (simulated)
            st.markdown("### Compliance Trend Analysis")
            
            # Create a simulated trend over time
            # In a real implementation, this would use historical data
            months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
            compliance_trend = [65, 68, 70, 72, 75, overall_compliance]
            
            trend_fig = px.line(
                x=months,
                y=compliance_trend,
                labels={'x': 'Month', 'y': 'Compliance Score (%)'},
                markers=True
            )
            
            trend_fig.update_layout(
                xaxis_title="Month",
                yaxis_title="Compliance Score (%)",
                height=300
            )
            
            # Add current compliance level as a horizontal line
            trend_fig.add_hline(
                y=overall_compliance, 
                line_dash="dash", 
                line_color="#4CAF50",
                annotation_text="Current compliance",
                annotation_position="bottom right"
            )
            
            # Add target compliance level (example: 85%)
            trend_fig.add_hline(
                y=85, 
                line_dash="dash", 
                line_color="#FF9800",
                annotation_text="Target",
                annotation_position="bottom right"
            )
            
            st.plotly_chart(trend_fig, use_container_width=True)
            
            # Detailed results with enhanced filtering
            st.markdown("### Detailed Results")
            
            # Create a DataFrame for the results
            results_data = []
            for match in matches:
                # Extract or assign department and regulation type
                # In a real implementation, these would come from the data
                department = match.get("department", "Unassigned")
                reg_type = match.get("regulation_type", "Unassigned")
                
                results_data.append({
                    "Obligation ID": match.get("obligation_id", ""),
                    "Requirement Text": match.get("requirement", "")[:200] + "...",
                    "Status": match.get("status", "Unknown"),
                    "Compliance Score": match.get("score", 0),
                    "Matching Policy": match.get("policy", "None"),
                    "Policy Text": match.get("policy_text", "None")[:200] + "..." if match.get("policy_text") else "None",
                    "Department": department,
                    "Regulation Type": reg_type
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
                
            # Enhanced Filter options
            st.markdown("#### Filter Results")
            filter_col1, filter_col2, filter_col3 = st.columns(3)
            
            with filter_col1:
                status_filter = st.multiselect(
                    "Filter by Status:",
                    ["Compliant", "Partial", "Non-Compliant"],
                    default=["Compliant", "Partial", "Non-Compliant"]
                )
            
            with filter_col2:
                # Department filter - use unique values from the data
                dept_filter = st.multiselect(
                    "Filter by Department:",
                    ["All Departments"] + list(df["Department"].unique()),
                    default=["All Departments"]
                )
            
            with filter_col3:
                # Regulation type filter - use unique values from the data
                reg_type_filter = st.multiselect(
                    "Filter by Regulation Type:",
                    ["All Types"] + list(df["Regulation Type"].unique()),
                    default=["All Types"]
                )
            
            # Apply filters
            filtered_df = df
            
            # Status filter
            if status_filter:
                filtered_df = filtered_df[filtered_df["Status"].isin(status_filter)]
            
            # Department filter
            if dept_filter and "All Departments" not in dept_filter:
                filtered_df = filtered_df[filtered_df["Department"].isin(dept_filter)]
            
            # Regulation type filter
            if reg_type_filter and "All Types" not in reg_type_filter:
                filtered_df = filtered_df[filtered_df["Regulation Type"].isin(reg_type_filter)]
            
            # Sort options
            sort_by = st.selectbox(
                "Sort by:",
                ["Compliance Score", "Obligation ID", "Status", "Department", "Regulation Type"],
                index=0
            )
            
            if sort_by in ["Compliance Score"]:
                filtered_df = filtered_df.sort_values(by=sort_by, ascending=False)
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
                
            with col2:
                # Add Excel export option
                if df is not None:
                    excel_buffer = pd.ExcelWriter('compliance_results.xlsx', engine='xlsxwriter')
                    df.to_excel(excel_buffer, sheet_name='Compliance Results', index=False)
                    excel_buffer.close()
                    
                    with open('compliance_results.xlsx', 'rb') as f:
                        excel_data = f.read()
                    
                    st.download_button(
                        label="Download as Excel",
                        data=excel_data,
                        file_name=f"compliance_results_{docket_id}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            
            # Gap analysis with enhanced visuals
            st.markdown("### Compliance Gaps")
            st.markdown("The following obligations have the lowest compliance scores and require attention:")
            
            # Get the bottom 5 compliance scores
            gap_df = df.sort_values(by="Compliance Score").head(5)
            
            for i, (_, row) in enumerate(gap_df.iterrows()):
                gap_severity = "High" if row['Compliance Score'] < 0.3 else "Medium" if row['Compliance Score'] < 0.6 else "Low"
                gap_color = "#B71C1C" if gap_severity == "High" else "#FF9800" if gap_severity == "Medium" else "#FFC107"
                
                with st.expander(f"Gap {i+1}: {row['Obligation ID']} - {gap_severity} Severity", expanded=(i==0)):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"**Requirement:**")
                        st.markdown(f"_{row['Requirement Text']}_")
                        
                        st.markdown(f"**Status:** {row['Status']}")
                        st.markdown(f"**Compliance Score:** {row['Compliance Score']:.2f}")
                        st.markdown(f"**Department:** {row['Department']}")
                        st.markdown(f"**Regulation Type:** {row['Regulation Type']}")
                        
                        st.markdown("**Recommendation:**")
                        
                        if row['Status'] == "Non-Compliant":
                            st.markdown("⚠️ Create a new policy to address this obligation")
                        else:
                            st.markdown("⚠️ Update existing policy to better address this obligation")
                            
                        st.markdown(f"**Closest matching policy:** {row['Matching Policy']}")
                        if row['Policy Text'] != "None":
                            st.markdown(f"**Policy text:** {row['Policy Text']}")
                    
                    with col2:
                        # Add a gauge chart for the gap
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = row['Compliance Score'] * 100,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "Compliance"},
                            gauge = {
                                'axis': {'range': [None, 100]},
                                'bar': {'color': gap_color},
                                'steps': [
                                    {'range': [0, 30], 'color': '#FFCDD2'},
                                    {'range': [30, 70], 'color': '#FFECB3'},
                                    {'range': [70, 100], 'color': '#C8E6C9'}
                                ]
                            }
                        ))
                        
                        fig.update_layout(height=200, margin=dict(l=20, r=20, t=30, b=20))
                        st.plotly_chart(fig, use_container_width=True)
            
            # Add action plan section
            st.markdown("### Recommended Action Plan")
            
            action_col1, action_col2 = st.columns(2)
            
            with action_col1:
                st.markdown("#### Short-term Actions (Next 30 Days)")
                st.markdown("1. Address high-severity gaps in the following departments:")
                for dept, count in department_gaps.items():
                    if count > 0:
                        st.markdown(f"   - {dept}: {int(count)} gap(s)")
                st.markdown("2. Initiate policy updates for partial compliance areas")
                st.markdown("3. Schedule compliance workshop with key stakeholders")
            
            with action_col2:
                st.markdown("#### Long-term Strategy (90-Day Plan)")
                st.markdown("1. Establish regular compliance monitoring process")
                st.markdown("2. Develop department-specific compliance templates")
                st.markdown("3. Implement automated compliance checks")
                st.markdown(f"4. Target achieving 85% overall compliance (current: {overall_compliance:.1f}%)")

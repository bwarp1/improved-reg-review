"""
Module for generating compliance reports in various formats.
"""

import csv
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import pandas as pd  # Add import for pandas
import plotly.graph_objects as go  # Add import for plotly
import plotly.express as px
from plotly.subplots import make_subplots

class ReportGenerator:
    """Generate compliance analysis reports in various formats."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the report generator.

        Args:
            config: Output configuration dictionary
        """
        self.config = config
        self.output_dir = Path(config.get("directory", "compliance_poc/reports"))
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.logger = logging.getLogger(__name__)
        
        # Register report generators (OCP principle)
        self.report_generators = {
            "console": self._generate_console_report,
            "csv": self._generate_csv_report,
            "json": self._generate_json_report,
            "html": self._generate_html_report,
            "excel": self.generate_excel_report,  # Add Excel report to generators
            "dashboard": self.generate_html_dashboard  # Add dashboard to generators
        }
    
    def generate_report(self, compliance_matrix: Dict[str, Any]) -> Path:
        """
        Generate a compliance report based on the compliance matrix.

        Args:
            compliance_matrix: Dictionary containing compliance analysis results

        Returns:
            Path to the generated report or directory containing reports
        """
        output_format = self.config.get("format", "console")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Extract document ID or regulation ID for filename
        regulation_id = "unknown"
        if "metadata" in compliance_matrix and "regulation_id" in compliance_matrix["metadata"]:
            regulation_id = compliance_matrix["metadata"]["regulation_id"]
        elif "document_id" in compliance_matrix:
            regulation_id = compliance_matrix["document_id"]
        elif "docket_id" in compliance_matrix:
            regulation_id = compliance_matrix["docket_id"]
        
        # Sanitize the ID for use in filenames
        regulation_id = regulation_id.replace('/', '-').replace(':', '_').replace(' ', '_')
        
        # Special case for "all" formats
        if output_format == "all":
            self.logger.info("Generating reports in all available formats")
            reports_dir = self.output_dir / f"compliance_report_{regulation_id}_{timestamp}"
            reports_dir.mkdir(exist_ok=True)
            
            for fmt, generator in self.report_generators.items():
                if fmt == "console":
                    generator(compliance_matrix)
                else:
                    filepath = reports_dir / f"{regulation_id}.{fmt}"
                    generator(compliance_matrix, filepath)
            
            self.logger.info(f"Reports generated in directory: {reports_dir}")
            return reports_dir
        
        # Single format generation
        if output_format in self.report_generators:
            if output_format == "console":
                return self.report_generators[output_format](compliance_matrix)
            else:
                filepath = self.output_dir / f"compliance_report_{regulation_id}_{timestamp}.{output_format}"
                return self.report_generators[output_format](compliance_matrix, filepath)
        
        # Fallback for unsupported format
        self.logger.warning(f"Unsupported output format: {output_format}, defaulting to console")
        return self._generate_console_report(compliance_matrix)

    def _generate_console_report(self, compliance_matrix: Dict[str, Any]) -> None:
        """Generate and print a report to the console."""
        print("\n=== COMPLIANCE ANALYSIS REPORT ===\n")
        
        # Print summary statistics
        print(f"Total obligations analyzed: {len(compliance_matrix.get('obligations', []))}")
        print(f"Total policies reviewed: {len(compliance_matrix.get('policies', []))}")
        
        # Compliance statistics
        compliant = sum(1 for o in compliance_matrix.get('matches', []) if o.get('compliance_level', 0) >= 0.8)
        partial = sum(1 for o in compliance_matrix.get('matches', []) if 0.3 <= o.get('compliance_level', 0) < 0.8)
        gaps = sum(1 for o in compliance_matrix.get('matches', []) if o.get('compliance_level', 0) < 0.3)
        
        print(f"Compliant obligations: {compliant}")
        print(f"Partially compliant obligations: {partial}")
        print(f"Non-compliant obligations (gaps): {gaps}")
        
        print("\n=== TOP COMPLIANCE GAPS ===\n")
        # Sort matches by compliance level (ascending) to show the biggest gaps first
        sorted_matches = sorted(
            compliance_matrix.get('matches', []),
            key=lambda x: x.get('compliance_level', 0)
        )
        
        # Show top 5 gaps
        for match in sorted_matches[:5]:
            obligation_id = match.get('obligation_id', 'Unknown')
            obligation_text = match.get('obligation_text', 'No text available')
            compliance_level = match.get('compliance_level', 0)
            
            print(f"Obligation ID: {obligation_id}")
            print(f"Compliance Level: {compliance_level:.2f}")
            print(f"Text: {obligation_text[:100]}...")
            print("-" * 50)
        
        self.logger.info("Console report generated")
        return None

    def _generate_csv_report(self, compliance_matrix: Dict[str, Any], filepath: Path) -> Path:
        """Generate a CSV report of the compliance analysis."""
        with open(filepath, 'w', newline='') as csvfile:
            fieldnames = [
                'obligation_id', 'obligation_text', 'compliance_level',
                'matching_policies', 'suggested_actions'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for match in compliance_matrix.get('matches', []):
                # Simplify the matching policies to just their IDs
                matching_policies = ', '.join([
                    p.get('policy_id', 'Unknown') 
                    for p in match.get('matching_policies', [])
                ])
                
                # Determine suggested actions based on compliance level
                compliance_level = match.get('compliance_level', 0)
                if compliance_level < 0.3:
                    action = "Create new policy to address this obligation"
                elif compliance_level < 0.8:
                    action = "Update existing policies to fully address this obligation"
                else:
                    action = "No action needed - fully compliant"
                
                writer.writerow({
                    'obligation_id': match.get('obligation_id', 'Unknown'),
                    'obligation_text': match.get('obligation_text', '')[:200],
                    'compliance_level': f"{compliance_level:.2f}",
                    'matching_policies': matching_policies,
                    'suggested_actions': action
                })
        
        self.logger.info(f"CSV report generated at: {filepath}")
        return filepath

    def _generate_json_report(self, compliance_matrix: Dict[str, Any], filepath: Path) -> Path:
        """Generate a JSON report of the compliance analysis."""
        # Add report metadata
        report_data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_obligations': len(compliance_matrix.get('obligations', [])),
                'total_policies': len(compliance_matrix.get('policies', [])),
            },
            'compliance_matrix': compliance_matrix
        }
        
        with open(filepath, 'w') as jsonfile:
            json.dump(report_data, jsonfile, indent=2)
        
        self.logger.info(f"JSON report generated at: {filepath}")
        return filepath

    def _generate_html_report(self, compliance_matrix: Dict[str, Any], filepath: Path) -> Path:
        """Generate an HTML report of the compliance analysis."""
        # Generate a simple but effective HTML report
        matches = compliance_matrix.get('matches', [])
        
        # Calculate summary statistics
        compliant = sum(1 for o in matches if o.get('compliance_level', 0) >= 0.8)
        partial = sum(1 for o in matches if 0.3 <= o.get('compliance_level', 0) < 0.8)
        gaps = sum(1 for o in matches if o.get('compliance_level', 0) < 0.3)
        
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Compliance Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1, h2 {{ color: #2c3e50; }}
        .summary {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; }}
        .stats {{ display: flex; margin: 20px 0; }}
        .stat-box {{ flex: 1; padding: 10px; border-radius: 5px; margin-right: 10px; text-align: center; }}
        .compliant {{ background-color: #d4edda; }}
        .partial {{ background-color: #fff3cd; }}
        .gap {{ background-color: #f8d7da; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f2f2f2; }}
        .progress-bar {{ width: 100px; background-color: #e9ecef; border-radius: 3px; }}
        .progress {{ height: 20px; border-radius: 3px; }}
        .high {{ background-color: #28a745; }}
        .medium {{ background-color: #ffc107; }}
        .low {{ background-color: #dc3545; }}
    </style>
</head>
<body>
    <h1>Compliance Analysis Report</h1>
    <div class="summary">
        <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        <p>Total obligations analyzed: {len(compliance_matrix.get('obligations', []))}</p>
        <p>Total policies reviewed: {len(compliance_matrix.get('policies', []))}</p>
    </div>
    
    <div class="stats">
        <div class="stat-box compliant">
            <h3>Compliant</h3>
            <p>{compliant}</p>
        </div>
        <div class="stat-box partial">
            <h3>Partially Compliant</h3>
            <p>{partial}</p>
        </div>
        <div class="stat-box gap">
            <h3>Non-Compliant (Gaps)</h3>
            <p>{gaps}</p>
        </div>
    </div>
    
    <h2>Compliance Details</h2>
    <table>
        <tr>
            <th>Obligation ID</th>
            <th>Compliance Level</th>
            <th>Obligation Text</th>
            <th>Matching Policies</th>
            <th>Suggested Action</th>
        </tr>
"""

        # Sort by compliance level to highlight gaps
        sorted_matches = sorted(matches, key=lambda x: x.get('compliance_level', 0))
        
        for match in sorted_matches:
            obligation_id = match.get('obligation_id', 'Unknown')
            obligation_text = match.get('obligation_text', 'No text available')[:150] + "..."
            compliance_level = match.get('compliance_level', 0)
            
            # Determine bar color and suggested action based on compliance level
            if compliance_level < 0.3:
                bar_class = "low"
                action = "Create new policy"
            elif compliance_level < 0.8:
                bar_class = "medium"
                action = "Update existing policies"
            else:
                bar_class = "high"
                action = "No action needed"
            
            # Format matching policies
            matching_policies = ", ".join([
                p.get('policy_id', 'Unknown') 
                for p in match.get('matching_policies', [])
            ]) or "None"
            
            html_content += f"""
        <tr>
            <td>{obligation_id}</td>
            <td>
                <div class="progress-bar">
                    <div class="progress {bar_class}" style="width: {compliance_level * 100}%"></div>
                </div>
                {compliance_level:.2f}
            </td>
            <td>{obligation_text}</td>
            <td>{matching_policies}</td>
            <td>{action}</td>
        </tr>"""
        
        html_content += """
    </table>
</body>
</html>
"""
        
        with open(filepath, 'w') as html_file:
            html_file.write(html_content)
        
        self.logger.info(f"HTML report generated at: {filepath}")
        return filepath

    def generate_excel_report(self, compliance_matrix: Dict[str, Any], filepath: Optional[Path] = None) -> Path:
        """
        Generate a detailed Excel report with filtering and sorting.
        
        Creates a workbook with multiple sheets for different views of the
        compliance data. Includes data validation and conditional formatting.
        
        Args:
            compliance_matrix: Dictionary containing compliance analysis results
            filepath: Output path for the Excel file
            
        Returns:
            Path to the generated Excel file
        """
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Extract regulation ID for filename
            regulation_id = "unknown"
            if "metadata" in compliance_matrix and "regulation_id" in compliance_matrix["metadata"]:
                regulation_id = compliance_matrix["metadata"]["regulation_id"]
            elif "document_id" in compliance_matrix:
                regulation_id = compliance_matrix["document_id"]
            elif "docket_id" in compliance_matrix:
                regulation_id = compliance_matrix["docket_id"]
            
            # Sanitize the ID for use in filenames
            regulation_id = regulation_id.replace('/', '-').replace(':', '_').replace(' ', '_')
            
            filepath = self.output_dir / f"compliance_report_{regulation_id}_{timestamp}.xlsx"
        
        self.logger.info(f"Generating Excel report at: {filepath}")
        
        # Create a Pandas Excel writer using openpyxl as the engine
        writer = pd.ExcelWriter(filepath, engine='openpyxl')
        
        # Convert compliance matches to DataFrame for easy manipulation
        matches_data = []
        for match in compliance_matrix.get('matches', []):
            compliance_level = match.get('compliance_level', 0)
            
            # Determine compliance status category
            if compliance_level >= 0.8:
                status = "Compliant"
            elif compliance_level >= 0.3:
                status = "Partially Compliant"
            else:
                status = "Non-Compliant"
                
            # Get matching policies information
            matching_policies = match.get('matching_policies', [])
            policy_ids = [p.get('policy_id', 'Unknown') for p in matching_policies]
            policy_scores = [p.get('match_score', 0) for p in matching_policies]
            
            # Create a row for the dataframe
            row = {
                'Obligation ID': match.get('obligation_id', 'Unknown'),
                'Compliance Level': compliance_level,
                'Status': status,
                'Obligation Text': match.get('obligation_text', '')[:500],  # Truncate long text
                'Matching Policies': ', '.join(policy_ids),
                'Policy Count': len(matching_policies),
                'Highest Match Score': max(policy_scores) if policy_scores else 0,
                'Category': match.get('category', 'Uncategorized'),
                'Last Updated': datetime.now().strftime("%Y-%m-%d")
            }
            matches_data.append(row)
        
        # Create the main DataFrame
        df = pd.DataFrame(matches_data)
        
        # 1. Summary Sheet
        summary_data = {
            'Metric': [
                'Total Obligations', 
                'Compliant', 
                'Partially Compliant', 
                'Non-Compliant',
                'Average Compliance Level',
                'Policies Reviewed'
            ],
            'Value': [
                len(compliance_matrix.get('obligations', [])),
                len(df[df['Status'] == 'Compliant']),
                len(df[df['Status'] == 'Partially Compliant']),
                len(df[df['Status'] == 'Non-Compliant']),
                df['Compliance Level'].mean(),
                len(compliance_matrix.get('policies', []))
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # 2. Main compliance report with all details
        df.to_excel(writer, sheet_name='Compliance Detail', index=False)
        
        # 3. Non-compliant items (filtered)
        if len(df[df['Status'] == 'Non-Compliant']) > 0:
            non_compliant_df = df[df['Status'] == 'Non-Compliant'].sort_values('Compliance Level')
            non_compliant_df.to_excel(writer, sheet_name='Compliance Gaps', index=False)
        
        # 4. Categories summary (pivot table)
        category_pivot = pd.pivot_table(
            df, 
            values='Obligation ID',
            index=['Category'], 
            columns=['Status'],
            aggfunc='count', 
            fill_value=0
        )
        category_pivot.to_excel(writer, sheet_name='Categories')
        
        # 5. Policy coverage sheet
        policy_data = []
        for policy in compliance_matrix.get('policies', []):
            policy_id = policy.get('policy_id', 'Unknown')
            covered_obligations = sum(1 for m in compliance_matrix.get('matches', []) 
                                    if any(p.get('policy_id') == policy_id 
                                           for p in m.get('matching_policies', [])))
            
            policy_data.append({
                'Policy ID': policy_id,
                'Policy Name': policy.get('title', 'Untitled'),
                'Covered Obligations': covered_obligations,
                'Coverage Ratio': covered_obligations / len(compliance_matrix.get('obligations', [])) 
                                  if compliance_matrix.get('obligations') else 0
            })
        
        if policy_data:
            policies_df = pd.DataFrame(policy_data)
            policies_df.to_excel(writer, sheet_name='Policy Coverage', index=False)
        
        # Save the Excel file
        writer.close()
        
        self.logger.info(f"Excel report generated at: {filepath}")
        return filepath
    
    def generate_html_dashboard(self, compliance_matrix: Dict[str, Any], filepath: Optional[Path] = None) -> Path:
        """
        Generate an interactive HTML dashboard with charts.
        
        Creates a self-contained, shareable HTML report optimized for distribution to
        stakeholders who don't have access to the Streamlit interface. Focuses on 
        static visualization and printable formats.
        
        Args:
            compliance_matrix: Dictionary containing compliance analysis results
            filepath: Output path for the HTML dashboard
            
        Returns:
            Path to the generated HTML dashboard
        """
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Extract regulation ID for filename
            regulation_id = "unknown"
            if "metadata" in compliance_matrix and "regulation_id" in compliance_matrix["metadata"]:
                regulation_id = compliance_matrix["metadata"]["regulation_id"]
            elif "document_id" in compliance_matrix:
                regulation_id = compliance_matrix["document_id"]
            elif "docket_id" in compliance_matrix:
                regulation_id = compliance_matrix["docket_id"]
            
            # Sanitize the ID for use in filenames
            regulation_id = regulation_id.replace('/', '-').replace(':', '_').replace(' ', '_')
            
            filepath = self.output_dir / f"compliance_dashboard_{regulation_id}_{timestamp}.html"
        
        self.logger.info(f"Generating HTML dashboard at: {filepath}")
        
        # Extract data from compliance matrix
        matches = compliance_matrix.get('matches', [])
        obligations = compliance_matrix.get('obligations', [])
        policies = compliance_matrix.get('policies', [])
        
        # Calculate compliance statistics
        total_obligations = len(obligations)
        compliant_count = sum(1 for m in matches if m.get('compliance_level', 0) >= 0.8)
        partial_count = sum(1 for m in matches if 0.3 <= m.get('compliance_level', 0) < 0.8)
        noncompliant_count = sum(1 for m in matches if m.get('compliance_level', 0) < 0.3)
        
        # Create a subplot with 2x2 grid - different from Streamlit for complementary view
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{"type": "indicator"}, {"type": "pie"}],
                   [{"type": "bar", "colspan": 2}, None]],
            subplot_titles=("Overall Compliance Score", 
                           "Compliance Status",
                           "Top Policy Coverage")
        )
        
        # 1. Gauge chart for overall compliance - different from Streamlit's metrics
        overall_score = (compliant_count + (0.5 * partial_count)) / total_obligations if total_obligations else 0
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=overall_score * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "#F44336"},
                        {'range': [30, 70], 'color': "#FF9800"},
                        {'range': [70, 100], 'color': "#4CAF50"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 80
                    }
                },
                title={'text': "Overall Score (%)"}
            ),
            row=1, col=1
        )
        
        # 2. Pie chart for compliance status distribution
        fig.add_trace(
            go.Pie(
                labels=["Compliant", "Partially Compliant", "Non-Compliant"],
                values=[compliant_count, partial_count, noncompliant_count],
                marker=dict(colors=['#4CAF50', '#FF9800', '#F44336']),
                textinfo='label+percent',
                hole=.3,
            ),
            row=1, col=2
        )
        
        # 3. Policy coverage bar chart - focusing on policy coverage which complements Streamlit's obligation focus
        policy_counts = {}
        for match in matches:
            for policy in match.get('matching_policies', []):
                policy_id = policy.get('policy_id', 'Unknown')
                policy_title = next((p.get('title', 'Untitled') for p in policies if p.get('policy_id') == policy_id), policy_id)
                policy_counts[policy_id] = {
                    'count': policy_counts.get(policy_id, {}).get('count', 0) + 1,
                    'title': policy_title
                }
        
        # Sort and get top policies
        top_policies = sorted(policy_counts.items(), key=lambda x: x[1]['count'], reverse=True)[:10]
        if top_policies:
            policy_ids = [f"{p[0]} ({p[1]['title'][:15]}...)" if len(p[1]['title']) > 15 else f"{p[0]} ({p[1]['title']})" for p in top_policies]
            policy_coverage = [p[1]['count']/total_obligations for p in top_policies]
            
            fig.add_trace(
                go.Bar(
                    x=policy_ids,
                    y=policy_coverage,
                    marker_color='rgba(26, 118, 255, 0.7)',
                    text=[f"{p*100:.1f}%" for p in policy_coverage],
                    textposition='auto',
                ),
                row=2, col=1
            )
        
        # Update layout
        fig.update_layout(
            title_text="Compliance Dashboard - Executive Summary",
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=800,
            template="plotly_white"
        )
        
        # Create regulatory gap table with printable format
        # Focus on gaps which complement Streamlit's full view
        gap_matches = sorted(matches, key=lambda x: x.get('compliance_level', 0))[:10]
        
        gap_table = go.Figure(
            data=[go.Table(
                header=dict(
                    values=['Obligation ID', 'Compliance Level', 'Gap Description', 'Recommended Action'],
                    fill_color='#e63946',
                    align='left',
                    font=dict(color='white', size=12)
                ),
                cells=dict(
                    values=[
                        [m.get('obligation_id', 'Unknown') for m in gap_matches],
                        [f"{m.get('compliance_level', 0):.2f}" for m in gap_matches],
                        [m.get('obligation_text', '')[:100] + "..." for m in gap_matches],
                        [self._get_action_recommendation(m.get('compliance_level', 0), m.get('matching_policies', [])) for m in gap_matches]
                    ],
                    fill_color=[['#f8f9fa', '#e9ecef'] * (len(gap_matches) + 1)],
                    align='left',
                    font=dict(size=11),
                    height=30
                )
            )]
        )
        gap_table.update_layout(
            title_text="Top 10 Compliance Gaps - Action Required", 
            height=400
        )
        
        # Generate HTML with print-friendly and sharing features
        dashboard_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Compliance Analysis Report</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/css/bootstrap.min.css" rel="stylesheet">
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                @media print {{
                    .no-print, .no-print * {{ display: none !important; }}
                    .page-break {{ page-break-before: always; }}
                    body {{ font-size: 12px; }}
                    h1 {{ font-size: 18px; }}
                    h2 {{ font-size: 16px; }}
                }}
                body {{ 
                    padding: 20px; 
                    font-family: Arial, sans-serif;
                }}
                .header {{ 
                    background-color: #f8f9fa; 
                    padding: 20px; 
                    border-radius: 5px; 
                    margin-bottom: 20px;
                    border-left: 5px solid #1a76ff;
                }}
                .summary-box {{
                    text-align: center;
                    padding: 15px;
                    border-radius: 5px;
                    margin-bottom: 20px;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                }}
                .compliant {{ background-color: #d4edda; border-left: 5px solid #4CAF50; }}
                .partial {{ background-color: #fff3cd; border-left: 5px solid #FF9800; }}
                .gap {{ background-color: #f8d7da; border-left: 5px solid #F44336; }}
                
                .actions-section {{
                    margin-top: 30px;
                    padding: 20px;
                    background-color: #e7f5ff;
                    border-radius: 5px;
                }}
                
                .print-button {{
                    background-color: #4CAF50;
                    color: white;
                    padding: 10px 20px;
                    border: none;
                    border-radius: 5px;
                    cursor: pointer;
                    margin-right: 10px;
                }}
                
                .email-button {{
                    background-color: #1a76ff;
                    color: white;
                    padding: 10px 20px;
                    border: none;
                    border-radius: 5px;
                    cursor: pointer;
                }}
                
                .footer {{
                    margin-top: 40px;
                    padding-top: 20px;
                    border-top: 1px solid #ddd;
                    font-size: 0.8em;
                    color: #666;
                }}
                
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                
                table, th, td {{
                    border: 1px solid #ddd;
                }}
                
                th, td {{
                    padding: 12px;
                    text-align: left;
                }}
                
                th {{
                    background-color: #f2f2f2;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <div class="row align-items-center">
                        <div class="col-md-8">
                            <h1>Compliance Analysis Report</h1>
                            <p class="text-muted">Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                        </div>
                        <div class="col-md-4 text-end no-print">
                            <button class="print-button" onclick="window.print()">Print Report</button>
                            <button class="email-button" onclick="shareReport()">Share Report</button>
                        </div>
                    </div>
                </div>
                
                <div class="row">
                    <div class="col-12">
                        <h2>Executive Summary</h2>
                        <p>
                            This report analyzes {total_obligations} regulatory obligations against {len(policies)} internal policies.
                            Overall compliance score is <strong>{overall_score*100:.1f}%</strong>.
                        </p>
                    </div>
                </div>
                
                <div class="row">
                    <div class="col-md-4">
                        <div class="summary-box compliant">
                            <h3>Compliant</h3>
                            <h2>{compliant_count}</h2>
                            <p>({(compliant_count/total_obligations*100):.1f}% of obligations)</p>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="summary-box partial">
                            <h3>Partially Compliant</h3>
                            <h2>{partial_count}</h2>
                            <p>({(partial_count/total_obligations*100):.1f}% of obligations)</p>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="summary-box gap">
                            <h3>Non-Compliant</h3>
                            <h2>{noncompliant_count}</h2>
                            <p>({(noncompliant_count/total_obligations*100):.1f}% of obligations)</p>
                        </div>
                    </div>
                </div>
                
                <div class="row mt-4">
                    <div class="col-12">
                        <div id="dashboard"></div>
                    </div>
                </div>
                
                <div class="page-break"></div>
                
                <div class="row mt-4">
                    <div class="col-12">
                        <h2>Compliance Gaps and Required Actions</h2>
                        <p>The following table highlights the top compliance gaps that require immediate attention:</p>
                        <div id="gap-table"></div>
                    </div>
                </div>
                
                <div class="actions-section">
                    <h3>Recommended Next Steps</h3>
                    <ol>
                        <li>Address the top {min(3, noncompliant_count)} compliance gaps identified in this report</li>
                        <li>Review and update policies that only partially address regulatory obligations</li>
                        <li>Schedule a follow-up analysis in 90 days to track progress</li>
                    </ol>
                </div>
                
                <div class="footer">
                    <p>This report is automatically generated and is intended for internal use only. For more detailed analysis and interactive exploration, please use the Compliance Analysis Dashboard app.</p>
                    <p>Report ID: {datetime.now().strftime("%Y%m%d")}_{hash(str(compliance_matrix))%1000000}</p>
                </div>
            </div>
            
            <script>
                var dashboardDiv = document.getElementById('dashboard');
                var gapTableDiv = document.getElementById('gap-table');
                
                Plotly.newPlot(dashboardDiv, {json.dumps(fig.to_dict())});
                Plotly.newPlot(gapTableDiv, {json.dumps(gap_table.to_dict())});
                
                // Share functionality 
                function shareReport() {{
                    // Create mailto link with minimal info for security
                    var subject = "Compliance Analysis Report - {datetime.now().strftime('%Y-%m-%d')}";
                    var body = "Attached is the compliance analysis report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}. " + 
                              "\\n\\nSummary:\\n" +
                              "- Total Obligations: {total_obligations}\\n" +
                              "- Compliance Score: {overall_score*100:.1f}%\\n" +
                              "- Compliant Items: {compliant_count}\\n" +
                              "- Non-compliant Items: {noncompliant_count}\\n\\n" +
                              "Please review the attached report for details.";
                    
                    var mailtoLink = "mailto:?subject=" + encodeURIComponent(subject) + "&body=" + encodeURIComponent(body);
                    window.location.href = mailtoLink;
                }}
            </script>
        </body>
        </html>
        """
        
        # Write the dashboard HTML to file
        with open(filepath, 'w') as f:
            f.write(dashboard_html)
        
        self.logger.info(f"HTML dashboard generated at: {filepath}")
        return filepath
        
    def _get_action_recommendation(self, compliance_level: float, matching_policies: List[Dict]) -> str:
        """Helper method to generate action recommendations based on compliance level."""
        if compliance_level < 0.3:
            if not matching_policies:
                return "Create new policy to address this obligation"
            else:
                return "Create dedicated policy section for this requirement"
        elif compliance_level < 0.6:
            return "Significantly enhance existing policy coverage"
        elif compliance_level < 0.8:
            return "Update policy with more specific compliance measures"
        else:
            return "No immediate action required"

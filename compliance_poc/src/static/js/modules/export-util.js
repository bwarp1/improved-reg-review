/**
 * Export Utility Module
 * Handles exporting data in various formats
 */

import dataManager from './data-manager.js';

class ExportUtil {
    initialize() {
        this._setupExportButtons();
    }
    
    _setupExportButtons() {
        const exportButtons = document.querySelectorAll('[data-export-format]');
        exportButtons.forEach(button => {
            button.addEventListener('click', () => {
                const format = button.getAttribute('data-export-format');
                this.exportFilteredData(format);
            });
        });
    }
    
    exportFilteredData(format) {
        if (dataManager.filteredData.length === 0) {
            this._showErrorMessage('No data to export.');
            return;
        }
        
        switch (format) {
            case 'csv':
                this._exportToCsv();
                break;
            case 'excel':
                this._exportToExcel();
                break;
            case 'pdf':
                this._exportToPdf();
                break;
            default:
                this._showErrorMessage(`Unsupported export format: ${format}`);
        }
    }
    
    _exportToCsv() {
        const data = dataManager.filteredData;
        if (data.length === 0) return;
        
        // Define columns to include
        const columns = [
            'id', 'title', 'category', 'risk_level', 
            'effective_date', 'compliance_status', 'department'
        ];
        
        // Create CSV header row
        const headerRow = columns.map(col => {
            return col.replace(/_/g, ' ')
                     .replace(/\w\S*/g, txt => txt.charAt(0).toUpperCase() + txt.substr(1).toLowerCase());
        });
        
        // Build CSV content
        let csvContent = headerRow.join(',') + '\r\n';
        
        // Add data rows
        data.forEach(reg => {
            const row = columns.map(col => {
                if (col === 'effective_date' && reg[col]) {
                    return new Date(reg[col]).toLocaleDateString();
                }
                
                let value = reg[col] || '';
                if (typeof value === 'string' && value.includes(',')) {
                    value = `"${value}"`;
                }
                
                return value;
            });
            
            csvContent += row.join(',') + '\r\n';
        });
        
        // Create and download the file
        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        
        link.setAttribute('href', url);
        link.setAttribute('download', `regulations_export_${new Date().toISOString().slice(0, 10)}.csv`);
        link.style.display = 'none';
        
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }
    
    _showErrorMessage(message) {
        alert(message);
    }
}

// Export singleton instance
const exportUtil = new ExportUtil();
export default exportUtil;

/**
 * Data Manager Module
 * Handles data processing, filtering, and state management
 */

class DataManager {
    constructor() {
        this.regulationData = [];
        this.filteredData = [];
        this.activeFilters = {
            searchTerm: '',
            category: 'all',
            riskLevel: 'all',
            status: 'all',
            dateRange: '',
            department: 'all',
            regulationType: 'all',
            tags: []
        };
        this.listeners = [];
    }
    
    loadData() {
        return fetch('/api/regulations/summary')
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                this.regulationData = data;
                this.filteredData = [...data];
                this._notifyListeners();
                return data;
            });
    }
    
    applyFilters(filters = {}) {
        // Update active filters with new filters
        Object.assign(this.activeFilters, filters);
        
        // Apply all filters to data
        this.filteredData = this.regulationData.filter(regulation => {
            // Search text filter
            if (this.activeFilters.searchTerm) {
                const searchTerm = this.activeFilters.searchTerm.toLowerCase();
                const searchFields = [
                    regulation.title || '', 
                    regulation.description || '', 
                    regulation.id || '',
                    regulation.category || ''
                ].map(field => field.toLowerCase());
                
                if (!searchFields.some(field => field.includes(searchTerm))) {
                    return false;
                }
            }
            
            // Category filter
            if (this.activeFilters.category && 
                this.activeFilters.category !== 'all' && 
                regulation.category !== this.activeFilters.category) {
                return false;
            }
            
            // Risk level filter
            if (this.activeFilters.riskLevel && 
                this.activeFilters.riskLevel !== 'all' && 
                regulation.risk_level !== this.activeFilters.riskLevel) {
                return false;
            }
            
            // Status filter
            if (this.activeFilters.status && 
                this.activeFilters.status !== 'all' && 
                regulation.status !== this.activeFilters.status) {
                return false;
            }
            
            // Department filter
            if (this.activeFilters.department && 
                this.activeFilters.department !== 'all' && 
                regulation.department !== this.activeFilters.department) {
                return false;
            }
            
            // Regulation type filter
            if (this.activeFilters.regulationType && 
                this.activeFilters.regulationType !== 'all' && 
                regulation.regulation_type !== this.activeFilters.regulationType) {
                return false;
            }
            
            // Date range filter
            if (this.activeFilters.dateRange) {
                try {
                    const [startDateStr, endDateStr] = this.activeFilters.dateRange.split(' to ');
                    const startDate = new Date(startDateStr);
                    const endDate = endDateStr ? new Date(endDateStr) : startDate;
                    const regDate = new Date(regulation.effective_date);
                    
                    if (isNaN(regDate) || regDate < startDate || regDate > endDate) {
                        return false;
                    }
                } catch (e) {
                    console.error('Error parsing date range:', e);
                }
            }
            
            // Tags filter
            if (this.activeFilters.tags.length > 0) {
                const regTags = regulation.tags || [];
                if (!this.activeFilters.tags.some(tag => regTags.includes(tag))) {
                    return false;
                }
            }
            
            return true;
        });
        
        // Notify listeners about the data change
        this._notifyListeners();
        
        return this.filteredData;
    }
    
    clearFilters() {
        this.activeFilters = {
            searchTerm: '',
            category: 'all',
            riskLevel: 'all',
            status: 'all',
            dateRange: '',
            department: 'all',
            regulationType: 'all',
            tags: []
        };
        
        this.filteredData = [...this.regulationData];
        this._notifyListeners();
        
        return this.filteredData;
    }
    
    getSummaryStatistics() {
        const stats = {
            totalRegulations: this.filteredData.length,
            highRiskCount: this.filteredData.filter(r => r.risk_level === 'High').length,
            upcomingCount: this.filteredData.filter(r => new Date(r.effective_date) > new Date()).length,
            complianceStatus: {
                'Compliant': 0,
                'Non-compliant': 0,
                'In progress': 0,
                'Not reviewed': 0
            }
        };
        
        // Calculate compliance status counts
        this.filteredData.forEach(reg => {
            const status = reg.compliance_status || 'Not reviewed';
            stats.complianceStatus[status] = (stats.complianceStatus[status] || 0) + 1;
        });
        
        return stats;
    }
    
    addChangeListener(callback) {
        if (typeof callback === 'function') {
            this.listeners.push(callback);
        }
    }
    
    removeChangeListener(callback) {
        const index = this.listeners.indexOf(callback);
        if (index !== -1) {
            this.listeners.splice(index, 1);
        }
    }
    
    _notifyListeners() {
        this.listeners.forEach(callback => {
            try {
                callback(this.filteredData);
            } catch (e) {
                console.error('Error in data change listener:', e);
            }
        });
    }
}

// Export singleton instance
const dataManager = new DataManager();
export default dataManager;

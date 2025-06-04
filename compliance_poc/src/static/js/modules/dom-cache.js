/**
 * DOM Cache Module
 * Caches DOM elements to avoid repeated querySelector calls
 */

class DOMCache {
    constructor() {
        this.elements = {};
        this.initialized = false;
    }
    
    initialize() {
        if (this.initialized) return;
        
        // Cache frequently used elements
        const selectors = [
            // Filter elements
            'searchInput', 'categoryFilter', 'riskLevelFilter', 'statusFilter', 'dateRange',
            'departmentFilter', 'regulationTypeFilter', 'tagFilter', 
            
            // Table elements
            'regulationsTableBody',
            
            // Summary elements
            'totalRegulations', 'highRiskCount', 'upcomingCount', 'complianceBreakdown',
            
            // Chart containers
            'categoryChart', 'riskChart', 'timelineChart', 'complianceGauge',
            'complianceForecast', 'departmentGapsChart', 'severityGapsChart',
            
            // Filter feedback elements
            'activeFilters', 'activeFiltersContainer', 'filterCount', 'filterFeedback',
            
            // Buttons
            'applyFilters', 'clearFilters'
        ];
        
        // Cache all elements by ID
        selectors.forEach(id => {
            this.elements[id] = document.getElementById(id);
        });
        
        this.initialized = true;
    }
    
    get(id) {
        // Return from cache if available
        if (this.elements[id]) {
            return this.elements[id];
        }
        
        // Get element and add to cache
        const element = document.getElementById(id);
        if (element) {
            this.elements[id] = element;
        }
        
        return element;
    }
}

// Export singleton instance
const domCache = new DOMCache();
export default domCache;

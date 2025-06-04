/**
 * Filter Manager Module
 * Handles filter UI and interactions
 */

import dataManager from './data-manager.js';
import domCache from './dom-cache.js';

class FilterManager {
    constructor() {
        this.debouncedApply = this._debounce(this._applyFilters, 300);
    }
    
    initialize() {
        this._setupFilterListeners();
        this._setupTagFilters();
        this._setupDatePicker();
    }
    
    _setupFilterListeners() {
        const dom = domCache;
        
        // Standard filter change events
        const filterElements = [
            'searchInput', 'categoryFilter', 'riskLevelFilter', 
            'statusFilter', 'departmentFilter', 'regulationTypeFilter'
        ];
        
        filterElements.forEach(id => {
            const element = dom.get(id);
            if (element) {
                element.addEventListener('change', () => this.debouncedApply());
                
                // For text input, also listen for keyup
                if (id === 'searchInput') {
                    element.addEventListener('input', () => this.debouncedApply());
                }
            }
        });
        
        // Clear filters button
        const clearFiltersBtn = dom.get('clearFilters');
        if (clearFiltersBtn) {
            clearFiltersBtn.addEventListener('click', () => this.clearAllFilters());
        }
    }
    
    _setupTagFilters() {
        const tagContainer = domCache.get('tagFilter');
        if (!tagContainer) return;
        
        // Create tag elements from data
        const allTags = new Set();
        dataManager.regulationData.forEach(reg => {
            if (reg.tags && Array.isArray(reg.tags)) {
                reg.tags.forEach(tag => allTags.add(tag));
            }
        });
        
        // Sort and add tags to container
        Array.from(allTags).sort().forEach(tag => {
            const tagEl = document.createElement('span');
            tagEl.className = 'badge badge-pill badge-light mr-2 mb-2 tag-item';
            tagEl.dataset.tag = tag;
            tagEl.textContent = tag;
            tagEl.addEventListener('click', () => {
                tagEl.classList.toggle('selected-tag');
                this.debouncedApply();
            });
            tagContainer.appendChild(tagEl);
        });
    }
    
    _setupDatePicker() {
        if (typeof flatpickr !== 'undefined') {
            const dateRange = domCache.get('dateRange');
            if (dateRange) {
                flatpickr(dateRange, { 
                    mode: 'range',
                    dateFormat: 'Y-m-d',
                    onChange: () => this.debouncedApply(),
                    maxDate: 'today',
                });
            }
        }
    }
    
    getSelectedTags() {
        const tagContainer = domCache.get('tagFilter');
        if (!tagContainer) return [];
        
        return Array.from(tagContainer.querySelectorAll('.selected-tag'))
            .map(tag => tag.dataset.tag);
    }
    
    highlightSearchTerms(term) {
        if (!term || term.length < 2) return;
        
        const tableBody = domCache.get('regulationsTableBody');
        if (!tableBody) return;
        
        const rows = tableBody.querySelectorAll('tr');
        rows.forEach(row => {
            const cells = row.querySelectorAll('td');
            cells.forEach(cell => {
                const text = cell.textContent;
                if (text.toLowerCase().includes(term.toLowerCase())) {
                    cell.innerHTML = text.replace(
                        new RegExp(`(${term.replace(/[-\/\\^$*+?.()|[\]{}]/g, '\\$&')})`, 'gi'), 
                        '<span class="search-highlight">$1</span>'
                    );
                }
            });
        });
    }
    
    clearHighlights() {
        const tableBody = domCache.get('regulationsTableBody');
        if (!tableBody) return;
        
        tableBody.querySelectorAll('.search-highlight').forEach(span => {
            span.parentNode.replaceChild(document.createTextNode(span.textContent), span);
        });
    }
    
    updateActiveFilters() {
        const dom = domCache;
        const activeFiltersContainer = dom.get('activeFilters');
        if (!activeFiltersContainer) return;
        
        activeFiltersContainer.innerHTML = '';
        
        const filters = {
            searchTerm: dom.get('searchInput')?.value,
            category: dom.get('categoryFilter')?.value,
            riskLevel: dom.get('riskLevelFilter')?.value,
            dateRange: dom.get('dateRange')?.value,
            status: dom.get('statusFilter')?.value,
            department: dom.get('departmentFilter')?.value,
            regulationType: dom.get('regulationTypeFilter')?.value,
            tags: this.getSelectedTags()
        };
        
        // Add filter pills for active filters
        if (filters.searchTerm) {
            this._addFilterPill(activeFiltersContainer, 'Search', filters.searchTerm);
        }
        
        if (filters.category && filters.category !== 'all') {
            this._addFilterPill(activeFiltersContainer, 'Category', filters.category);
        }
        
        // Add remaining filter pills with similar pattern
        // ...
        
        // Show or hide container based on active filters
        const hasActiveFilters = activeFiltersContainer.children.length > 0;
        const container = dom.get('activeFiltersContainer');
        if (container) {
            container.style.display = hasActiveFilters ? 'block' : 'none';
        }
        
        // Add "Clear all" button if there are active filters
        if (hasActiveFilters) {
            const clearAllBtn = document.createElement('button');
            clearAllBtn.className = 'btn btn-sm btn-outline-secondary ml-2';
            clearAllBtn.innerText = 'Clear all filters';
            clearAllBtn.addEventListener('click', () => this.clearAllFilters());
            activeFiltersContainer.appendChild(clearAllBtn);
        }
    }
    
    clearAllFilters() {
        const dom = domCache;
        
        // Reset basic filters
        if (dom.get('searchInput')) dom.get('searchInput').value = '';
        if (dom.get('categoryFilter')) dom.get('categoryFilter').value = 'all';
        if (dom.get('riskLevelFilter')) dom.get('riskLevelFilter').value = 'all';
        if (dom.get('statusFilter')) dom.get('statusFilter').value = 'all';
        if (dom.get('dateRange')) dom.get('dateRange').value = '';
        if (dom.get('departmentFilter')) dom.get('departmentFilter').value = 'all';
        if (dom.get('regulationTypeFilter')) dom.get('regulationTypeFilter').value = 'all';
        
        // Reset tag filters
        const tagContainer = dom.get('tagFilter');
        if (tagContainer) {
            tagContainer.querySelectorAll('.tag-item.selected-tag').forEach(tag => {
                tag.classList.remove('selected-tag');
            });
        }
        
        // Clear any highlights
        this.clearHighlights();
        
        // Apply filters (reset to show all data)
        this._applyFilters();
    }
    
    _applyFilters() {
        const dom = domCache;
        
        // Gather filter values
        const filters = {
            searchTerm: dom.get('searchInput')?.value || '',
            category: dom.get('categoryFilter')?.value || 'all',
            riskLevel: dom.get('riskLevelFilter')?.value || 'all',
            dateRange: dom.get('dateRange')?.value || '',
            status: dom.get('statusFilter')?.value || 'all',
            department: dom.get('departmentFilter')?.value || 'all',
            regulationType: dom.get('regulationTypeFilter')?.value || 'all',
            tags: this.getSelectedTags()
        };
        
        // Apply filters to data manager
        dataManager.applyFilters(filters);
        
        // Update UI elements
        this.updateActiveFilters();
        
        // Clear highlights and re-apply for search term
        this.clearHighlights();
        if (filters.searchTerm && filters.searchTerm.length > 2) {
            this.highlightSearchTerms(filters.searchTerm);
        }
    }
    
    _addFilterPill(container, type, value) {
        const pill = document.createElement('span');
        pill.className = 'filter-badge';
        pill.innerHTML = `
            ${type}: ${value}
            <span class="close" aria-label="Remove filter">&times;</span>
        `;
        
        pill.querySelector('.close').addEventListener('click', () => {
            // Handle removing this specific filter
            switch (type) {
                case 'Search':
                    domCache.get('searchInput').value = '';
                    break;
                case 'Category':
                    domCache.get('categoryFilter').value = 'all';
                    break;
                // Other cases follow the same pattern
            }
            
            // Apply the updated filters
            this._applyFilters();
        });
        
        container.appendChild(pill);
    }
    
    _debounce(func, wait) {
        let timeout;
        return function(...args) {
            clearTimeout(timeout);
            timeout = setTimeout(() => func.apply(this, args), wait);
        };
    }
}

// Export singleton instance
const filterManager = new FilterManager();
export default filterManager;

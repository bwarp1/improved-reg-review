// Dashboard functionality for regulatory analysis

// Import modules
import domCache from './modules/dom-cache.js';
import dataManager from './modules/data-manager.js';
import chartFactory from './modules/chart-factory.js';
import filterManager from './modules/filter-manager.js';
import exportUtil from './modules/export-util.js';
import * as chartRenderers from './modules/chart-renderers.js';

// Initialize dashboard on page load
document.addEventListener('DOMContentLoaded', function() {
    // Initialize DOM cache first
    domCache.initialize();
    
    // Initialize modules
    filterManager.initialize();
    exportUtil.initialize();
    
    // Set up keyboard shortcuts
    registerKeyboardShortcuts();
    
    // Load data and render dashboard
    loadDashboardData();
});

// Load regulation data and initialize visualizations
function loadDashboardData() {
    showLoadingSpinner();
    
    dataManager.loadData()
        .then(() => {
            renderDashboard();
            hideLoadingSpinner();
        })
        .catch(error => {
            console.error('Error loading dashboard data:', error);
            showErrorMessage('Failed to load dashboard data. Please try again later.');
            hideLoadingSpinner();
        });
}

// Render all dashboard components
function renderDashboard() {
    renderRegulationTable();
    chartRenderers.renderAllCharts();
    updateSummaryStats();
}

// Render the regulations table with filtered data
function renderRegulationTable() {
    const tableBody = domCache.get('regulationsTableBody');
    if (!tableBody) return;
    
    tableBody.innerHTML = '';
    
    if (dataManager.filteredData.length === 0) {
        const noDataRow = document.createElement('tr');
        noDataRow.innerHTML = `<td colspan="5" class="text-center">No regulations match your filters</td>`;
        tableBody.appendChild(noDataRow);
        return;
    }
    
    dataManager.filteredData.forEach(regulation => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${regulation.id}</td>
            <td>
                <a href="/regulation/${regulation.id}" class="regulation-link">
                    ${regulation.title}
                </a>
            </td>
            <td>${regulation.category}</td>
            <td>
                <span class="risk-badge risk-${regulation.risk_level.toLowerCase()}">${regulation.risk_level}</span>
            </td>
            <td>${new Date(regulation.effective_date).toLocaleDateString()}</td>
        `;
        tableBody.appendChild(row);
    });
}

// Update summary statistics with additional metrics
function updateSummaryStats() {
    const stats = dataManager.getSummaryStatistics();
    
    // Update summary cards
    if (domCache.get('totalRegulations')) {
        domCache.get('totalRegulations').textContent = stats.totalRegulations;
    }
    
    if (domCache.get('highRiskCount')) {
        domCache.get('highRiskCount').textContent = stats.highRiskCount;
    }
    
    if (domCache.get('upcomingCount')) {
        domCache.get('upcomingCount').textContent = stats.upcomingCount;
    }
    
    // Update compliance status breakdown
    const complianceBreakdown = domCache.get('complianceBreakdown');
    if (complianceBreakdown) {
        complianceBreakdown.innerHTML = '';
        
        for (const [status, count] of Object.entries(stats.complianceStatus)) {
            if (count > 0) {
                const statusClass = status.toLowerCase().replace(' ', '-');
                const percentage = Math.round(count / stats.totalRegulations * 100);
                
                complianceBreakdown.innerHTML += `
                    <div class="status-pill ${statusClass}">
                        ${status}: ${count}
                        <span class="percentage">(${percentage}%)</span>
                    </div>
                `;
            }
        }
    }
}

// Show loading spinner while fetching data
function showLoadingSpinner() {
    const spinnerHtml = `
        <div class="text-center my-4" id="loadingSpinner">
            <div class="spinner-border text-primary" role="status">
                <span class="sr-only">Loading...</span>
            </div>
            <p class="mt-2">Loading dashboard data...</p>
        </div>
    `;
    const container = document.querySelector('.dashboard-container');
    if (container) {
        const loadingElem = document.createElement('div');
        loadingElem.innerHTML = spinnerHtml;
        container.prepend(loadingElem.firstChild);
    }
}

// Hide loading spinner
function hideLoadingSpinner() {
    document.getElementById('loadingSpinner')?.remove();
}

// Register keyboard shortcuts
function registerKeyboardShortcuts() {
    document.addEventListener('keydown', function(event) {
        // Only activate shortcuts when not in form inputs
        if (event.target.tagName === 'INPUT' || 
            event.target.tagName === 'TEXTAREA' || 
            event.target.tagName === 'SELECT') {
            return;
        }
        
        // Ctrl+/ - Focus search
        if (event.ctrlKey && event.key === '/') {
            domCache.get('searchInput')?.focus();
            event.preventDefault();
        }
        
        // Alt+R - Refresh data
        if (event.altKey && event.key === 'r') {
            loadDashboardData();
            event.preventDefault();
        }
    });
}

// Show error message
function showErrorMessage(msg) {
    alert(msg);
}

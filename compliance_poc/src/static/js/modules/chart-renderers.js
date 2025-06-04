/**
 * Chart Renderers Module
 * Contains functions to render various chart types
 */

import chartFactory from './chart-factory.js';
import dataManager from './data-manager.js';
import domCache from './dom-cache.js';

// Render all charts
export function renderAllCharts() {
    renderCategoryDistribution();
    renderRiskDistribution();
    renderTimelineChart();
    renderComplianceGauge();
    renderComplianceGapsByDepartment();
    renderComplianceGapsBySeverity();
}

// Render category distribution chart
export function renderCategoryDistribution() {
    const ctx = domCache.get('categoryChart');
    if (!ctx || dataManager.filteredData.length === 0) return;
    
    // Count categories
    const categories = {};
    dataManager.filteredData.forEach(reg => {
        const category = reg.category || 'Uncategorized';
        categories[category] = (categories[category] || 0) + 1;
    });
    
    // Prepare chart data
    const labels = Object.keys(categories);
    const data = Object.values(categories);
    
    const colors = [
        'rgba(78, 115, 223, 0.8)',
        'rgba(54, 185, 204, 0.8)',
        'rgba(28, 200, 138, 0.8)',
        'rgba(246, 194, 62, 0.8)',
        'rgba(231, 74, 59, 0.8)',
        'rgba(90, 92, 105, 0.8)'
    ];
    
    // Create chart
    chartFactory.createChart('categoryChart', 'pie', {
        labels: labels,
        datasets: [{
            data: data,
            backgroundColor: colors.slice(0, labels.length),
            borderWidth: 1
        }]
    });
}

// Render risk level distribution chart
export function renderRiskDistribution() {
    const ctx = domCache.get('riskChart');
    if (!ctx || dataManager.filteredData.length === 0) return;
    
    // Count risk levels
    const riskLevels = {
        'High': 0,
        'Medium': 0,
        'Low': 0
    };
    
    dataManager.filteredData.forEach(reg => {
        const risk = reg.risk_level || 'Medium';
        riskLevels[risk] = (riskLevels[risk] || 0) + 1;
    });
    
    const labels = Object.keys(riskLevels);
    const data = Object.values(riskLevels);
    
    const colors = {
        'High': 'rgba(231, 74, 59, 0.8)',
        'Medium': 'rgba(246, 194, 62, 0.8)',
        'Low': 'rgba(28, 200, 138, 0.8)'
    };
    
    chartFactory.createChart('riskChart', 'doughnut', {
        labels: labels,
        datasets: [{
            data: data,
            backgroundColor: labels.map(label => colors[label]),
            borderWidth: 1
        }]
    });
}

// Render timeline chart
export function renderTimelineChart() {
    const ctx = domCache.get('timelineChart');
    if (!ctx || dataManager.filteredData.length === 0) return;
    
    // Sort data by date
    const sortedData = [...dataManager.filteredData].sort((a, b) => 
        new Date(a.effective_date) - new Date(b.effective_date)
    );
    
    const dates = sortedData.map(r => new Date(r.effective_date).toLocaleDateString());
    
    // Calculate cumulative count
    const cumulativeData = [];
    let count = 0;
    sortedData.forEach(() => {
        count++;
        cumulativeData.push(count);
    });
    
    chartFactory.createChart('timelineChart', 'line', {
        labels: dates,
        datasets: [{
            label: 'Cumulative Regulations',
            data: cumulativeData,
            backgroundColor: 'rgba(78, 115, 223, 0.2)',
            borderColor: 'rgba(78, 115, 223, 1)',
            borderWidth: 2,
            tension: 0.1,
            fill: true
        }]
    });
}

// Additional chart rendering functions would follow the same pattern
export function renderComplianceGauge() {
    // Implementation similar to other chart functions
}

export function renderComplianceGapsByDepartment(chartType = 'pie') {
    // Implementation similar to other chart functions
}

export function renderComplianceGapsBySeverity() {
    // Implementation similar to other chart functions
}

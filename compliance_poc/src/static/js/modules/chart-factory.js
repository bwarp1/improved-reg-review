/**
 * Chart Factory Module
 * Creates and manages chart instances with standard configurations
 */

class ChartFactory {
    constructor() {
        this.charts = {};
        this.chartDefaults = {
            pie: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom'
                    }
                }
            },
            bar: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: { precision: 0 }
                    }
                }
            },
            line: {
                responsive: true,
                maintainAspectRatio: false,
                tension: 0.1
            },
            doughnut: {
                responsive: true,
                maintainAspectRatio: false,
                cutout: '60%'
            }
        };
    }

    createChart(id, type, data, customOptions = {}) {
        const ctx = document.getElementById(id);
        if (!ctx) return null;
        
        // Destroy existing chart if it exists
        if (this.charts[id]) {
            this.charts[id].destroy();
        }
        
        // Merge default options with custom options
        const defaultOptions = this.chartDefaults[type] || {};
        const options = this._mergeDeep(defaultOptions, customOptions);
        
        // Create new chart
        this.charts[id] = new Chart(ctx, {
            type: type === 'area' ? 'line' : type,
            data: data,
            options: options
        });
        
        return this.charts[id];
    }
    
    updateChartData(id, newData) {
        if (!this.charts[id]) return false;
        
        const chart = this.charts[id];
        chart.data = newData;
        chart.update();
        return true;
    }
    
    changeChartType(id, newType) {
        if (!this.charts[id]) return false;
        
        const currentChart = this.charts[id];
        const data = currentChart.data;
        const options = currentChart.options;
        
        currentChart.destroy();
        
        this.charts[id] = new Chart(
            document.getElementById(id),
            {
                type: newType === 'area' ? 'line' : newType,
                data: data,
                options: options
            }
        );
        
        return true;
    }
    
    _mergeDeep(target, source) {
        const output = Object.assign({}, target);
        if (this._isObject(target) && this._isObject(source)) {
            Object.keys(source).forEach(key => {
                if (this._isObject(source[key])) {
                    if (!(key in target))
                        Object.assign(output, { [key]: source[key] });
                    else
                        output[key] = this._mergeDeep(target[key], source[key]);
                } else {
                    Object.assign(output, { [key]: source[key] });
                }
            });
        }
        return output;
    }
    
    _isObject(item) {
        return (item && typeof item === 'object' && !Array.isArray(item));
    }
}

// Export singleton instance
const chartFactory = new ChartFactory();
export default chartFactory;

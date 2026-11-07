/**
 * script.js - AcrePredict AI Client-Side Scripts
 * Handles preset autofill, form validation, and Chart.js initialization.
 */

document.addEventListener("DOMContentLoaded", function () {
    // 1. Initialize Bootstrap Tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // 2. Preset Form Autofill
    const presetButtons = document.querySelectorAll(".preset-chip, .preset-btn");
    if (presetButtons.length > 0) {
        const presets = {
            family: {
                overall_qual: "7",
                overall_cond: "5",
                gr_liv_area: 1850,
                total_bsmt_sf: 1100,
                lot_area: 10200,
                bedroom_abv_gr: "3",
                full_bath: "2",
                floors: "2.0",
                garage_cars: "2",
                year_built: 2004,
                neighborhood: "CollgCr"
            },
            luxury: {
                overall_qual: "9",
                overall_cond: "6",
                gr_liv_area: 3200,
                total_bsmt_sf: 1850,
                lot_area: 14500,
                bedroom_abv_gr: "4",
                full_bath: "3",
                floors: "2.0",
                garage_cars: "3",
                year_built: 2012,
                neighborhood: "NoRidge"
            },
            starter: {
                overall_qual: "5",
                overall_cond: "6",
                gr_liv_area: 1050,
                total_bsmt_sf: 750,
                lot_area: 6800,
                bedroom_abv_gr: "2",
                full_bath: "1",
                floors: "1.0",
                garage_cars: "1",
                year_built: 1978,
                neighborhood: "OldTown"
            }
        };

        presetButtons.forEach(button => {
            button.addEventListener("click", function () {
                const presetKey = this.getAttribute("data-preset");
                const data = presets[presetKey];
                if (!data) return;

                // Populate form fields
                for (const [key, value] of Object.entries(data)) {
                    const field = document.getElementById(key);
                    if (field) {
                        field.value = value;
                    }
                }

                // Visual button state feedback
                presetButtons.forEach(btn => btn.classList.remove("active"));
                this.classList.add("active");
            });
        });
    }

    // 3. Form Validation & Submit Loading Spinner
    const predictionForm = document.getElementById("predictionForm");
    if (predictionForm) {
        predictionForm.addEventListener("submit", function (event) {
            if (!predictionForm.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
            } else {
                const submitBtn = document.getElementById("submitBtn");
                if (submitBtn) {
                    submitBtn.disabled = true;
                    submitBtn.innerHTML = `
                        <span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>
                        Computing House Valuation...
                    `;
                }
            }
            predictionForm.classList.add("was-validated");
        });
    }

    // 4. Chart.js Dashboard Charts
    if (typeof dashboardMetadata !== "undefined" && dashboardMetadata.metrics) {
        initDashboardCharts(dashboardMetadata);
    }
});

/**
 * Render Chart.js charts on the ML Dashboard with modern aesthetic
 */
function initDashboardCharts(data) {
    const metrics = data.metrics || {};
    const modelNames = Object.keys(metrics);

    // Extract test R2 and RMSE
    const testR2Scores = modelNames.map(name => (metrics[name].test_r2 * 100).toFixed(2));
    const rmseScores = modelNames.map(name => Math.round(metrics[name].rmse));

    // Chart 1: Model Benchmark (Dual Axis R2 vs RMSE)
    const ctxModel = document.getElementById("modelComparisonChart");
    if (ctxModel) {
        new Chart(ctxModel, {
            type: "bar",
            data: {
                labels: modelNames.map(n => n.replace(" Regressor", "")),
                datasets: [
                    {
                        label: "Test R² Score (%)",
                        data: testR2Scores,
                        backgroundColor: modelNames.map(n => n === data.best_model_name ? "#16a34a" : "#cbd5e1"),
                        borderRadius: 8,
                        yAxisID: "y"
                    },
                    {
                        label: "Test RMSE ($)",
                        data: rmseScores,
                        type: "line",
                        borderColor: "#ef4444",
                        backgroundColor: "#ef4444",
                        borderWidth: 2.5,
                        tension: 0.35,
                        pointRadius: 5,
                        pointBackgroundColor: "#ef4444",
                        yAxisID: "y1"
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: 'index',
                    intersect: false
                },
                scales: {
                    y: {
                        type: 'linear',
                        display: true,
                        position: 'left',
                        title: { display: true, text: 'R² Score (%)', color: '#16a34a', font: { weight: 'bold', size: 12 } },
                        min: 60,
                        max: 100,
                        grid: { color: 'rgba(0,0,0,0.04)' }
                    },
                    y1: {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        title: { display: true, text: 'RMSE ($)', color: '#ef4444', font: { weight: 'bold', size: 12 } },
                        grid: { drawOnChartArea: false },
                        ticks: {
                            callback: value => '$' + Number(value).toLocaleString()
                        }
                    },
                    x: {
                        grid: { display: false }
                    }
                },
                plugins: {
                    legend: { position: 'top' },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                let label = context.dataset.label || '';
                                if (label.includes('R²')) {
                                    return `${label}: ${context.parsed.y}%`;
                                } else {
                                    return `${label}: $${Number(context.parsed.y).toLocaleString()}`;
                                }
                            }
                        }
                    }
                }
            }
        });
    }

    // Chart 2: Feature Importance
    const ctxFeature = document.getElementById("featureImportanceChart");
    if (ctxFeature && data.top_features) {
        const featureLabels = data.top_features.map(f => {
            const raw = f.feature;
            if (raw === "OverallQual") return "Overall Quality";
            if (raw === "GrLivArea") return "Living Area (sq ft)";
            if (raw === "TotalBsmtSF") return "Basement Area";
            if (raw === "GarageCars") return "Garage Capacity";
            if (raw === "YearBuilt") return "Year Built";
            if (raw === "LotArea") return "Lot Area";
            if (raw === "OverallCond") return "Overall Condition";
            if (raw === "FullBath") return "Full Bathrooms";
            if (raw.startsWith("Neighborhood_")) return "Loc: " + raw.replace("Neighborhood_", "");
            return raw;
        });

        const featureValues = data.top_features.map(f => (f.importance * 100).toFixed(1));

        new Chart(ctxFeature, {
            type: "bar",
            data: {
                labels: featureLabels,
                datasets: [{
                    label: "Relative Importance (%)",
                    data: featureValues,
                    backgroundColor: [
                        "#0f2744", "#1e3a8a", "#1d4ed8", "#2563eb",
                        "#16a34a", "#22c55e", "#4ade80", "#86efac",
                        "#94a3b8", "#cbd5e1"
                    ],
                    borderRadius: 8
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        title: { display: true, text: 'Importance Weight (%)', font: { weight: 'bold', size: 12 } },
                        grid: { color: 'rgba(0,0,0,0.04)' }
                    },
                    y: {
                        grid: { display: false }
                    }
                },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: context => ` Importance: ${context.parsed.x}%`
                        }
                    }
                }
            }
        });
    }
}

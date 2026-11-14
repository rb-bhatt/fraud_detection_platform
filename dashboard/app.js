
const API_BASE_URL =
    window.FRAUD_API_URL ||
    ((window.location.port === "5500" &&
        window.location.hostname === "127.0.0.1")
        ? "http://127.0.0.1:8000"
        : (window.location.port === "5500" &&
            window.location.hostname === "localhost")
            ? "http://localhost:8000"
            : "/api");

let decisionChart;
let riskChart;

const e = {
    apiStatus: document.getElementById("api-status"),
    total: document.getElementById("total-transactions"),
    approved: document.getElementById("approved-count"),
    review: document.getElementById("review-count"),
    blocked: document.getElementById("blocked-count"),
    fraudRate: document.getElementById("fraud-rate"),
    anomalyScore: document.getElementById("anomaly-score"),
    openCases: document.getElementById("open-cases"),
    reviewTable: document.getElementById("review-table"),
    transactionsTable: document.getElementById("transactions-table"),
    primaryModel: document.getElementById("primary-model"),
    secondaryModel: document.getElementById("secondary-model"),
    approvalThreshold: document.getElementById("approval-threshold"),
    blockThreshold: document.getElementById("block-threshold"),
    predictionForm: document.getElementById("prediction-form"),
    predictionResult: document.getElementById("prediction-result"),
    summary: document.getElementById("investigation-summary"),
    summaryText: document.getElementById("summary-text")
};

async function fetchJson(endpoint, options = {}) {
    const response = await fetch(`${API_BASE_URL}${endpoint}`, options);

    if (!response.ok) {
        throw new Error(`Request failed: ${response.status}`);
    }

    return response.json();
}

function number(value, decimals = 0) {
    return Number(value || 0).toLocaleString(undefined, {
        maximumFractionDigits: decimals
    });
}

function badge(value) {
    return `<span class="decision-badge decision-${String(value).toLowerCase()}">${value}</span>`;
}

function renderMetrics(metrics) {
    e.total.textContent = number(metrics.total_transactions);
    e.approved.textContent = number(metrics.approved);
    e.review.textContent = number(metrics.review);
    e.blocked.textContent = number(metrics.blocked);
    e.fraudRate.textContent =
        `${(Number(metrics.fraud_rate || 0) * 100).toFixed(2)}%`;
    e.anomalyScore.textContent =
        number(metrics.average_anomaly_score, 2);
    e.openCases.textContent =
        `${number(metrics.open_review_cases)} open`;
}

function renderReviews(cases) {
    const open = cases.filter(item => item.status === "open");

    if (!open.length) {
        e.reviewTable.innerHTML =
            '<tr><td colspan="4" class="empty-state">No review cases found.</td></tr>';
        return;
    }

    e.reviewTable.innerHTML = open.slice(0, 8).map(item => `
        <tr>
            <td>${item.transaction_id.slice(0, 14)}...</td>
            <td>${badge(item.status)}</td>
            <td>${new Date(item.created_at).toLocaleString()}</td>
            <td>
                <button class="action-button"
                    data-case-action="approve"
                    data-transaction-id="${item.transaction_id}">
                    Approve
                </button>
                <button class="action-button"
                    data-case-action="reject"
                    data-transaction-id="${item.transaction_id}">
                    Reject
                </button>
            </td>
        </tr>
    `).join("");
}

function renderTransactions(rows) {
    if (!rows.length) {
        e.transactionsTable.innerHTML =
            '<tr><td colspan="6" class="empty-state">No transactions found.</td></tr>';
        return;
    }

    e.transactionsTable.innerHTML = rows.slice(0, 12).map(item => `
        <tr>
            <td>${item.transaction_id.slice(0, 10)}...</td>
            <td>$${number(item.amount, 2)}</td>
            <td>${(Number(item.fraud_probability) * 100).toFixed(2)}%</td>
            <td>${number(item.anomaly_score, 2)}</td>
            <td>${badge(item.decision)}</td>
            <td>${number(item.latency_ms, 2)} ms</td>
        </tr>
    `).join("");
}

function renderModelInfo(info) {
    e.primaryModel.textContent = info.model;
    e.secondaryModel.textContent = info.secondary_model;
    e.approvalThreshold.textContent = info.approval_threshold;
    e.blockThreshold.textContent = info.block_threshold;
}

function renderCharts(metrics) {
    const options = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                labels: {
                    color: "#e8f0fa"
                }
            }
        }
    };

    if (decisionChart) decisionChart.destroy();

    decisionChart = new Chart(
        document.getElementById("decision-chart"),
        {
            type: "doughnut",
            data: {
                labels: ["Approved", "Review", "Blocked"],
                datasets: [{
                    data: [
                        metrics.approved,
                        metrics.review,
                        metrics.blocked
                    ],
                    backgroundColor: [
                        "#39d98a",
                        "#ffc857",
                        "#ff6b7a"
                    ],
                    borderWidth: 0
                }]
            },
            options
        }
    );

    if (riskChart) riskChart.destroy();

    riskChart = new Chart(
        document.getElementById("risk-chart"),
        {
            type: "bar",
            data: {
                labels: ["Fraud rate (%)", "Anomaly score"],
                datasets: [{
                    label: "Current value",
                    data: [
                        Number(metrics.fraud_rate || 0) * 100,
                        Number(metrics.average_anomaly_score || 0)
                    ],
                    backgroundColor: ["#42a5f5", "#a78bfa"],
                    borderRadius: 6
                }]
            },
            options: {
                ...options,
                scales: {
                    x: {
                        ticks: { color: "#8da1b8" },
                        grid: { display: false }
                    },
                    y: {
                        ticks: { color: "#8da1b8" },
                        grid: { color: "#243951" }
                    }
                }
            }
        }
    );
}

async function loadDashboard() {
    try {
        const [metrics, transactions, cases, modelInfo] =
            await Promise.all([
                fetchJson("/metrics"),
                fetchJson("/transactions"),
                fetchJson("/review-cases"),
                fetchJson("/model-info")
            ]);

        renderMetrics(metrics);
        renderTransactions(transactions);
        renderReviews(cases);
        renderCharts(metrics);
        renderModelInfo(modelInfo);

        e.apiStatus.textContent = "API connected";
        document.querySelector(".status-dot").style.background = "#39d98a";

    } catch (error) {
        e.apiStatus.textContent = "API unavailable";
        document.querySelector(".status-dot").style.background = "#ff6b7a";
        console.error("Dashboard loading failed:", error);
    }
}

async function updateCase(transactionId, action) {
    const button = document.querySelector(
        `[data-transaction-id="${transactionId}"][data-case-action="${action}"]`
    );

    if (button) button.disabled = true;

    try {
        await fetchJson(`/${action}-case`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                transaction_id: transactionId,
                notes: `${action}d from dashboard`
            })
        });

        await loadDashboard();

    } catch (error) {
        console.error("Case update failed:", error);
        if (button) button.disabled = false;
    }
}

function buildPredictionPayload(amount, time) {
    const payload = {
        Time: Number(time),
        Amount: Number(amount)
    };

    for (let i = 1; i <= 28; i++) {
        payload[`V${i}`] = 0;
    }

    return payload;
}

async function submitPrediction(event) {
    event.preventDefault();

    const formData = new FormData(e.predictionForm);

    try {
        const result = await fetchJson("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(
                buildPredictionPayload(
                    formData.get("Amount"),
                    formData.get("Time")
                )
            )
        });

        const prediction = result.prediction;

        console.log("FULL PREDICTION RESPONSE:", result);
        console.log("INVESTIGATION SUMMARY:", result.investigation_summary);

        e.predictionResult.hidden = false;
        e.predictionResult.innerHTML = `
            ${badge(prediction.decision)}
            <strong>
                ${(prediction.fraud_probability * 100).toFixed(2)}%
                fraud probability
            </strong>
            <br>
            <span class="form-help">
                Anomaly score: ${number(prediction.anomaly_score, 2)}
                · Latency: ${number(result.latency_ms, 2)} ms
            </span>
        `;

        // Display AI investigation summary
        const summary = result.investigation_summary;

        e.summary.hidden = !summary;
        e.summaryText.textContent = summary || "";

        await loadDashboard();

    } catch (error) {
        e.predictionResult.hidden = false;
        e.predictionResult.textContent =
            "Prediction failed. Check that the API is running.";

        e.summary.hidden = true;
        e.summaryText.textContent = "";

        console.error("Prediction failed:", error);
    }
}

document
    .getElementById("refresh-button")
    .addEventListener("click", loadDashboard);

e.reviewTable.addEventListener("click", event => {
    const button = event.target.closest("[data-case-action]");

    if (button) {
        updateCase(
            button.dataset.transactionId,
            button.dataset.caseAction
        );
    }
});

e.predictionForm.addEventListener("submit", submitPrediction);

loadDashboard();

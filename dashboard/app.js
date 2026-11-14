
'use strict';

// ================================================
// 1. API CONFIGURATION
// ================================================

const API_BASE_URL = window.FRAUD_API_URL || (
    window.location.port === '5500' &&
    ['127.0.0.1', 'localhost'].includes(window.location.hostname)
        ? `${window.location.protocol}//${window.location.hostname}:8000`
        : '/api'
);

// ================================================
// 2. APPLICATION STATE
// ================================================

const state = {
    source: 'checking',
    metrics: null,
    transactions: [],
    reviewCases: [],
    modelInfo: {},
    lastReport: null,
    lastUpdated: null,
};

// ================================================
// 3. DOM ELEMENTS
// ================================================

const el = {
    apiStatus: document.getElementById('api-status'),
    sidebarConnection: document.getElementById('sidebar-connection'),
    modeBadge: document.getElementById('mode-badge'),
    sidebarOpenCases: document.getElementById('sidebar-open-cases'),
    lastUpdated: document.getElementById('last-updated'),

    total: document.getElementById('total-transactions'),
    approved: document.getElementById('approved-count'),
    review: document.getElementById('review-count'),
    blocked: document.getElementById('blocked-count'),
    anomaly: document.getElementById('anomaly-score'),

    networkTotal: document.getElementById('network-total'),
    networkApproved: document.getElementById('network-approved'),
    networkReview: document.getElementById('network-review'),
    networkBlocked: document.getElementById('network-blocked'),
    networkState: document.getElementById('network-state'),

    primaryModel: document.getElementById('primary-model'),
    secondaryModel: document.getElementById('secondary-model'),

    recentTransactions: document.getElementById('recent-transactions'),
    transactionTable: document.getElementById('transactions-table'),
    transactionCount: document.getElementById('transaction-count'),

    reviewTable: document.getElementById('review-table'),
    openCases: document.getElementById('open-cases'),

    predictionForm: document.getElementById('prediction-form'),
    predictionSubmit: document.getElementById('prediction-submit'),
    predictionPlaceholder: document.getElementById('prediction-placeholder'),
    predictionResult: document.getElementById('prediction-result'),

    investigationSummary: document.getElementById('investigation-summary'),
    summaryText: document.getElementById('summary-text'),
    agentRecommendation: document.getElementById('agent-recommendation'),

    reportsList: document.getElementById('reports-list'),

    apiEndpoint: document.getElementById('api-endpoint'),
    settingsConnection: document.getElementById('settings-connection'),
    settingsSource: document.getElementById('settings-source'),
    settingsMode: document.getElementById('settings-mode'),
};

const pageTitles = {
    overview: 'Overview',
    transactions: 'Transactions',
    prediction: 'Run prediction',
    reviews: 'Investigation & review',
    models: 'Model performance',
    reports: 'AI investigation reports',
    settings: 'Settings',
};


// ================================================
// 4. UTILITY FUNCTIONS
// ================================================

function formatNumber(value, digits = 0) {
    const numeric = Number(value);

    if (!Number.isFinite(numeric)) {
        return '—';
    }

    return numeric.toLocaleString(undefined, {
        maximumFractionDigits: digits,
    });
}

function formatPercent(value) {
    const numeric = Number(value);

    return Number.isFinite(numeric)
        ? `${(numeric * 100).toFixed(2)}%`
        : '—';
}

function formatDate(value) {
    if (!value) return '—';

    const date = new Date(value);

    return Number.isNaN(date.getTime())
        ? '—'
        : date.toLocaleString();
}

function escapeHtml(value) {
    return String(value ?? '').replace(/[&<>"']/g, character => ({
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#39;',
    })[character]);
}


// ================================================
// 5. API FUNCTIONS
// ================================================

async function fetchJson(endpoint, options = {}) {
    const response = await fetch(
        `${API_BASE_URL}${endpoint}`,
        options
    );

    const raw = await response.text();

    let payload;

    try {
        payload = raw ? JSON.parse(raw) : {};
    } catch {
        throw new Error(
            `Unexpected response received from ${endpoint}.`
        );
    }

    if (!response.ok) {
        const detail = typeof payload.detail === 'string'
            ? payload.detail
            : `HTTP ${response.status}`;

        throw new Error(detail);
    }

    return payload;
}

async function loadDemoData() {
    const response = await fetch('demo-data.json', {
        cache: 'no-store',
    });

    if (!response.ok) {
        throw new Error('Demo data file could not be loaded.');
    }

    return response.json();
}


// ================================================
// 6. CONNECTION STATUS
// ================================================

function setConnection(source, message) {
    state.source = source;

    const live = source === 'live';

    const dotColor = live
        ? 'var(--green)'
        : source === 'demo'
            ? 'var(--amber)'
            : 'var(--red)';

    el.apiStatus.innerHTML = `
        <i class="status-dot"
           style="background:${dotColor}"></i>
        ${escapeHtml(message)}
    `;

    el.sidebarConnection.textContent = live
        ? 'API connected'
        : source === 'demo'
            ? 'Demo preview active'
            : 'API unavailable';

    el.modeBadge.textContent = live
        ? 'LIVE API DATA'
        : source === 'demo'
            ? 'DEMO DATA · SAMPLE ONLY'
            : 'DATA SOURCE UNKNOWN';

    el.modeBadge.style.color = live
        ? '#8ce0b7'
        : '#f1c477';

    el.modeBadge.style.background = live
        ? 'var(--green-soft)'
        : 'var(--amber-soft)';

    el.sidebarConnection.previousElementSibling.style.background =
        dotColor;

    el.settingsConnection.textContent = live
        ? 'Connected'
        : source === 'demo'
            ? 'Unavailable · showing sample data'
            : 'Unavailable';

    el.settingsSource.textContent = live
        ? 'FastAPI backend'
        : source === 'demo'
            ? 'demo-data.json'
            : 'Not available';

    el.settingsMode.textContent = live
        ? 'Live'
        : source === 'demo'
            ? 'Demo preview'
            : 'Unavailable';

    el.networkState.textContent = live
        ? 'Live connection'
        : 'Sample preview';
}


// ================================================
// 7. DECISION BADGES
// ================================================

function decisionBadge(decision) {
    const safe = String(decision || 'Unknown')
        .toLowerCase()
        .replace(/[^a-z-]/g, '');

    return `
        <span class="decision-badge decision-${safe}">
            ${escapeHtml(decision || 'Unknown')}
        </span>
    `;
}


// ================================================
// 8. METRICS
// ================================================

function renderMetrics(metrics) {
    const total = Number(metrics.total_transactions || 0);
    const approved = Number(metrics.approved || 0);
    const review = Number(metrics.review || 0);
    const blocked = Number(metrics.blocked || 0);
    const openCases = Number(metrics.open_review_cases || 0);

    el.total.textContent = formatNumber(total);
    el.approved.textContent = formatNumber(approved);
    el.review.textContent = formatNumber(review);
    el.blocked.textContent = formatNumber(blocked);

    el.anomaly.textContent = formatNumber(
        metrics.average_anomaly_score,
        3
    );

    el.networkTotal.textContent = formatNumber(total);
    el.networkApproved.textContent = formatNumber(approved);
    el.networkReview.textContent = formatNumber(review);
    el.networkBlocked.textContent = formatNumber(blocked);

    el.openCases.textContent =
        `${formatNumber(openCases)} open cases`;

    el.sidebarOpenCases.textContent = formatNumber(openCases);

    document.getElementById('model-anomaly-score').textContent =
        formatNumber(metrics.average_anomaly_score, 3);

    document.getElementById('average-latency').textContent =
        `${formatNumber(metrics.average_latency_ms, 1)} ms`;
}


// ================================================
// 9. MODEL INFORMATION
// ================================================

function renderModelInfo(modelInfo) {
    state.modelInfo = modelInfo || {};

    const primary = modelInfo.model || 'XGBoost';

    const secondary =
        modelInfo.secondary_model || 'PyTorch Autoencoder';

    el.primaryModel.textContent = primary;
    el.secondaryModel.textContent = secondary;

    document.getElementById('model-primary-title').textContent =
        primary;

    document.getElementById('model-secondary-title').textContent =
        secondary;

    document.getElementById('approval-threshold').textContent =
        formatNumber(modelInfo.approval_threshold, 2);

    document.getElementById('block-threshold').textContent =
        formatNumber(modelInfo.block_threshold, 2);
}


// ================================================
// 10. TRANSACTION TABLES
// ================================================

function renderRecentTransactions(rows) {
    if (!rows.length) {
        el.recentTransactions.innerHTML = `
            <div class="empty-inline">
                No transactions have been recorded yet.
            </div>
        `;
        return;
    }

    el.recentTransactions.innerHTML = rows.slice(0, 4).map(item => `
        <div class="mini-transaction">

            <div class="mini-transaction-main">
                <strong>
                    ${escapeHtml(
                        item.transaction_id || 'Unknown transaction'
                    )}
                </strong>

                <small>
                    ${escapeHtml(formatDate(item.created_at))}
                </small>
            </div>

            <span class="mini-transaction-amount">
                $${formatNumber(item.amount, 2)}
            </span>

            ${decisionBadge(item.decision)}

        </div>
    `).join('');
}

function renderTransactions(rows) {
    state.transactions = Array.isArray(rows) ? rows : [];

    el.transactionCount.textContent =
        `${formatNumber(state.transactions.length)} records`;

    if (!state.transactions.length) {
        el.transactionTable.innerHTML = `
            <tr>
                <td colspan="7" class="empty-state">
                    No transactions have been recorded yet.
                </td>
            </tr>
        `;

        renderRecentTransactions([]);
        return;
    }

    el.transactionTable.innerHTML = state.transactions.map(item => `
        <tr>

            <td class="id-cell">
                ${escapeHtml(item.transaction_id || 'Unknown')}
            </td>

            <td>
                $${formatNumber(item.amount, 2)}
            </td>

            <td>
                ${formatPercent(item.fraud_probability)}
            </td>

            <td>
                ${formatNumber(item.anomaly_score, 3)}
            </td>

            <td>
                ${decisionBadge(item.decision)}
            </td>

            <td>
                ${formatNumber(item.latency_ms, 1)} ms
            </td>

            <td>
                ${escapeHtml(formatDate(item.created_at))}
            </td>

        </tr>
    `).join('');

    renderRecentTransactions(state.transactions);
}


// ================================================
// 11. REVIEW QUEUE
// ================================================

function renderReviews(cases) {
    state.reviewCases = Array.isArray(cases) ? cases : [];

    const open = state.reviewCases.filter(
        item => String(item.status).toLowerCase() === 'open'
    );

    el.openCases.textContent = `${open.length} open cases`;

    el.sidebarOpenCases.textContent = formatNumber(open.length);

    if (!open.length) {
        el.reviewTable.innerHTML = `
            <tr>
                <td colspan="5" class="empty-state">
                    There are no open cases in the review queue.
                </td>
            </tr>
        `;

        return;
    }

    el.reviewTable.innerHTML = open.map(item => `
        <tr>

            <td class="id-cell">
                ${escapeHtml(item.transaction_id || 'Unknown')}
            </td>

            <td>
                ${decisionBadge(item.status)}
            </td>

            <td>
                ${escapeHtml(formatDate(item.created_at))}
            </td>

            <td>
                ${escapeHtml(item.analyst_decision || 'Pending')}
            </td>

            <td>

                <button
                    class="action-button"
                    type="button"
                    data-case-action="approve"
                    data-transaction-id="${escapeHtml(item.transaction_id)}"
                    ${state.source !== 'live' ? 'disabled title="Actions are disabled in demo mode"' : ''}>
                    Approve
                </button>

                <button
                    class="action-button reject"
                    type="button"
                    data-case-action="reject"
                    data-transaction-id="${escapeHtml(item.transaction_id)}"
                    ${state.source !== 'live' ? 'disabled title="Actions are disabled in demo mode"' : ''}>
                    Reject
                </button>

            </td>

        </tr>
    `).join('');
}


// ================================================
// 12. RENDER DASHBOARD
// ================================================

function renderAll(data, source) {
    state.metrics = data.metrics || {};
    state.lastUpdated = new Date();

    setConnection(
        source,
        source === 'live'
            ? 'API connected'
            : 'Demo preview'
    );

    renderMetrics(state.metrics);

    renderModelInfo(data.model_info || {});

    renderTransactions(data.transactions || []);

    renderReviews(data.review_cases || []);

    el.lastUpdated.textContent =
        state.lastUpdated.toLocaleTimeString([], {
            hour: '2-digit',
            minute: '2-digit',
        });
}


// ================================================
// 13. LOAD DASHBOARD DATA
// ================================================

async function loadDashboard() {
    const refresh = document.getElementById('refresh-button');

    refresh.disabled = true;
    refresh.classList.add('is-loading');

    try {
        const health = await fetchJson('/health');

        if (health.status !== 'healthy') {
            throw new Error(
                'API health check did not return healthy.'
            );
        }

        const [
            metrics,
            transactions,
            reviewCases,
            modelInfo,
        ] = await Promise.all([
            fetchJson('/metrics'),
            fetchJson('/transactions'),
            fetchJson('/review-cases'),
            fetchJson('/model-info'),
        ]);

        renderAll({
            metrics,
            transactions,
            review_cases: reviewCases,
            model_info: modelInfo,
        }, 'live');

    } catch (error) {
        console.warn(
            'Live API unavailable. Loading clearly labelled demo data.',
            error
        );

        try {
            const demoData = await loadDemoData();

            renderAll(demoData, 'demo');

        } catch (demoError) {
            setConnection('unavailable', 'API unavailable');

            console.error(
                'Unable to load demo data:',
                demoError
            );
        }

    } finally {
        refresh.disabled = false;
        refresh.classList.remove('is-loading');
    }
}


// ================================================
// 14. NAVIGATION
// ================================================

function navigate(viewName) {
    const selected = pageTitles[viewName]
        ? viewName
        : 'overview';

    document.querySelectorAll('.view-panel').forEach(panel => {
        const active = panel.id === `view-${selected}`;

        panel.hidden = !active;
        panel.classList.toggle('active', active);
    });

    document.querySelectorAll('[data-view]').forEach(button => {
        if (button.classList.contains('nav-item')) {
            button.classList.toggle(
                'active',
                button.dataset.view === selected
            );

            if (button.dataset.view === selected) {
                button.setAttribute('aria-current', 'page');
            } else {
                button.removeAttribute('aria-current');
            }
        }
    });

    document.getElementById('page-title').textContent =
        pageTitles[selected];

    document.getElementById('breadcrumb-current').textContent =
        pageTitles[selected].toUpperCase();

    if (selected === 'reports') {
        renderReports();
    }

    if (selected === 'settings') {
        el.apiEndpoint.textContent = API_BASE_URL;
    }

    window.scrollTo({
        top: 0,
        behavior: 'smooth',
    });
}


// ================================================
// 15. PREDICTION PAYLOAD
// ================================================

function buildPredictionPayload(amount, time) {
    const payload = {
        Time: Number(time),
        Amount: Number(amount),
    };

    for (let index = 1; index <= 28; index += 1) {
        payload[`V${index}`] = 0;
    }

    return payload;
}


// ================================================
// 16. CLEAR OLD PREDICTION
// ================================================

function clearPredictionOutput() {
    el.predictionResult.hidden = true;
    el.predictionResult.innerHTML = '';

    el.investigationSummary.hidden = true;
    el.summaryText.textContent = '';

    el.agentRecommendation.hidden = true;
    el.agentRecommendation.textContent = '';

    el.predictionPlaceholder.hidden = false;

    el.predictionPlaceholder.innerHTML = `
        <div class="placeholder-icon">⌁</div>

        <h3>Prediction results will appear here</h3>

        <p>
            Run a transaction through the model to see its
            decision, fraud probability, anomaly score
            and investigation summary.
        </p>
    `;

    state.lastReport = null;
}


// ================================================
// 17. DISPLAY PREDICTION RESULT
// ================================================

function renderPredictionResult(result) {
    const prediction = result.prediction || {};

    const decision = prediction.decision || 'Unknown';

    el.predictionPlaceholder.hidden = true;
    el.predictionResult.hidden = false;

    el.predictionResult.innerHTML = `
        <div class="prediction-result-top">

            <div>
                <p class="section-kicker">MODEL OUTPUT</p>

                <h3>Prediction complete</h3>

                <p>
                    The model returned the following result.
                </p>
            </div>

            ${decisionBadge(decision)}

        </div>

        <div class="result-metrics">

            <div class="result-metric">
                <span>Fraud probability</span>

                <strong>
                    ${formatPercent(prediction.fraud_probability)}
                </strong>
            </div>

            <div class="result-metric">
                <span>Anomaly score</span>

                <strong>
                    ${formatNumber(prediction.anomaly_score, 3)}
                </strong>
            </div>

            <div class="result-metric">
                <span>Latency</span>

                <strong>
                    ${formatNumber(result.latency_ms, 1)} ms
                </strong>
            </div>

        </div>
    `;

    const summary =
        typeof result.investigation_summary === 'string'
            ? result.investigation_summary.trim()
            : '';

    el.investigationSummary.hidden = !summary;

    el.summaryText.textContent = summary;

    const recommendation = result.agent_recommendation || {};

    if (
        recommendation.recommended_action ||
        recommendation.reason
    ) {
        el.agentRecommendation.hidden = false;

        el.agentRecommendation.innerHTML = `
            <strong>
                Workflow recommendation ·
                ${escapeHtml(
                    recommendation.recommended_action ||
                    'Not provided'
                )}
            </strong>

            ${escapeHtml(recommendation.reason || '')}
        `;
    } else {
        el.agentRecommendation.hidden = true;
        el.agentRecommendation.textContent = '';
    }

    state.lastReport = {
        summary: summary || 'No investigation summary was returned by the API.',
        decision,
        probability: prediction.fraud_probability,
        amount: document.getElementById('amount-input').value,
        createdAt: new Date(),
        recommendation,
    };
}


// ================================================
// 18. SUBMIT PREDICTION
// ================================================

async function submitPrediction(event) {
    event.preventDefault();

    if (state.source !== 'live') {
        el.predictionPlaceholder.hidden = false;

        el.predictionPlaceholder.innerHTML = `
            <div class="placeholder-icon">!</div>

            <h3>Live API required</h3>

            <p>
                Prediction requests are disabled in demo preview mode.
                Connect the FastAPI backend and refresh the dashboard.
            </p>
        `;

        return;
    }

    const formData = new FormData(el.predictionForm);

    const amount = formData.get('Amount');
    const time = formData.get('Time');

    if (
        !Number.isFinite(Number(amount)) ||
        !Number.isFinite(Number(time)) ||
        amount === '' ||
        time === ''
    ) {
        return;
    }

    el.predictionSubmit.disabled = true;
    el.predictionSubmit.textContent = 'Running prediction…';

    try {
        const result = await fetchJson('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(
                buildPredictionPayload(amount, time)
            ),
        });

        renderPredictionResult(result);

        navigate('prediction');

        await loadDashboard();

    } catch (error) {
        el.predictionPlaceholder.hidden = false;

        el.predictionPlaceholder.innerHTML = `
            <div class="placeholder-icon">!</div>

            <h3>Prediction failed</h3>

            <p>
                ${escapeHtml(error.message)}.
                Check the API connection and try again.
            </p>
        `;

        console.error('Prediction failed:', error);

    } finally {
        el.predictionSubmit.disabled = false;

        el.predictionSubmit.innerHTML =
            'Run prediction <span>→</span>';
    }
}


// ================================================
// 19. APPROVE / REJECT CASE
// ================================================

async function updateCase(transactionId, action, button) {
    if (state.source !== 'live') return;

    if (button) {
        button.disabled = true;
    }

    try {
        await fetchJson(`/${action}-case`, {
            method: 'POST',

            headers: {
                'Content-Type': 'application/json',
            },

            body: JSON.stringify({
                transaction_id: transactionId,
                notes: `${action}d from dashboard`,
            }),
        });

        await loadDashboard();

    } catch (error) {
        console.error('Case update failed:', error);

        window.alert(
            `Could not ${action} this case: ${error.message}`
        );

        if (button) {
            button.disabled = false;
        }
    }
}


// ================================================
// 20. AI INVESTIGATION REPORTS
// ================================================

function renderReports() {
    if (!state.lastReport) {
        el.reportsList.innerHTML = `
            <article class="surface-card report-empty">

                <div class="placeholder-icon">✎</div>

                <h3>No reports in this session</h3>

                <p>
                    Run a prediction to generate an investigation summary.
                    Reports are not stored separately by the dashboard.
                </p>

                <button
                    class="secondary-button"
                    type="button"
                    data-view="prediction">
                    Run a prediction
                </button>

            </article>
        `;

        return;
    }

    const report = state.lastReport;

    el.reportsList.innerHTML = `
        <article class="surface-card report-item">

            <div class="card-heading">

                <div>
                    <p class="section-kicker">LATEST REPORT</p>

                    <h3>Transaction assessment</h3>

                    <p class="report-meta">
                        ${escapeHtml(report.createdAt.toLocaleString())}
                        · Amount $${formatNumber(report.amount, 2)}
                    </p>
                </div>

                ${decisionBadge(report.decision)}

            </div>

            <div class="report-meta">
                Fraud probability:
                ${formatPercent(report.probability)}
            </div>

            <pre>${escapeHtml(report.summary)}</pre>

        </article>
    `;
}


// ================================================
// 21. EVENT LISTENERS
// ================================================

// Navigation
document.addEventListener('click', event => {
    const viewButton = event.target.closest('[data-view]');

    if (viewButton) {
        event.preventDefault();

        navigate(viewButton.dataset.view);

        return;
    }

    // Review actions
    const actionButton = event.target.closest(
        '[data-case-action]'
    );

    if (actionButton) {
        updateCase(
            actionButton.dataset.transactionId,
            actionButton.dataset.caseAction,
            actionButton
        );
    }
});

// Refresh dashboard
document
    .getElementById('refresh-button')
    .addEventListener('click', loadDashboard);

// Submit prediction
el.predictionForm.addEventListener(
    'submit',
    submitPrediction
);

// IMPORTANT: Clear old results when inputs change.
document
    .getElementById('amount-input')
    .addEventListener('input', clearPredictionOutput);

document
    .getElementById('time-input')
    .addEventListener('input', clearPredictionOutput);


// ================================================
// 22. INITIALIZE
// ================================================

el.apiEndpoint.textContent = API_BASE_URL;

loadDashboard();

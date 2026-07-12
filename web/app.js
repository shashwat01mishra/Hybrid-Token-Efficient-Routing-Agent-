// Global App State
let systemConfig = null;
let allLogs = [];

// DOM Elements
const systemConfigEl = document.getElementById('system-config');
const promptInput = document.getElementById('prompt-input');
const runBtn = document.getElementById('run-btn');
const runBtnText = runBtn.querySelector('.btn-text');
const resetFlowBtn = document.getElementById('reset-flow-btn');
const categoryBadge = document.getElementById('category-badge');

// Stats Elements
const avgLatencyEl = document.getElementById('avg-latency');
const totalCostEl = document.getElementById('total-cost');
const escalateRateEl = document.getElementById('escalate-rate');

// Telemetry Elements
const telRoute = document.getElementById('tel-route');
const telLatency = document.getElementById('tel-latency');
const telCost = document.getElementById('tel-cost');
const telMeanLp = document.getElementById('tel-mean-lp');
const telMinLp = document.getElementById('tel-min-lp');
const gaugeMean = document.getElementById('gauge-mean');
const gaugeMin = document.getElementById('gauge-min');
const telReason = document.getElementById('tel-reason');
const outputBox = document.getElementById('output-box');

// History Table Elements
const historyRows = document.getElementById('history-rows');
const historySearch = document.getElementById('history-search');

// Initial Setup
document.addEventListener('DOMContentLoaded', () => {
    fetchConfig();
    fetchLogs();
    setupEventListeners();
});

// Event Listeners Setup
function setupEventListeners() {
    // Preset Buttons
    document.querySelectorAll('.preset-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const prompt = btn.getAttribute('data-prompt');
            const cat = btn.getAttribute('data-category');
            promptInput.value = prompt;
            categoryBadge.textContent = cat ? cat.replace('_', ' ') : 'Auto-detect';
            categoryBadge.className = `badge badge-${cat === 'math' ? 'math' : (cat.startsWith('code') ? 'verify' : 'primary')}`;
            
            // Visual bounce feedback on clicking preset
            btn.style.transform = 'scale(0.96)';
            setTimeout(() => btn.style.transform = 'scale(1)', 100);
        });
    });

    // Run Button
    runBtn.addEventListener('click', executePrompt);

    // Reset Flow Button
    resetFlowBtn.addEventListener('click', () => {
        resetFlowHighlight();
        clearTelemetry();
    });

    // Search Box Filter
    historySearch.addEventListener('input', filterHistoryLogs);
}

// Fetch System Configuration
async function fetchConfig() {
    try {
        const response = await fetch('/api/config');
        if (!response.ok) throw new Error('HTTP error ' + response.status);
        systemConfig = await response.json();
        renderConfig(systemConfig);
    } catch (err) {
        console.error('Error loading config:', err);
        systemConfigEl.innerHTML = `<div class="config-item" style="color: var(--red);">Failed to load config. Make sure server is running.</div>`;
    }
}

// Render Settings in Sidebar
function renderConfig(config) {
    systemConfigEl.innerHTML = '';
    
    const settings = [
        { label: 'Local Model', val: config.LOCAL_MODEL },
        { label: 'Local Backend', val: config.LOCAL_BACKEND },
        { label: 'Remote Model', val: config.FIREWORKS_MODEL },
        { label: 'Mean LP Threshold', val: config.MEAN_LOGPROB_THRESHOLD },
        { label: 'Min LP Threshold', val: config.MIN_LOGPROB_THRESHOLD },
        { label: 'Mock Model Mode', val: config.MOCK_LOCAL_MODEL ? 'ON (Simulated)' : 'OFF (Inference)' },
        { label: 'Mock Remote Tier', val: config.MOCK_REMOTE_CLIENT ? 'ON (Simulated)' : 'OFF (Fireworks)' }
    ];

    settings.forEach(s => {
        const item = document.createElement('div');
        item.className = 'config-item';
        
        let valStyle = '';
        if (s.label.includes('Mock') && s.val.includes('ON')) {
            valStyle = 'color: var(--amber); font-weight: bold;';
        } else if (s.label.includes('Mock') && s.val.includes('OFF')) {
            valStyle = 'color: var(--green);';
        }

        item.innerHTML = `
            <span class="label">${s.label}</span>
            <span class="val" style="${valStyle}">${s.val}</span>
        `;
        systemConfigEl.appendChild(item);
    });
}

// Fetch Run Logs and Render
async function fetchLogs() {
    try {
        const response = await fetch('/api/logs');
        if (!response.ok) throw new Error('HTTP error ' + response.status);
        allLogs = await response.json();
        
        renderStats(allLogs);
        renderHistoryTable(allLogs);
    } catch (err) {
        console.error('Error fetching logs:', err);
        historyRows.innerHTML = `<tr><td colspan="8" class="loading-text" style="color: var(--red);">Error retrieving logs.</td></tr>`;
    }
}

// Calculate Stats from Logs
function renderStats(logs) {
    if (!logs || logs.length === 0) {
        avgLatencyEl.textContent = '0 ms';
        totalCostEl.textContent = '$0.00';
        escalateRateEl.textContent = '0%';
        return;
    }

    let totalLatency = 0;
    let totalCost = 0;
    let escalationCount = 0;
    let validLogsCount = 0;

    logs.forEach(log => {
        if (log.latency_ms) {
            totalLatency += log.latency_ms;
            validLogsCount++;
        }
        if (log.cost_usd) {
            totalCost += log.cost_usd;
        }
        
        // Count anything routed to remote as escalated
        if (log.route && (log.route.includes('remote') || log.route.includes('escalate'))) {
            escalationCount++;
        }
    });

    const avgLatency = validLogsCount > 0 ? (totalLatency / validLogsCount).toFixed(1) : 0;
    const escalationRate = logs.length > 0 ? ((escalationCount / logs.length) * 100).toFixed(1) : 0;

    avgLatencyEl.textContent = `${avgLatency} ms`;
    totalCostEl.textContent = `$${totalCost.toFixed(5)}`;
    escalateRateEl.textContent = `${escalationRate}%`;
}

// Render Logs in History Table
function renderHistoryTable(logs) {
    historyRows.innerHTML = '';
    
    if (logs.length === 0) {
        historyRows.innerHTML = `<tr><td colspan="8" class="loading-text">No runs recorded yet. Write a prompt to test.</td></tr>`;
        return;
    }

    logs.forEach((log, index) => {
        const tr = document.createElement('tr');
        tr.dataset.index = index;
        
        // Formatted timestamp
        let dateStr = '-';
        if (log.timestamp) {
            try {
                const date = new Date(log.timestamp);
                dateStr = date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
            } catch (e) {}
        }

        // Badges for routes
        let badgeClass = 'badge-primary';
        const route = log.route ? log.route.toLowerCase() : '';
        if (route.includes('math')) badgeClass = 'badge-math';
        else if (route.includes('remote')) badgeClass = 'badge-remote';
        else if (route.includes('local')) badgeClass = 'badge-local';
        else if (route.includes('verify')) badgeClass = 'badge-verify';
        else if (route.includes('error')) badgeClass = 'badge-error';

        const meanLp = log.local_mean_logprob !== undefined ? log.local_mean_logprob.toFixed(3) : '-';
        const minLp = log.local_min_logprob !== undefined ? log.local_min_logprob.toFixed(3) : '-';
        const latency = log.latency_ms ? `${log.latency_ms.toFixed(0)} ms` : '-';
        const cost = log.cost_usd ? `$${log.cost_usd.toFixed(5)}` : '$0.00000';

        tr.innerHTML = `
            <td>${dateStr}</td>
            <td class="task-col" title="${escapeHtml(log.task || '')}">${escapeHtml(log.task || '')}</td>
            <td><span class="badge badge-primary">${escapeHtml(log.category || 'factual')}</span></td>
            <td><span class="badge ${badgeClass}">${escapeHtml(log.route || 'local')}</span></td>
            <td style="font-family: monospace;">${meanLp}</td>
            <td style="font-family: monospace;">${minLp}</td>
            <td>${latency}</td>
            <td>${cost}</td>
        `;

        tr.addEventListener('click', () => {
            // Remove previous selected row styles
            document.querySelectorAll('.history-table tbody tr').forEach(r => r.classList.remove('selected-row'));
            tr.classList.add('selected-row');
            
            // Load this history data to visually display it
            loadHistoryToVisuals(log);
        });

        historyRows.appendChild(tr);
    });
}

// Filter logs via search
function filterHistoryLogs() {
    const q = historySearch.value.toLowerCase().trim();
    if (!q) {
        renderHistoryTable(allLogs);
        return;
    }

    const filtered = allLogs.filter(log => {
        const task = (log.task || '').toLowerCase();
        const answer = (log.answer || '').toLowerCase();
        const reason = (log.reason || '').toLowerCase();
        const category = (log.category || '').toLowerCase();
        const route = (log.route || '').toLowerCase();
        return task.includes(q) || answer.includes(q) || reason.includes(q) || category.includes(q) || route.includes(q);
    });
    
    renderHistoryTable(filtered);
}

// Load a selected log row back to visual flowchart and telemetry cards
function loadHistoryToVisuals(log) {
    resetFlowHighlight();
    
    // Set outputs
    outputBox.textContent = log.answer || '';
    telRoute.textContent = (log.route || 'local').toUpperCase();
    
    // Set route badge classes
    let route = log.route ? log.route.toLowerCase() : 'local';
    let badgeClass = 'badge-local';
    if (route.includes('math')) badgeClass = 'badge-math';
    else if (route.includes('remote') || route.includes('escalate')) badgeClass = 'badge-remote';
    else if (route.includes('verify')) badgeClass = 'badge-verify';
    telRoute.className = `val badge ${badgeClass}`;

    telLatency.textContent = log.latency_ms ? `${log.latency_ms.toFixed(1)} ms` : '-';
    telCost.textContent = log.cost_usd ? `$${log.cost_usd.toFixed(6)}` : '$0.000000';
    telReason.textContent = log.reason || 'Confidence logs successfully loaded.';

    const meanLp = log.local_mean_logprob !== undefined ? log.local_mean_logprob : 0;
    const minLp = log.local_min_logprob !== undefined ? log.local_min_logprob : 0;

    updateLogprobGauges(meanLp, minLp);

    const isEscalated = route.includes('remote') || route.includes('escalate');
    const isVerifyFailed = route.includes('verify_escalate') || route.includes('verify_retry');

    // Trigger visual routing chart animation immediately without wait state
    playFlowAnimation(
        log.category || 'factual', 
        log.route || 'local', 
        meanLp, 
        minLp, 
        isEscalated, 
        isVerifyFailed,
        150 // Faster trace speed for history load
    );
}

// Reset flow diagram visual styles
function resetFlowHighlight() {
    // Clear all highlighted elements
    document.querySelectorAll('.flow-node, .flow-arrow, .branch-line').forEach(el => {
        el.className = el.className.split(' ').filter(c => 
            !c.startsWith('active') && 
            !c.startsWith('highlight-')
        ).join(' ');
    });

    // Reset flowchart node subtexts
    document.getElementById('flow-category').textContent = '-';
    document.getElementById('flow-math-expr').textContent = '-';
    document.getElementById('flow-local-tokens').textContent = '-';
    document.getElementById('flow-syntax-status').textContent = '-';
    document.getElementById('flow-confidence-status').textContent = '-';
    document.getElementById('flow-remote-model-name').textContent = '-';
    document.getElementById('flow-route-chosen').textContent = '-';

    // Show/hide correct code gate pathing
    document.querySelector('.flow-card').classList.remove('code-mode');
}

// Clear telemetry display card
function clearTelemetry() {
    telRoute.textContent = '-';
    telRoute.className = 'val badge';
    telLatency.textContent = '-';
    telCost.textContent = '-';
    telMeanLp.textContent = '-';
    telMinLp.textContent = '-';
    gaugeMean.style.width = '0%';
    gaugeMin.style.width = '0%';
    telReason.textContent = 'Select a query to view decision trace...';
    outputBox.textContent = 'Output will appear here after execution...';
}

// Map logprob value to a percentage gauge
function getLogprobPercent(lp, isMean) {
    const threshold = isMean 
        ? (systemConfig ? systemConfig.MEAN_LOGPROB_THRESHOLD : -0.5)
        : (systemConfig ? systemConfig.MIN_LOGPROB_THRESHOLD : -2.0);
    
    // Scale ranges from -4.0 (0%) to 0.0 (100%)
    const minScale = -4.0;
    const maxScale = 0.0;
    
    if (lp >= maxScale) return 100;
    if (lp <= minScale) return 0;
    
    return ((lp - minScale) / (maxScale - minScale)) * 100;
}

// Update logprob progress bar gauges
function updateLogprobGauges(meanLp, minLp) {
    telMeanLp.textContent = meanLp !== undefined ? meanLp.toFixed(3) : '-';
    telMinLp.textContent = minLp !== undefined ? minLp.toFixed(3) : '-';

    const meanPct = getLogprobPercent(meanLp, true);
    const minPct = getLogprobPercent(minLp, false);

    gaugeMean.style.width = `${meanPct}%`;
    gaugeMin.style.width = `${minPct}%`;

    // Apply color logic
    const meanThreshold = systemConfig ? systemConfig.MEAN_LOGPROB_THRESHOLD : -0.5;
    const minThreshold = systemConfig ? systemConfig.MIN_LOGPROB_THRESHOLD : -2.0;

    gaugeMean.style.backgroundColor = meanLp < meanThreshold ? 'var(--amber)' : 'var(--green)';
    gaugeMin.style.backgroundColor = minLp < minThreshold ? 'var(--amber)' : 'var(--green)';
}

// Run routing agent loop on server
async function executePrompt() {
    const prompt = promptInput.value.trim();
    if (!prompt) return;

    // Loading State UI
    runBtn.disabled = true;
    runBtnText.textContent = 'Executing agent inference...';
    runBtn.querySelector('.btn-icon').setAttribute('data-lucide', 'loader');
    lucide.createIcons();
    
    resetFlowHighlight();
    clearTelemetry();

    try {
        const response = await fetch('/api/run', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ task: prompt })
        });

        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.error || 'Server error');
        }

        const trace = await response.json();
        
        // Render Output text
        outputBox.textContent = trace.final_answer || '';
        
        // Update statistics and history logs
        await fetchLogs();

        // Render decision telemetry cards
        telRoute.textContent = (trace.final_route || 'local').toUpperCase();
        let badgeClass = 'badge-local';
        const route = trace.final_route ? trace.final_route.toLowerCase() : '';
        if (route.includes('math')) badgeClass = 'badge-math';
        else if (route.includes('remote') || route.includes('escalate')) badgeClass = 'badge-remote';
        else if (route.includes('verify')) badgeClass = 'badge-verify';
        telRoute.className = `val badge ${badgeClass}`;

        telLatency.textContent = `${trace.total_latency_ms.toFixed(1)} ms`;
        telCost.textContent = `$${trace.total_cost_usd.toFixed(6)}`;
        telReason.textContent = trace.routing_decision ? trace.routing_decision.reason : '-';

        const meanLp = trace.local_generation ? trace.local_generation.mean_logprob : 0;
        const minLp = trace.local_generation ? trace.local_generation.min_logprob : 0;
        updateLogprobGauges(meanLp, minLp);

        // Perform animation mapping
        const isEscalated = route.includes('remote') || route.includes('escalate');
        const isVerifyFailed = trace.code_verify_failed || false;

        playFlowAnimation(
            trace.category, 
            trace.final_route, 
            meanLp, 
            minLp, 
            isEscalated, 
            isVerifyFailed,
            300 // Standard animation step delay
        );

    } catch (err) {
        console.error('Execution failed:', err);
        outputBox.textContent = `CRITICAL LOOP FAILURE:\n\n${err.message}`;
        telReason.textContent = 'API network request or local execution failed. Verify model loading and server console.';
        telRoute.textContent = 'ERROR';
        telRoute.className = 'val badge badge-error';
    } finally {
        // Reset Button UI State
        runBtn.disabled = false;
        runBtnText.textContent = 'Execute Agent Loop';
        runBtn.querySelector('.btn-icon').setAttribute('data-lucide', 'play');
        lucide.createIcons();
    }
}

// Sequential flow path animation
function playFlowAnimation(category, finalRoute, meanLp, minLp, isEscalated, isVerifyFailed, stepDelay = 300) {
    const steps = ['node-start', 'arrow-start-to-classify', 'node-classify'];
    
    const flowCard = document.querySelector('.flow-card');
    
    // Code Mode toggles node layouts
    if (category === 'code_debugging' || category === 'code_generation') {
        flowCard.classList.add('code-mode');
        steps.push('branch-code-line', 'node-local-model', 'arrow-local-to-gate', 'node-code-gate');
        if (isVerifyFailed) {
            steps.push('arrow-code-to-gate', 'node-confidence-gate', 'arrow-gate-to-escalate', 'node-remote-model');
        } else {
            steps.push('arrow-code-to-gate', 'node-confidence-gate');
            if (isEscalated) {
                steps.push('arrow-gate-to-escalate', 'node-remote-model');
            } else {
                steps.push('arrow-gate-to-keep');
            }
        }
    } else if (category === 'math') {
        flowCard.classList.remove('code-mode');
        steps.push('branch-math-line', 'node-math-eval');
    } else {
        flowCard.classList.remove('code-mode');
        steps.push('branch-standard-line', 'node-local-model', 'arrow-local-to-gate', 'node-confidence-gate');
        if (isEscalated) {
            steps.push('arrow-gate-to-escalate', 'node-remote-model');
        } else {
            steps.push('arrow-gate-to-keep');
        }
    }
    
    steps.push('node-end');

    // Run active triggers step by step
    let delay = 0;
    steps.forEach((stepId) => {
        setTimeout(() => {
            const el = document.getElementById(stepId);
            if (!el) return;

            el.classList.add('active');

            // Apply node-specific metadata on activation
            if (stepId === 'node-classify') {
                document.getElementById('flow-category').textContent = (category || 'factual').toUpperCase();
            } else if (stepId === 'node-math-eval') {
                document.getElementById('flow-math-expr').textContent = 'Solved via safe AST eval()';
            } else if (stepId === 'node-local-model') {
                document.getElementById('flow-local-tokens').textContent = `Mean LP: ${meanLp !== undefined ? meanLp.toFixed(3) : '-'}`;
            } else if (stepId === 'node-code-gate') {
                document.getElementById('flow-syntax-status').textContent = isVerifyFailed ? 'SYNTAX FAILURE' : 'Syntax Check Passed';
                el.classList.add(isVerifyFailed ? 'highlight-amber' : 'highlight-green');
            } else if (stepId === 'node-confidence-gate') {
                const status = isEscalated ? 'Low Confidence' : 'High Confidence';
                document.getElementById('flow-confidence-status').textContent = status;
                el.classList.add(isEscalated ? 'highlight-amber' : 'highlight-green');
            } else if (stepId === 'node-remote-model') {
                document.getElementById('flow-remote-model-name').textContent = 'Escalated response';
            } else if (stepId === 'node-end') {
                document.getElementById('flow-route-chosen').textContent = finalRoute.toUpperCase();
                
                // Colorize the end node
                if (finalRoute.includes('math')) {
                    el.classList.add('highlight-purple');
                } else if (finalRoute.includes('remote') || finalRoute.includes('escalate')) {
                    el.classList.add('highlight-blue');
                } else if (finalRoute.includes('verify')) {
                    el.classList.add('highlight-amber');
                } else {
                    el.classList.add('highlight-green');
                }
            }
        }, delay);
        delay += stepDelay;
    });
}

// Utility to escape HTML strings safely
function escapeHtml(str) {
    return str
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

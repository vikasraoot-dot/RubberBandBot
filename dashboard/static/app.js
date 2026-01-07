// Dashboard Application JavaScript V2

let pnlChart = null;
let tradesChart = null;

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    // Set default date to today
    const today = new Date().toISOString().split('T')[0];
    document.getElementById('dateSelect').value = today;

    // Load initial data
    loadDashboard();

    // Event listeners - ALL filters trigger full reload
    document.getElementById('refreshBtn').addEventListener('click', loadDashboard);
    document.getElementById('dateSelect').addEventListener('change', loadDashboard);
    document.getElementById('periodSelect').addEventListener('change', loadDashboard);
    document.getElementById('botSelect').addEventListener('change', loadDashboard);
});

async function loadDashboard() {
    const date = document.getElementById('dateSelect').value;
    const period = document.getElementById('periodSelect').value;
    const bot = document.getElementById('botSelect').value;

    // Update date label
    document.getElementById('selectedDateLabel').textContent = date;

    try {
        // Load portfolio value
        const portfolioRes = await fetch('/api/portfolio');
        if (portfolioRes.ok) {
            const portfolio = await portfolioRes.json();
            updatePortfolioCard(portfolio);
        }

        // Load PnL summary (filtered by bot)
        const pnlRes = await fetch(`/api/pnl?date=${date}&bot=${bot}`);
        const pnl = await pnlRes.json();
        updateSummaryCards(pnl, bot);
        updateBotCards(pnl);

        // Load positions
        const posRes = await fetch('/api/positions');
        const positions = await posRes.json();
        updatePositionsTables(positions);

        // Load charts (filtered by bot and period)
        await loadCharts(period, bot);

        // Update timestamp
        document.getElementById('lastUpdate').textContent = new Date().toLocaleTimeString();
    } catch (err) {
        console.error('Error loading dashboard:', err);
    }
}

function updatePortfolioCard(data) {
    document.getElementById('portfolioValue').textContent = formatCurrencyLarge(data.portfolio_value);

    const dayChangeEl = document.getElementById('dayChange');
    dayChangeEl.textContent = formatCurrency(data.day_change);
    dayChangeEl.style.color = data.day_change >= 0 ? 'var(--accent-green)' : 'var(--accent-red)';

    document.getElementById('cashBalance').textContent = formatCurrencyLarge(data.cash);
    document.getElementById('buyingPower').textContent = formatCurrencyLarge(data.buying_power);
}

function updateSummaryCards(data, botFilter) {
    const { totals } = data;

    document.getElementById('realizedPnl').textContent = formatCurrency(totals.realized);
    document.getElementById('unrealizedPnl').textContent = formatCurrency(totals.unrealized);
    document.getElementById('netPnl').textContent = formatCurrency(totals.net);

    // Count entries based on filter
    let totalEntries;
    if (botFilter === 'all') {
        totalEntries = Object.values(data.bots).reduce((sum, bot) => sum + (bot.entries || 0), 0);
    } else {
        totalEntries = data.bots[botFilter]?.entries || 0;
    }
    document.getElementById('totalTrades').textContent = totalEntries;

    // Color coding
    applyPnlColor('realizedPnl', totals.realized);
    applyPnlColor('unrealizedPnl', totals.unrealized);
    applyPnlColor('netPnl', totals.net);
}

function updateBotCards(data) {
    const bots = data.bots;

    // 15m Stock
    const stk = bots['15m_stock'];
    const stkCard = document.getElementById('card15mStock');
    stkCard.querySelector('.entries b').textContent = stk.entries || 0;
    stkCard.querySelector('.trades b').textContent = stk.trades;
    stkCard.querySelector('.winrate b').textContent = stk.trades > 0 ?
        (stk.winners / stk.trades * 100).toFixed(0) + '%' : '0%';
    updatePnlElement(stkCard.querySelector('.pnl b'), stk.realized_pnl);

    // 15m Options
    const opt = bots['15m_options'];
    const optCard = document.getElementById('card15mOptions');
    optCard.querySelector('.entries b').textContent = opt.entries || 0;
    optCard.querySelector('.open b').textContent = opt.open_count || 0;
    updatePnlElement(optCard.querySelector('.realized b'), opt.realized_pnl);
    updatePnlElement(optCard.querySelector('.unrealized b'), opt.unrealized_pnl || 0);

    // Weekly Stock
    const wkStk = bots['weekly_stock'];
    const wkStkCard = document.getElementById('cardWeeklyStock');
    wkStkCard.querySelector('.entries b').textContent = wkStk.entries || 0;
    wkStkCard.querySelector('.trades b').textContent = wkStk.trades;
    wkStkCard.querySelector('.winrate b').textContent = wkStk.trades > 0 ?
        (wkStk.winners / wkStk.trades * 100).toFixed(0) + '%' : '0%';
    updatePnlElement(wkStkCard.querySelector('.pnl b'), wkStk.realized_pnl);

    // Weekly Options
    const wkOpt = bots['weekly_options'];
    const wkOptCard = document.getElementById('cardWeeklyOptions');
    wkOptCard.querySelector('.entries b').textContent = wkOpt.entries || 0;
    wkOptCard.querySelector('.trades b').textContent = wkOpt.trades;
    wkOptCard.querySelector('.winrate b').textContent = wkOpt.trades > 0 ?
        (wkOpt.winners / wkOpt.trades * 100).toFixed(0) + '%' : '0%';
    updatePnlElement(wkOptCard.querySelector('.pnl b'), wkOpt.realized_pnl);
}

function updatePnlElement(el, value) {
    el.textContent = formatCurrency(value);
    el.style.color = value >= 0 ? 'var(--accent-green)' : 'var(--accent-red)';
}

function updatePositionsTables(positions) {
    // Stocks table
    const stocksTbody = document.querySelector('#stocksTable tbody');
    stocksTbody.innerHTML = '';

    positions.stocks.forEach(pos => {
        const row = document.createElement('tr');
        const pnlClass = pos.unrealized_pnl >= 0 ? 'positive' : 'negative';
        row.innerHTML = `
            <td>${pos.symbol}</td>
            <td>${pos.qty}</td>
            <td>$${pos.avg_entry.toFixed(2)}</td>
            <td>$${pos.current_price.toFixed(2)}</td>
            <td class="${pnlClass}">${formatCurrency(pos.unrealized_pnl)}</td>
        `;
        stocksTbody.appendChild(row);
    });

    // Options table (consolidated)
    const optionsTbody = document.querySelector('#optionsTable tbody');
    optionsTbody.innerHTML = '';

    positions.options.forEach(pos => {
        const row = document.createElement('tr');
        const pnlClass = pos.unrealized_pnl >= 0 ? 'positive' : 'negative';
        const typeBadge = pos.type === 'spread' ?
            '<span class="spread-badge">SPREAD</span>' :
            '<span class="single-badge">SINGLE</span>';

        row.innerHTML = `
            <td>${pos.description}${typeBadge}</td>
            <td>${pos.type === 'spread' ? 'Spread' : 'Single'}</td>
            <td>${Math.abs(pos.qty)}</td>
            <td>$${Math.abs(pos.avg_entry).toFixed(2)}</td>
            <td>$${Math.abs(pos.current_price).toFixed(2)}</td>
            <td class="${pnlClass}">${formatCurrency(pos.unrealized_pnl)}</td>
        `;
        optionsTbody.appendChild(row);
    });
}

async function loadCharts(period, bot) {
    try {
        // PnL Chart (filtered by bot)
        const pnlRes = await fetch(`/api/chart/pnl?period=${period}&bot=${bot}`);
        const pnlData = await pnlRes.json();
        renderPnlChart(pnlData);

        // Trades Chart (filtered by bot)
        const tradesRes = await fetch(`/api/chart/trades?period=${period}&bot=${bot}`);
        const tradesData = await tradesRes.json();
        renderTradesChart(tradesData);
    } catch (err) {
        console.error('Error loading charts:', err);
    }
}

function renderPnlChart(data) {
    const ctx = document.getElementById('pnlChart').getContext('2d');

    if (pnlChart) pnlChart.destroy();

    const labels = data.map(d => d.date);
    const values = data.map(d => d.pnl);

    // Cumulative PnL
    const cumulative = values.reduce((acc, val, i) => {
        acc.push((acc[i - 1] || 0) + val);
        return acc;
    }, []);

    pnlChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels,
            datasets: [{
                label: 'Cumulative PnL',
                data: cumulative,
                borderColor: '#58a6ff',
                backgroundColor: 'rgba(88, 166, 255, 0.1)',
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            plugins: { legend: { display: false } },
            scales: {
                y: { grid: { color: '#30363d' }, ticks: { color: '#8b949e' } },
                x: { grid: { display: false }, ticks: { color: '#8b949e' } }
            }
        }
    });
}

function renderTradesChart(data) {
    const ctx = document.getElementById('tradesChart').getContext('2d');

    if (tradesChart) tradesChart.destroy();

    tradesChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: data.map(d => d.date),
            datasets: [{
                label: 'Entries',
                data: data.map(d => d.count),
                backgroundColor: '#a371f7',
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            plugins: { legend: { display: false } },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: { color: '#30363d' },
                    ticks: { color: '#8b949e', stepSize: 1 }
                },
                x: { grid: { display: false }, ticks: { color: '#8b949e' } }
            },
            onClick: async (event, elements) => {
                if (elements.length > 0) {
                    const index = elements[0].index;
                    const clickedDate = data[index].date;
                    const bot = document.getElementById('botSelect').value;
                    await showTradeDetails(clickedDate, bot);
                }
            }
        }
    });
}

async function showTradeDetails(date, bot) {
    try {
        const res = await fetch(`/api/trades?date=${date}&bot=${bot}`);
        const trades = await res.json();

        // Create modal if doesn't exist
        let modal = document.getElementById('tradeModal');
        if (!modal) {
            modal = document.createElement('div');
            modal.id = 'tradeModal';
            modal.className = 'modal';
            modal.innerHTML = `
                <div class="modal-content">
                    <div class="modal-header">
                        <h3>Trade Details - <span id="modalDate"></span></h3>
                        <button onclick="closeModal()">✕</button>
                    </div>
                    <div class="modal-body" id="modalBody"></div>
                </div>
            `;
            document.body.appendChild(modal);
        }

        document.getElementById('modalDate').textContent = date;
        const body = document.getElementById('modalBody');

        if (trades.length === 0) {
            body.innerHTML = '<p>No trades found for this date and filter.</p>';
        } else {
            let html = '<table class="trades-table"><thead><tr><th>Symbol</th><th>Side</th><th>Qty</th><th>Price</th><th>Bot</th></tr></thead><tbody>';
            trades.forEach(t => {
                html += `<tr>
                    <td>${t.symbol}</td>
                    <td class="${t.side === 'buy' ? 'positive' : 'negative'}">${t.side.toUpperCase()}</td>
                    <td>${t.qty}</td>
                    <td>$${t.price.toFixed(2)}</td>
                    <td>${t.category}</td>
                </tr>`;
            });
            html += '</tbody></table>';
            body.innerHTML = html;
        }

        modal.style.display = 'flex';
    } catch (err) {
        console.error('Error loading trade details:', err);
    }
}

function closeModal() {
    const modal = document.getElementById('tradeModal');
    if (modal) modal.style.display = 'none';
}

function formatCurrency(value) {
    const sign = value >= 0 ? '+' : '';
    return sign + '$' + Math.abs(value).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function formatCurrencyLarge(value) {
    return '$' + value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function applyPnlColor(elementId, value) {
    const el = document.getElementById(elementId);
    el.style.color = value >= 0 ? 'var(--accent-green)' : 'var(--accent-red)';
}

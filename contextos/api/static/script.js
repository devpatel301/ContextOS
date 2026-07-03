document.addEventListener("DOMContentLoaded", () => {
    fetchStatus();
    setInterval(fetchStatus, 3000); // Poll every 3 seconds

    const queryBtn = document.getElementById("run-query-btn");
    queryBtn.addEventListener("click", runQuery);

    const resetBtn = document.getElementById("reset-memory-btn");
    if (resetBtn) {
        resetBtn.addEventListener("click", resetMemory);
    }
});

async function fetchStatus() {
    try {
        const response = await fetch("/api/status");
        const data = await response.json();
        
        updateTierStats(data.memory.tiers);
        updateCacheStats(data.cache);
        
        // Only update budget if no query result is displaying it
        if (!document.getElementById("budget-fill").hasAttribute("data-locked")) {
            const usedTokens = data.memory.tiers.working.tokens + data.memory.tiers.episodic.tokens;
            updateBudgetBar(usedTokens, data.config.budget);
        }
    } catch (e) {
        console.error("Failed to fetch status:", e);
    }
}

function updateTierStats(tiers) {
    const container = document.getElementById("tier-stats");
    container.innerHTML = "";
    
    const colors = {
        working: "var(--skills-bg)",
        episodic: "var(--accent-orange)",
        semantic: "var(--black)",
        archived: "#ccc"
    };

    for (const [tier, stats] of Object.entries(tiers)) {
        const div = document.createElement("div");
        div.className = "stat-item";
        div.innerHTML = `
            <span style="text-transform: uppercase;">
                <span style="display:inline-block;width:12px;height:12px;background:${colors[tier]};border:1px solid black;margin-right:5px;"></span>
                ${tier}
            </span>
            <span>${stats.blocks} blk / ${stats.tokens} tk</span>
        `;
        container.appendChild(div);
    }
}

function updateCacheStats(cache) {
    const container = document.getElementById("cache-stats");
    container.innerHTML = `
        <div class="stat-item"><span>Total Entries</span> <span>${cache.entries}</span></div>
        <div class="stat-item"><span>Cache Hits</span> <span>${cache.hits}</span></div>
    `;
}

function updateBudgetBar(used, total) {
    const pct = Math.min(100, Math.round((used / total) * 100));
    document.getElementById("budget-fill").style.width = pct + "%";
    document.getElementById("budget-text").innerText = `${used} / ${total} Tokens (${pct}%)`;
}

async function runQuery() {
    const input = document.getElementById("query-input").value;
    if (!input.trim()) return;

    const btn = document.getElementById("run-query-btn");
    const statusMsg = document.getElementById("query-status");
    const blocksContainer = document.getElementById("context-window-blocks");
    
    btn.disabled = true;
    btn.innerText = "Processing...";
    statusMsg.innerHTML = "";
    blocksContainer.innerHTML = '<div class="empty-state">Running Context Scheduler...</div>';
    
    try {
        const start = performance.now();
        const response = await fetch("/api/query", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query: input, simulate_latency: true })
        });
        const data = await response.json();
        const latency = ((performance.now() - start) / 1000).toFixed(2);
        
        // Update Budget bar lock
        const fill = document.getElementById("budget-fill");
        fill.setAttribute("data-locked", "true");
        updateBudgetBar(data.tokens_used, data.budget);
        setTimeout(() => fill.removeAttribute("data-locked"), 5000);

        // Update Cache status message
        if (data.cache_hit) {
            statusMsg.innerHTML = `<span class="hit">⚡ Cache HIT</span> — Returned instantly in ${latency}s`;
            blocksContainer.innerHTML = '<div class="empty-state">Served entirely from Semantic Cache. No LLM packing needed.</div>';
        } else {
            statusMsg.innerHTML = `<span class="miss">Network MISS</span> — LLM generated in ${latency}s`;
            renderBlocks(data.blocks, blocksContainer);
        }
        
        fetchStatus(); // immediate refresh
    } catch (e) {
        statusMsg.innerHTML = `<span class="miss">Error executing query</span>`;
    } finally {
        btn.disabled = false;
        btn.innerText = "Allocate Context";
    }
}

function renderBlocks(blocks, container) {
    container.innerHTML = "";
    blocks.forEach(b => {
        const div = document.createElement("div");
        div.className = `block-card ${b.tier}`;
        div.innerHTML = `
            <div class="block-header">
                <span>[${b.tier.toUpperCase()}]</span>
                <span>Score: ${b.score}</span>
            </div>
            <div class="block-content">${b.content}</div>
        `;
        container.appendChild(div);
    });
}

async function resetMemory() {
    const btn = document.getElementById("reset-memory-btn");
    if (!btn) return;
    
    if (!confirm("Are you sure you want to clear all memory logs, cache entries, and vector data? This will reseed with fresh defaults.")) {
        return;
    }
    
    btn.disabled = true;
    btn.innerText = "Clearing...";
    
    try {
        const response = await fetch("/api/reset", {
            method: "POST"
        });
        const data = await response.json();
        
        // Reset query status and console
        document.getElementById("query-status").innerHTML = "";
        document.getElementById("context-window-blocks").innerHTML = '<div class="empty-state">Memory reset completed successfully! Run a query to begin.</div>';
        
        await fetchStatus();
    } catch (e) {
        console.error("Failed to reset memory:", e);
        alert("Failed to reset memory.");
    } finally {
        btn.disabled = false;
        btn.innerText = "Reset Memory";
    }
}

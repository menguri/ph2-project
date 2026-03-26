// ===== PH2 Human-AI Overcooked Study — Frontend =====

const TILE_SIZE = 80;  // 큰 타일

// 원작 스타일 색상
const COLORS = {
    floor: "#E8D5B7",
    counter: "#8B7355",
    counterTop: "#A0896C",
    wall: "#5C4033",
    potBase: "#4A4A4A",
    potInner: "#2A2A2A",
    serving: "#4CAF50",
    servingBorder: "#388E3C",
    platePile: "#D2B48C",
    onionPile: "#F5DEB3",
    onion: "#DAA520",
    onionSkin: "#B8860B",
    tomato: "#DC143C",
    plate: "#FFFACD",
    plateBorder: "#BDB76B",
    soup: "#FF6347",
    soupBowl: "#FFFACD",
    playerYou: "#2196F3",
    playerAI: "#FF5722",
    hat: "#FFFFFF",
};

let participantId = crypto.randomUUID();
let ws = null;
let gameState = null;
let terrain = null;
let humanPlayerIndex = 0;
let episodeId = null;
let episodeLength = 400;
let currentAlgo = "";
let selectedLayout = "cramped_room";

// ===== Page Navigation =====
function showPage(pageId) {
    document.querySelectorAll(".page").forEach(p => p.classList.remove("active"));
    document.getElementById(pageId).classList.add("active");
}

// ===== Likert slider value display =====
document.querySelectorAll("input[type='range']").forEach(el => {
    el.addEventListener("input", e => {
        const span = e.target.parentElement.querySelector(".likert-val, #gaming-val");
        if (span) span.textContent = e.target.value;
    });
});

// ===== Pre-Survey =====
document.getElementById("pre-survey-form").addEventListener("submit", async e => {
    e.preventDefault();
    const data = {
        participant_id: participantId,
        age: parseInt(document.getElementById("survey-age").value) || null,
        gender: document.getElementById("survey-gender").value || null,
        gaming_exp: parseInt(document.getElementById("survey-gaming").value) || null,
        overcooked_exp: document.getElementById("survey-overcooked").value || null,
    };
    await fetch("/survey/pre", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(data),
    });
    showLayoutSelection();
});

// ===== Layout Selection =====
async function showLayoutSelection() {
    showPage("page-layout");
    const container = document.getElementById("layout-buttons");
    container.innerHTML = "";
    document.getElementById("layout-loading").style.display = "block";

    try {
        const resp = await fetch("/api/layouts");
        const data = await resp.json();
        document.getElementById("layout-loading").style.display = "none";

        const layoutNames = {
            "cramped_room": "Cramped Room",
            "asymmetric_advantages": "Asymmetric Advantages",
            "counter_circuit": "Counter Circuit",
            "coordination_ring": "Coordination Ring",
            "forced_coordination": "Forced Coordination",
        };

        for (const [layout, info] of Object.entries(data.layouts)) {
            const btn = document.createElement("button");
            btn.className = "layout-btn";
            btn.innerHTML = `<strong>${layoutNames[layout] || layout}</strong><br>
                <small>${info.algos.join(", ")} — ${info.count} model(s)</small>`;
            btn.onclick = () => {
                selectedLayout = layout;
                startNewGame();
            };
            container.appendChild(btn);
        }

        if (Object.keys(data.layouts).length === 0) {
            container.innerHTML = "<p>No models available. Please add models to the models/ directory.</p>";
        }
    } catch (e) {
        document.getElementById("layout-loading").style.display = "none";
        container.innerHTML = "<p>Failed to load layouts.</p>";
    }
}

// ===== Game =====
function startNewGame() {
    showPage("page-game");
    connectWebSocket();
}

function connectWebSocket() {
    const protocol = location.protocol === "https:" ? "wss:" : "ws:";
    ws = new WebSocket(`${protocol}//${location.host}/ws/${participantId}`);

    ws.onopen = () => {
        ws.send(JSON.stringify({type: "start_game", layout: selectedLayout}));
        document.getElementById("game-status").textContent = "Connecting...";
    };

    ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        if (msg.type === "game_start") {
            terrain = msg.terrain;
            humanPlayerIndex = msg.human_player_index;
            episodeId = msg.episode_id;
            episodeLength = msg.episode_length;
            currentAlgo = msg.algo_name || "";
            gameState = msg.state;
            resizeCanvas();
            render();
            // human은 항상 파란색(playerYou=#2196F3)으로 렌더링됨
            const colorName = currentLang === "ko" ? "파란색" : "BLUE";
            document.getElementById("game-status").textContent = t("game_status_you").replace("{color}", colorName);
            document.getElementById("hud-score").textContent = `${t("hud_score")}: 0`;
            document.getElementById("hud-step").textContent = `${t("hud_step")}: 0 / ${episodeLength}`;
        } else if (msg.type === "state_update") {
            gameState = msg.state;
            render();
            document.getElementById("hud-score").textContent = `${t("hud_score")}: ${msg.score}`;
            document.getElementById("hud-step").textContent = `${t("hud_step")}: ${msg.timestep} / ${episodeLength}`;
        } else if (msg.type === "episode_end") {
            document.getElementById("post-score-display").textContent =
                currentLang === "ko" ? `최종 점수: ${msg.final_score}` : `Final Score: ${msg.final_score}`;
            episodeId = msg.episode_id;
            showPage("page-post-survey");
        } else if (msg.error) {
            document.getElementById("game-status").textContent = `Error: ${msg.error}`;
        }
    };

    ws.onclose = () => {
        document.getElementById("game-status").textContent =
            currentLang === "ko" ? "연결이 끊어졌습니다." : "Disconnected.";
    };
}

// Keyboard input
document.addEventListener("keydown", (e) => {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    const page = document.querySelector(".page.active");
    if (!page || page.id !== "page-game") return;

    const keyMap = {
        "ArrowRight": 0, "ArrowDown": 1, "ArrowLeft": 2, "ArrowUp": 3,
        "d": 0, "s": 1, "a": 2, "w": 3,
        " ": 5, "e": 5, "Enter": 5,
    };
    const action = keyMap[e.key];
    if (action !== undefined) {
        e.preventDefault();
        ws.send(JSON.stringify({action}));
    }
});

// ===== Canvas Rendering (원작 스타일) =====
function resizeCanvas() {
    if (!terrain) return;
    const canvas = document.getElementById("game-canvas");
    const h = terrain.length;
    const w = terrain[0].length;
    canvas.width = w * TILE_SIZE;
    canvas.height = h * TILE_SIZE;
}

function render() {
    if (!terrain || !gameState) return;
    const canvas = document.getElementById("game-canvas");
    const ctx = canvas.getContext("2d");
    const h = terrain.length;
    const w = terrain[0].length;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // 1) Draw terrain tiles
    for (let r = 0; r < h; r++) {
        for (let c = 0; c < w; c++) {
            const ch = terrain[r][c];
            const x = c * TILE_SIZE;
            const y = r * TILE_SIZE;
            drawTile(ctx, x, y, ch);
        }
    }

    // 2) Draw objects on the grid (soup in pots, items on counters)
    if (gameState.objects) {
        for (const [key, obj] of Object.entries(gameState.objects)) {
            const [cx, cy] = key.split(",").map(Number);
            const x = cx * TILE_SIZE;
            const y = cy * TILE_SIZE;
            drawObject(ctx, x, y, obj);
        }
    }

    // 3) Draw players
    if (gameState.players) {
        gameState.players.forEach((player, idx) => {
            const isHuman = idx === humanPlayerIndex;
            drawPlayer(ctx, player, isHuman);
        });
    }
}

function drawTile(ctx, x, y, ch) {
    const T = TILE_SIZE;
    switch (ch) {
        case "X": // Wall / Counter
            // Counter body
            ctx.fillStyle = COLORS.counter;
            ctx.fillRect(x, y, T, T);
            // Counter top surface
            ctx.fillStyle = COLORS.counterTop;
            ctx.fillRect(x + 2, y + 2, T - 4, T - 4);
            break;
        case " ": // Empty floor
            ctx.fillStyle = COLORS.floor;
            ctx.fillRect(x, y, T, T);
            // Floor pattern
            ctx.strokeStyle = "#D4C4A8";
            ctx.lineWidth = 0.5;
            ctx.strokeRect(x + 1, y + 1, T - 2, T - 2);
            break;
        case "P": // Pot
            ctx.fillStyle = COLORS.counter;
            ctx.fillRect(x, y, T, T);
            ctx.fillStyle = COLORS.counterTop;
            ctx.fillRect(x + 2, y + 2, T - 4, T - 4);
            drawPot(ctx, x, y);
            break;
        case "S": // Serving window (수프 제출하는 곳)
            ctx.fillStyle = COLORS.counter;
            ctx.fillRect(x, y, T, T);
            ctx.fillStyle = "#2E7D32";
            ctx.fillRect(x + 6, y + 6, T - 12, T - 12);
            ctx.strokeStyle = "#1B5E20";
            ctx.lineWidth = 3;
            ctx.strokeRect(x + 6, y + 6, T - 12, T - 12);
            ctx.fillStyle = "#FFF";
            ctx.font = `bold ${T * 0.16}px sans-serif`;
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText("SERVE", x + T / 2, y + T * 0.4);
            ctx.beginPath();
            ctx.moveTo(x + T / 2 - 10, y + T * 0.55);
            ctx.lineTo(x + T / 2 + 10, y + T * 0.55);
            ctx.lineTo(x + T / 2, y + T * 0.72);
            ctx.closePath();
            ctx.fill();
            break;
        case "D": // Dish dispenser (접시 꺼내는 곳)
            ctx.fillStyle = COLORS.counter;
            ctx.fillRect(x, y, T, T);
            ctx.fillStyle = "#C9B896";
            ctx.fillRect(x + 4, y + 4, T - 8, T - 8);
            // Stack of plates
            for (let i = 0; i < 3; i++) {
                drawPlate(ctx, x + T / 2, y + T * 0.3 + i * 10, T * 0.28);
            }
            ctx.fillStyle = "#555";
            ctx.font = `bold ${T * 0.13}px sans-serif`;
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText("DISH", x + T / 2, y + T * 0.85);
            break;
        case "O": // Onion pile
            ctx.fillStyle = COLORS.counter;
            ctx.fillRect(x, y, T, T);
            ctx.fillStyle = COLORS.onionPile;
            ctx.fillRect(x + 2, y + 2, T - 4, T - 4);
            // Draw 3 onions
            drawOnion(ctx, x + T * 0.25, y + T * 0.35, T * 0.18);
            drawOnion(ctx, x + T * 0.55, y + T * 0.3, T * 0.18);
            drawOnion(ctx, x + T * 0.4, y + T * 0.6, T * 0.18);
            break;
        case "T": // Tomato pile
            ctx.fillStyle = COLORS.counter;
            ctx.fillRect(x, y, T, T);
            ctx.fillStyle = "#FFE4E1";
            ctx.fillRect(x + 2, y + 2, T - 4, T - 4);
            drawTomato(ctx, x + T * 0.3, y + T * 0.35, T * 0.18);
            drawTomato(ctx, x + T * 0.6, y + T * 0.35, T * 0.18);
            drawTomato(ctx, x + T * 0.45, y + T * 0.6, T * 0.18);
            break;
        case "B": // Plate/Dish pile
            ctx.fillStyle = COLORS.counter;
            ctx.fillRect(x, y, T, T);
            ctx.fillStyle = "#C9B896";
            ctx.fillRect(x + 2, y + 2, T - 4, T - 4);
            // Stack of plates (white, clear)
            for (let i = 0; i < 3; i++) {
                const py = y + T * 0.3 + i * 10;
                drawPlate(ctx, x + T / 2, py, T * 0.3);
            }
            // Label
            ctx.fillStyle = "#555";
            ctx.font = `bold ${T * 0.13}px sans-serif`;
            ctx.textAlign = "center";
            ctx.fillText("DISHES", x + T / 2, y + T * 0.85);
            break;
        default:
            ctx.fillStyle = COLORS.floor;
            ctx.fillRect(x, y, T, T);
    }
}

function drawPot(ctx, x, y) {
    const T = TILE_SIZE;
    const cx = x + T / 2;
    const cy = y + T / 2;
    const pw = T * 0.55;
    const ph = T * 0.35;

    // Pot body
    ctx.fillStyle = COLORS.potBase;
    ctx.beginPath();
    ctx.roundRect(cx - pw / 2, cy - ph / 2 + 5, pw, ph, 4);
    ctx.fill();

    // Pot rim
    ctx.fillStyle = "#666";
    ctx.fillRect(cx - pw / 2 - 3, cy - ph / 2 + 3, pw + 6, 6);

    // Handles
    ctx.fillStyle = "#555";
    ctx.fillRect(cx - pw / 2 - 8, cy, 8, 4);
    ctx.fillRect(cx + pw / 2, cy, 8, 4);
}

function drawOnion(ctx, cx, cy, r) {
    // Onion body
    ctx.fillStyle = COLORS.onion;
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.fill();
    // Onion skin lines
    ctx.strokeStyle = COLORS.onionSkin;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(cx, cy, r * 0.6, -0.5, 0.5);
    ctx.stroke();
    // Top sprout
    ctx.strokeStyle = "#228B22";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(cx, cy - r);
    ctx.lineTo(cx, cy - r - 4);
    ctx.stroke();
}

function drawTomato(ctx, cx, cy, r) {
    ctx.fillStyle = COLORS.tomato;
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.fill();
    // Stem
    ctx.fillStyle = "#228B22";
    ctx.beginPath();
    ctx.arc(cx, cy - r + 2, 3, 0, Math.PI * 2);
    ctx.fill();
}

function drawPlate(ctx, cx, cy, r) {
    ctx.fillStyle = COLORS.plate;
    ctx.beginPath();
    ctx.ellipse(cx, cy, r, r * 0.4, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = COLORS.plateBorder;
    ctx.lineWidth = 1.5;
    ctx.stroke();
}

function drawObject(ctx, x, y, obj) {
    const T = TILE_SIZE;
    const cx = x + T / 2;
    const cy = y + T / 2;

    if (obj.name === "soup") {
        // Soup in pot
        const isCooking = obj.is_cooking || false;
        const isReady = obj.is_ready || false;
        const ingredients = obj.ingredients || [];
        const numIng = ingredients.length;

        // Draw ingredients inside pot (small circles)
        if (numIng > 0) {
            // Soup liquid fill
            const soupColor = isReady ? "#E65100" : (isCooking ? "#FF8C00" : "#FFAB40");
            ctx.fillStyle = soupColor;
            ctx.beginPath();
            ctx.roundRect(cx - T * 0.23, cy - T * 0.02, T * 0.46, T * 0.18, 3);
            ctx.fill();

            // Individual ingredient icons in pot
            const positions = [
                [cx - 8, cy + 3],
                [cx + 8, cy + 3],
                [cx, cy - 2],
            ];
            for (let i = 0; i < Math.min(numIng, 3); i++) {
                const [ix, iy] = positions[i];
                const ing = ingredients[i];
                if (ing === "onion") drawOnion(ctx, ix, iy, 5);
                else if (ing === "tomato") drawTomato(ctx, ix, iy, 5);
            }
        }

        // Ingredient count badge
        ctx.fillStyle = "#FFF";
        ctx.strokeStyle = "#333";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(cx + T * 0.3, cy - T * 0.25, 10, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
        ctx.fillStyle = "#333";
        ctx.font = "bold 12px sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(`${numIng}/3`, cx + T * 0.3, cy - T * 0.25);

        // Cooking progress bar
        if (isCooking && !isReady) {
            const barW = T * 0.5;
            const barH = 6;
            const barX = cx - barW / 2;
            const barY = cy + T * 0.28;
            // Background
            ctx.fillStyle = "#555";
            ctx.fillRect(barX, barY, barW, barH);
            // Progress
            if (obj.cooking_tick !== undefined && obj.cook_time) {
                const pct = Math.min(obj.cooking_tick / obj.cook_time, 1);
                ctx.fillStyle = "#FF9800";
                ctx.fillRect(barX, barY, barW * pct, barH);
            }
            // Label
            ctx.fillStyle = "#FFF";
            ctx.font = `bold ${T * 0.12}px sans-serif`;
            ctx.textAlign = "center";
            ctx.fillText("COOKING...", cx, barY + barH + 10);
        }

        // Ready indicator
        if (isReady) {
            ctx.fillStyle = "#4CAF50";
            ctx.font = `bold ${T * 0.15}px sans-serif`;
            ctx.textAlign = "center";
            ctx.fillText("READY!", cx, cy + T * 0.35);
            // Glow
            ctx.strokeStyle = "#4CAF50";
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.arc(cx, cy, T * 0.35, 0, Math.PI * 2);
            ctx.stroke();
        }

        // Waiting (ingredients added but not cooking yet)
        if (numIng > 0 && !isCooking && !isReady) {
            ctx.fillStyle = "#999";
            ctx.font = `${T * 0.11}px sans-serif`;
            ctx.textAlign = "center";
            ctx.fillText(`${numIng}/3 ingredients`, cx, cy + T * 0.35);
        }
    } else if (obj.name === "onion") {
        drawOnion(ctx, cx, cy + 5, T * 0.15);
    } else if (obj.name === "tomato") {
        drawTomato(ctx, cx, cy + 5, T * 0.15);
    } else if (obj.name === "dish") {
        drawPlate(ctx, cx, cy + 5, T * 0.2);
    }
}

function drawPlayer(ctx, player, isHuman) {
    const T = TILE_SIZE;
    const [px, py] = player.position;
    const x = px * TILE_SIZE;
    const y = py * TILE_SIZE;
    const cx = x + T / 2;
    const cy = y + T / 2;
    const color = isHuman ? COLORS.playerYou : COLORS.playerAI;
    const [dx, dy] = player.orientation;

    // Body
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(cx, cy + 4, T * 0.3, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "#333";
    ctx.lineWidth = 2;
    ctx.stroke();

    // Chef hat
    ctx.fillStyle = COLORS.hat;
    ctx.beginPath();
    ctx.ellipse(cx, cy - T * 0.15, T * 0.2, T * 0.12, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillRect(cx - T * 0.12, cy - T * 0.3, T * 0.24, T * 0.18);
    ctx.strokeStyle = "#DDD";
    ctx.lineWidth = 1;
    ctx.stroke();

    // Eyes (direction indicator)
    const eyeX = cx + dx * 5;
    const eyeY = cy + dy * 5;
    ctx.fillStyle = "#333";
    ctx.beginPath();
    ctx.arc(eyeX - 4, eyeY, 2.5, 0, Math.PI * 2);
    ctx.arc(eyeX + 4, eyeY, 2.5, 0, Math.PI * 2);
    ctx.fill();

    // Label above
    ctx.fillStyle = color;
    ctx.font = "bold 13px sans-serif";
    ctx.textAlign = "center";
    ctx.strokeStyle = "#000";
    ctx.lineWidth = 3;
    ctx.strokeText(isHuman ? "You" : "AI", cx, y - 4);
    ctx.fillText(isHuman ? "You" : "AI", cx, y - 4);

    // Held object
    if (player.held_object) {
        const held = player.held_object;
        const hx = cx + dx * T * 0.35;
        const hy = cy + dy * T * 0.35 + 4;

        if (held.name === "onion") {
            drawOnion(ctx, hx, hy, T * 0.12);
        } else if (held.name === "tomato") {
            drawTomato(ctx, hx, hy, T * 0.12);
        } else if (held.name === "dish") {
            drawPlate(ctx, hx, hy, T * 0.15);
        } else if (held.name === "soup") {
            // Bowl of soup
            drawPlate(ctx, hx, hy, T * 0.15);
            ctx.fillStyle = COLORS.soup;
            ctx.beginPath();
            ctx.ellipse(hx, hy - 2, T * 0.12, T * 0.06, 0, 0, Math.PI * 2);
            ctx.fill();
        }
    }
}

// ===== Post-Survey =====
document.getElementById("post-survey-form").addEventListener("submit", async e => {
    e.preventDefault();
    const data = {
        participant_id: participantId,
        episode_id: episodeId,
        fluency: parseInt(document.getElementById("post-fluency").value),
        contribution: parseInt(document.getElementById("post-contribution").value),
        trust: parseInt(document.getElementById("post-trust").value),
        human_likeness: parseInt(document.getElementById("post-human-likeness").value),
        obstruction: parseInt(document.getElementById("post-obstruction").value),
        frustration: parseInt(document.getElementById("post-frustration").value),
        play_again: parseInt(document.getElementById("post-play-again").value),
        open_text: document.getElementById("post-open-text").value || "",
    };
    await fetch("/survey/post", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(data),
    });
    ["post-fluency","post-contribution","post-trust","post-human-likeness","post-obstruction","post-frustration","post-play-again"].forEach(id => {
        document.getElementById(id).value = 4;
    });
    document.querySelectorAll(".likert-val").forEach(el => el.textContent = "4");
    document.getElementById("post-open-text").value = "";
    applyI18n();
    showPage("page-again");
});

console.log("PH2 Human-AI Study webapp loaded. Participant:", participantId);

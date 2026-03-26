// ===== i18n: Korean / English =====
const I18N = {
    en: {
        // Consent
        consent_title: "Welcome",
        consent_desc: 'You will play the cooperative cooking game <strong>Overcooked</strong> with an AI partner.',
        consent_task: "Your task is to work together to prepare and deliver soup orders as efficiently as possible.",
        consent_controls_title: "Controls",
        consent_controls_move: "<strong>Arrow keys</strong> or <strong>WASD</strong> — Move",
        consent_controls_interact: "<strong>Space</strong> or <strong>E</strong> — Interact (pick up / place / deliver)",
        consent_survey_note: "After each game, you'll be asked to fill out a short survey about your experience.",
        consent_agree: "I Agree & Continue",

        // Pre-survey
        pre_title: "About You",
        pre_age: "Age",
        pre_gender: "Gender",
        pre_select: "Select",
        pre_male: "Male",
        pre_female: "Female",
        pre_other: "Other",
        pre_prefer_not: "Prefer not to say",
        pre_gaming: "Gaming Experience (1=None, 7=Expert)",
        pre_overcooked: "Overcooked Experience",
        pre_oc_none: "None",
        pre_oc_some: "Some",
        pre_oc_alot: "A lot",
        pre_start: "Start Game",

        // Tutorial
        tut1_title: "Goal",
        tut1_desc: "Cook and deliver onion soup with your AI partner to earn points!",
        tut1_s1: "Pick up onions",
        tut1_s2: "Put 3 onions in the pot",
        tut1_s3: "Wait for soup to cook",
        tut1_s4: "Pick up a plate, then the soup",
        tut1_s5: "Deliver to the green serving area",
        tut2_title: "Items",
        tut2_onion: "Onion — pick up from the onion pile",
        tut2_plate: "Plate — pick up from the plate dispenser",
        tut2_soup: "Soup — cooked soup on a plate, ready to deliver",
        tut3_title: "Kitchen Objects",
        tut3_pot: "Pot — put 3 onions in, cooking starts automatically",
        tut3_onion_pile: "Onion Pile — unlimited onion supply",
        tut3_plate_pile: "Plate Dispenser — unlimited plate supply",
        tut3_serve: "Serving Area (green) — deliver soup here for points!",
        tut4_title: "Controls",
        tut4_move: "Arrow keys / WASD — Move & face direction",
        tut4_act: "Space / E — Interact: pick up, put down, or serve",
        tut4_face: "Face the object and press ACT to interact with it.",
        tut5_title: "Tips",
        tut5_you: "You are the BLUE chef",
        tut5_ai: "The ORANGE chef is your AI partner",
        tut5_coop: "Work together — divide tasks for better scores!",
        tut_next: "Next",
        tut_start: "Choose Layout",

        // Layout
        layout_title: "Select Layout",
        layout_desc: "Choose a kitchen layout to play on:",
        layout_loading: "Loading layouts...",

        // Game HUD
        hud_score: "Score",
        hud_step: "Step",
        game_status_you: "You are the {color} chef. Arrow keys to move, Space to interact.",

        // Post-survey
        post_title: "Rate Your AI Partner",
        post_fluency: "The agent and I worked together fluently.",
        post_contribution: "The agent contributed to our team performance.",
        post_trust: "I trusted the agent's actions.",
        post_human_likeness: "The agent felt like a human player.",
        post_obstruction: "The agent got in my way.",
        post_frustration: "I felt frustrated during the game.",
        post_play_again: "I would want to play with this agent again.",
        post_open: "Describe your experience with the agent.",
        post_submit: "Submit & Continue",

        // Again / Thanks
        again_title: "Thank You!",
        again_desc: "Your response has been recorded.",
        again_play: "Play Again (New AI Partner)",
        again_done: "Done",
        thanks_title: "Thank you for participating!",
        thanks_desc: "You may close this window.",
    },
    ko: {
        // Consent
        consent_title: "환영합니다",
        consent_desc: 'AI 파트너와 함께 협동 요리 게임 <strong>Overcooked</strong>를 플레이합니다.',
        consent_task: "가능한 효율적으로 함께 수프 주문을 준비하고 배달하는 것이 목표입니다.",
        consent_controls_title: "조작법",
        consent_controls_move: "<strong>방향키</strong> 또는 <strong>WASD</strong> — 이동",
        consent_controls_interact: "<strong>스페이스바</strong> 또는 <strong>E</strong> — 상호작용 (줍기 / 놓기 / 배달)",
        consent_survey_note: "게임이 끝난 후 짧은 설문조사에 응답해 주세요.",
        consent_agree: "동의하고 계속하기",

        // Pre-survey
        pre_title: "기본 정보",
        pre_age: "나이",
        pre_gender: "성별",
        pre_select: "선택",
        pre_male: "남성",
        pre_female: "여성",
        pre_other: "기타",
        pre_prefer_not: "응답하지 않음",
        pre_gaming: "게임 경험 (1=없음, 7=매우 많음)",
        pre_overcooked: "Overcooked 경험",
        pre_oc_none: "없음",
        pre_oc_some: "조금 있음",
        pre_oc_alot: "많이 있음",
        pre_start: "게임 시작",

        // Tutorial
        tut1_title: "게임 목표",
        tut1_desc: "AI 파트너와 함께 양파 수프를 만들어 배달하면 점수를 얻습니다!",
        tut1_s1: "양파를 집는다",
        tut1_s2: "냄비에 양파 3개를 넣는다",
        tut1_s3: "수프가 익을 때까지 기다린다",
        tut1_s4: "접시를 집고, 완성된 수프를 담는다",
        tut1_s5: "초록색 제출대에 배달한다",
        tut2_title: "아이템",
        tut2_onion: "양파 — 양파 더미에서 집기",
        tut2_plate: "접시 — 접시 보관대에서 집기",
        tut2_soup: "수프 — 접시에 담긴 완성 수프, 배달 가능",
        tut3_title: "주방 시설",
        tut3_pot: "냄비 — 양파 3개 넣으면 자동으로 요리 시작",
        tut3_onion_pile: "양파 더미 — 무제한 양파 공급",
        tut3_plate_pile: "접시 보관대 — 무제한 접시 공급",
        tut3_serve: "제출대 (초록색) — 여기에 수프를 배달하면 점수 획득!",
        tut4_title: "조작법",
        tut4_move: "방향키 / WASD — 이동 및 방향 전환",
        tut4_act: "스페이스 / E — 상호작용: 집기, 놓기, 배달",
        tut4_face: "물체를 바라본 상태에서 ACT를 누르면 상호작용합니다.",
        tut5_title: "팁",
        tut5_you: "당신은 파란색 셰프입니다",
        tut5_ai: "주황색 셰프가 AI 파트너입니다",
        tut5_coop: "함께 협력하세요 — 역할을 나누면 더 높은 점수를 얻을 수 있습니다!",
        tut_next: "다음",
        tut_start: "레이아웃 선택",

        // Layout
        layout_title: "레이아웃 선택",
        layout_desc: "플레이할 주방 레이아웃을 선택하세요:",
        layout_loading: "레이아웃 로딩 중...",

        // Game HUD
        hud_score: "점수",
        hud_step: "스텝",
        game_status_you: "당신은 {color} 셰프입니다. 방향키로 이동, 스페이스바로 상호작용.",

        // Post-survey
        post_title: "AI 파트너 평가",
        post_fluency: "AI와 원활하게 협력했다.",
        post_contribution: "AI가 팀 성과에 기여했다.",
        post_trust: "AI의 행동을 신뢰했다.",
        post_human_likeness: "AI가 사람처럼 느껴졌다.",
        post_obstruction: "AI가 내 길을 방해했다.",
        post_frustration: "게임 중 답답함을 느꼈다.",
        post_play_again: "이 AI와 다시 플레이하고 싶다.",
        post_open: "AI와의 경험을 자유롭게 적어주세요.",
        post_submit: "제출하고 계속하기",

        // Again / Thanks
        again_title: "감사합니다!",
        again_desc: "응답이 기록되었습니다.",
        again_play: "다시 플레이 (새 AI 파트너)",
        again_done: "종료",
        thanks_title: "참여해 주셔서 감사합니다!",
        thanks_desc: "이 창을 닫으셔도 됩니다.",
    },
};

let currentLang = "en";

function setLang(lang) {
    currentLang = lang;
    applyI18n();
    showPage("page-consent");
}

function t(key) {
    return (I18N[currentLang] || I18N.en)[key] || (I18N.en[key] || key);
}

function applyI18n() {
    document.querySelectorAll("[data-i18n]").forEach(el => {
        const key = el.getAttribute("data-i18n");
        const text = t(key);
        if (el.tagName === "OPTION" || el.tagName === "BUTTON" || el.tagName === "SPAN" || el.tagName === "P" || el.tagName === "H2" || el.tagName === "H3" || el.tagName === "LI") {
            el.innerHTML = text;
        }
    });
}

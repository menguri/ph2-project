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

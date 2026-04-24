// ===== i18n: Korean / English =====
const I18N = {
    en: {
        // Consent
        consent_page_title: "Research Study Information",
        consent_study_title: "State-Blocked Multi-Agent Learning for Zero-Shot Coordination",
        consent_intro: "This is a <strong>computer-based cooperation game study</strong> to measure how well AI agents cooperate with humans.",
        consent_summary_title: "Summary",
        consent_bullet_task: "Play the cooperative cooking game <strong>Overcooked</strong> with an AI partner.",
        consent_bullet_time: "Approximately <strong>30–40 minutes</strong> (multiple short episodes + brief post-episode surveys).",
        consent_bullet_gift: "Upon completion, a gift card worth approximately <strong>10,000 KRW</strong> will be provided (not provided if terminated early).",
        consent_bullet_privacy: "<strong>No personally identifiable information</strong> (name, contact, email, etc.) will be collected.",
        consent_bullet_voluntary: "Participation is voluntary — you may close the browser to withdraw at any time.",
        consent_bullet_contact: "Contact: <strong>mingukang@unist.ac.kr</strong>",
        consent_full_toggle: "View full disclosure",
        consent_check_18: "I am <strong>18 years of age or older</strong>.",
        consent_check_voluntary: "I have read and understood the information and <strong>voluntarily agree to participate</strong>.",
        consent_agree: "I Agree",

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
        tut_start: "Start Game!",

        // Layout
        layout_title: "Select Layout",
        layout_desc: "Choose a kitchen layout to play on:",
        layout_loading: "Loading layouts...",

        // Game HUD
        hud_score: "Score",
        hud_timer: "Time",
        hud_step: "Step",
        game_status_you: "You are the {color} chef. Arrow keys to move, Space to interact.",
        countdown_start: "Start!",

        // Post-survey
        post_title: "Rate Your AI Partner",
        post_legend: "1 = Strongly Disagree, 7 = Strongly Agree",
        post_adaptive: "The AI partner adapted to my actions and strategy.",
        post_consistent: "The AI partner's behavior was consistent and predictable.",
        post_human_like: "The AI partner behaved in a human-like way.",
        post_in_my_way: "The AI partner often got in my way.",
        post_frustrating: "Playing with this AI partner was frustrating.",
        post_enjoyed: "I enjoyed collaborating with this AI partner.",
        post_coordination: "The AI partner and I coordinated well as a team.",
        post_workload: "How mentally demanding was it to work with this partner? (1 = Very Low, 7 = Very High)",
        post_open: "Describe your experience with the agent.",
        post_submit: "Submit & Continue",

        // Again / Quit / Thanks
        again_title: "Thank You!",
        again_desc: "Your response has been recorded.",
        again_play: "Continue to Next Game",
        again_done: "End Study",
        quit_title: "Are you sure?",
        quit_desc: "You have not completed all games. A gift card will only be provided upon full completion.",
        quit_contact: 'If you experienced issues, please contact <strong>mingukang@unist.ac.kr</strong>.',
        quit_continue: "Continue Playing",
        quit_confirm: "Confirm End",
        thanks_title: "Thank you for participating!",
        thanks_desc: "You may close this window.",
        thanks_complete: "You have completed all games! A gift card will be sent to you. You may close this window.",
    },
    ko: {
        // Consent
        consent_page_title: "연구 참여 안내",
        consent_study_title: "제로샷 협력을 위한 상태 차단 기반 다중에이전트 학습",
        consent_intro: "본 연구는 AI 에이전트가 사람과 얼마나 잘 협력하는지 측정하기 위한 <strong>컴퓨터 기반 협력 게임 연구</strong>입니다.",
        consent_summary_title: "요약",
        consent_bullet_task: "AI 파트너와 <strong>Overcooked</strong> 협력 요리 게임을 플레이합니다.",
        consent_bullet_time: "약 <strong>30–40분</strong> 소요 (짧은 에피소드 여러 회 + 에피소드 후 짧은 설문).",
        consent_bullet_gift: "완료 시 약 <strong>10,000원 기프티콘</strong> 지급 (중도 종료 시 미지급).",
        consent_bullet_privacy: "개인식별정보(성명·연락처·이메일 등)를 <strong>일절 수집하지 않음</strong>.",
        consent_bullet_voluntary: "참여는 자발적이며 언제든지 브라우저를 닫아 중단할 수 있습니다.",
        consent_bullet_contact: "문의: <strong>mingukang@unist.ac.kr</strong>",
        consent_full_toggle: "전체 안내문 보기",
        consent_check_18: "본인은 <strong>만 18세 이상</strong>입니다.",
        consent_check_voluntary: "본인은 안내문의 내용을 이해하였으며 <strong>자발적으로 본 연구에 참여</strong>합니다.",
        consent_agree: "동의합니다",

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
        tut_start: "게임 시작!",

        // Layout
        layout_title: "레이아웃 선택",
        layout_desc: "플레이할 주방 레이아웃을 선택하세요:",
        layout_loading: "레이아웃 로딩 중...",

        // Game HUD
        hud_score: "점수",
        hud_timer: "시간",
        hud_step: "스텝",
        game_status_you: "당신은 {color} 셰프입니다. 방향키로 이동, 스페이스바로 상호작용.",
        countdown_start: "시작!",

        // Post-survey
        post_title: "AI 파트너 평가",
        post_legend: "1 = 전혀 그렇지 않다, 7 = 매우 그렇다",
        post_adaptive: "AI 파트너가 내 행동과 전략에 맞춰 적응했다.",
        post_consistent: "AI 파트너의 행동은 일관되고 예측 가능했다.",
        post_human_like: "AI 파트너의 행동이 사람처럼 자연스러웠다.",
        post_in_my_way: "AI 파트너가 자주 내 길을 방해했다.",
        post_frustrating: "이 AI 파트너와 플레이하는 것이 답답했다.",
        post_enjoyed: "이 AI 파트너와 협력하는 것이 즐거웠다.",
        post_coordination: "나와 AI 파트너는 팀으로서 잘 협력했다.",
        post_workload: "이 파트너와 작업하는 것이 정신적으로 얼마나 부담됐는가? (1 = 매우 낮음, 7 = 매우 높음)",
        post_open: "AI와의 경험을 자유롭게 적어주세요.",
        post_submit: "제출하고 계속하기",

        // Again / Quit / Thanks
        again_title: "감사합니다!",
        again_desc: "응답이 기록되었습니다.",
        again_play: "다음 게임 계속하기",
        again_done: "연구 종료",
        quit_title: "정말 종료하시겠습니까?",
        quit_desc: "아직 모든 게임을 완료하지 않았습니다. 기프티콘은 모든 게임을 완료해야 지급됩니다.",
        quit_contact: '문제가 있으시면 <strong>mingukang@unist.ac.kr</strong>로 문의해 주세요.',
        quit_continue: "계속 플레이",
        quit_confirm: "종료 확인",
        thanks_title: "참여해 주셔서 감사합니다!",
        thanks_desc: "이 창을 닫으셔도 됩니다.",
        thanks_complete: "모든 게임을 완료하셨습니다! 기프티콘이 지급될 예정입니다. 이 창을 닫으셔도 됩니다.",
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
        if (el.tagName === "OPTION" || el.tagName === "BUTTON" || el.tagName === "SPAN" || el.tagName === "P" || el.tagName === "H1" || el.tagName === "H2" || el.tagName === "H3" || el.tagName === "H4" || el.tagName === "LI" || el.tagName === "SUMMARY") {
            el.innerHTML = text;
        }
    });
}

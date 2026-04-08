import pandas as pd
import os
from typing import Dict, Optional
import re

def _is_not_self_play(label: str) -> bool:
    """
    'policy_labels' (예: 'cross-0_1', 'cross-0_1_1_1')를 파싱하여
    모든 인덱스가 동일한 self-play 가 아닌지 확인합니다.
    n-agent 환경 (GridSpread 의 4-agent 등) 도 정확히 처리.
    """
    match = re.search(r'cross-([\d_]+)', str(label))
    if not match:
        return False
    tokens = match.group(1).split('_')
    return len(set(tokens)) > 1

def calculate_metrics_for_run(run_path: str) -> Optional[Dict[str, float]]:
    """
    단일 실행 폴더에 대한 SP 및 XP 성능 지표를 계산합니다.

    Args:
        run_path (str): 분석할 개별 실행 폴더의 경로.

    Returns:
        Optional[Dict[str, float]]: 계산된 지표(sp-mean, sp-std, xp-mean, xp-std, gap)가 포함된 딕셔너리.
                                     필요한 파일이 없으면 None을 반환합니다.
    """
    run_name = os.path.basename(run_path)
    cross_csv_path = os.path.join(run_path, 'reward_summary_cross.csv')

    # 필수 CSV 파일 존재 여부 확인
    if not os.path.exists(cross_csv_path):
        print(f"[{run_name}] 필수 'reward_summary_cross.csv' 파일이 없어 건너뜁니다.")
        return None

    try:
        # Cross-Play 파일 읽기
        cross_df = pd.read_csv(cross_csv_path)
        if 'total_reward' not in cross_df.columns or 'policy_labels' not in cross_df.columns:
            print(f"[{run_name}] Cross-Play 파일에 'total_reward' 또는 'policy_labels' 열이 없어 건너뜁니다.")
            return None

        # 페어 내 집계 행(annotation == 'mean') 은 통계에서 제외 — seed-* 행만 사용
        if 'annotation' in cross_df.columns:
            cross_df = cross_df[cross_df['annotation'] != 'mean']

        # 1. Self-Play(SP) 성능 계산: 자기 자신과의 대결 (예: cross-0_0, cross-1_1)
        sp_df = cross_df[~cross_df['policy_labels'].apply(_is_not_self_play)]
        if sp_df.empty:
            print(f"[{run_name}] SP 데이터가 없어 계산할 수 없습니다.")
            sp_mean = 0.0
            sp_std = 0.0
        else:
            sp_mean = sp_df['total_reward'].mean()
            sp_std = sp_df['total_reward'].std()

        # 2. Cross-Play(XP) 성능 계산: 서로 다른 인덱스 간의 대결 (예: cross-0_1, cross-1_0)
        xp_df = cross_df[cross_df['policy_labels'].apply(_is_not_self_play)]
        if xp_df.empty:
            print(f"[{run_name}] XP 데이터가 없어 계산할 수 없습니다.")
            xp_mean = 0.0
            xp_std = 0.0
        else:
            xp_mean = xp_df['total_reward'].mean()
            xp_std = xp_df['total_reward'].std()

        # 3. Gap 계산 및 모든 값을 정수로 변환
        sp_mean_int = int(round(sp_mean))
        sp_std_int = int(round(sp_std)) if pd.notna(sp_std) else 0
        xp_mean_int = int(round(xp_mean))
        xp_std_int = int(round(xp_std)) if pd.notna(xp_std) else 0
        gap = sp_mean_int - xp_mean_int

        return {
            'run_name': run_name,
            'sp-mean': sp_mean_int,
            'sp-std': sp_std_int,
            'xp-mean': xp_mean_int,
            'xp-std': xp_std_int,
            'gap': gap
        }

    except Exception as e:
        print(f"[{run_name}] 처리 중 오류 발생: {e}")
        return None


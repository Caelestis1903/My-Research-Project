"""Append closed core/AO04 evidence while preserving all historical question records."""
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parent
REG = ROOT/'question_registry'


def update():
    assert json.loads((ROOT/'core_mechanism/analysis_status.json').read_text())['complete']
    assert json.loads((ROOT/'ao04/analysis_status.json').read_text())['complete']
    path = REG/'N17_question_crosswalk.json'
    x = json.loads(path.read_text())
    before = [json.dumps(q['historical_record'], sort_keys=True, ensure_ascii=False) for q in x['questions']]
    edits = {
        1: dict(required_natural_work_remaining='none within core/N17'),
        4: dict(current_assessment='N17与core同状态诊断均完整；core合成两事件m/v/current网格有强范数交互，但不安装theta、不测风险，独立m/v来源不保证普通历史可达性，不能识别长期m或v单独必要性。', required_natural_work_remaining='none within frozen core/N17; long-term necessity remains unidentified'),
        8: dict(required_natural_work_remaining='transfer baseline cells'),
        9: dict(current_assessment='AC2D/N17与AO04完整自然结果仍按逐域任务保留；域比例比人工接近均衡，但匹配N仍改变支持/标签/特征条件，未建立失衡量的单因素识别。', required_natural_work_remaining='transfer'),
        10: dict(required_natural_work_remaining='transfer'),
        11: dict(required_natural_work_remaining='transfer'),
        13: dict(required_natural_work_remaining='transfer'),
        14: dict(required_natural_work_remaining='N19;CM19'),
        15: dict(current_status='NATIVE_FIELD_IDENTITY_NUMERICALLY_RETAINED', evidence_strength='STRONG', current_assessment='同人口同alpha的SN/SB原生detach训练场期望身份保留。160有限96行oracle相对残差约5.34e−8–9.84e−8；完整人口MC均值差/RMS和为.186–.510，不当作置信区间或无偏证明。重新连接的函数风险导数另列。', required_natural_work_remaining='none within core/N17'),
        17: dict(required_natural_work_remaining='none within frozen core/N17; pure-clock causality not identified'),
        21: dict(current_assessment='core人工SB/SN梯度trace1.113–2.435，自然全支持1.001–1.090；匹配N有一个末轮.987818反例。N17 P预条件排序反转仍只解释局部方差，core没有实际更新方差或风险投影证据，标量trace不是长期中介。', required_natural_work_remaining='none within prescribed core/N17; risk mediation remains unidentified'),
        28: dict(current_status='FORWARD_IDENTITY_AND_DERIVATIVE_BOUNDARY_MEASURED', evidence_strength='STRONG', current_assessment='160/160有限pool risk clone前向差严格0；重新连接导数与原生训练场的相对范数差约.000200–.349066，C占差异平方范数81.17%–近100%。不把有限pool风险导数冒充完整人口或实际训练场。', required_natural_work_remaining='none within core'),
        29: dict(current_assessment='core完整坐标统计包含C/P/A_arch，实际main排除A_arch；全trace与每块排序不完全相同，N17不同块定义及Adam更新trace不直接池化。', required_natural_work_remaining='none within core'),
        34: dict(required_natural_work_remaining='none within core; true m/v/delta telemetry retained'),
        35: dict(required_natural_work_remaining='none within core; sparse checkpoints do not identify lag causality'),
        40: dict(current_assessment='S5/S6持久history/current控制与core合成两事件网格均完整；前者共同改变路径与m/v，后者不测风险且部分来源组合非普通历史。状态/范数差不能识别m单独必要性。', required_natural_work_remaining='none within frozen core; isolated long-term necessity remains unidentified'),
        45: dict(required_natural_work_remaining='transfer'),
        51: dict(current_assessment='N17即时更新/局部风险与core160合成m/v/current网格均完成；非线性均值更新差、可达历史状态、持久训练作用和长期风险必要性是不同问题。', required_natural_work_remaining='none within prescribed core/N17; long-term necessity remains unidentified'),
    }
    for q in x['questions']:
        n = int(q['natural_reassessment']['id'][1:])
        if n not in edits:
            continue
        r = q['natural_reassessment']
        r.update(edits[n])
        uses_ao04 = n in {8, 9, 10, 11, 14, 45}
        evidence = [item for item in r['current_evidence'].split(';')
            if uses_ao04 or item != 'AO04_PHASE_ANALYSIS.md']
        additions = ['CORE_MECHANISM_ANALYSIS.md'] + (['AO04_PHASE_ANALYSIS.md'] if uses_ao04 else [])
        for item in additions:
            if item not in evidence:
                evidence.append(item)
        r['current_evidence'] = ';'.join(evidence)
    assert before == [json.dumps(q['historical_record'], sort_keys=True, ensure_ascii=False) for q in x['questions']]
    assert len(x['questions']) == 53 and x['all_53_historical_records_preserved']
    path.write_text(json.dumps(x, ensure_ascii=False, indent=2)+'\n')
    pd.DataFrame([q['natural_reassessment'] for q in x['questions']]).to_csv(REG/'N17_question_crosswalk.csv', index=False)
    print('Updated closed core/AO04 evidence; all 53 historical records preserved.')


if __name__ == '__main__':
    update()

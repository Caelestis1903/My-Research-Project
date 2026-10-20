"""Preserve all 8+11 original transfer questions and state frozen scope, without invented results."""
from pathlib import Path
import csv
import hashlib
import json
import re

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'analysis/question_registry'
rows=[];sources=[]
n19={
1:('E-P3;F-G0/G1/G4/G4LONG;F-M','自然路径/原生基线及匹配N/曝光可约束N或时钟单独解释；新自然结果未出。','函数保持gauge、纯几何/正则单因子仍未识别。'),
2:('E-NATIVE/P3 matched controls;F-G0M/G0MC','domain/cell codebook、STAR同任务逐cell的selected/final AUC/LL完整保留。','独立双backbone改变容量/成本，不是纯共享冲突估计。'),
3:('E-P2 all72;N19 audit/rank diagnostics','同原模拟器重算自然支持、class审核数、错设与强基线；四优化seed仍共享同审核draw。','独立审核draw、真实噪声通道与可识别收缩估计器不由此建立。'),
4:('E-P1 all40 including known-rule controls','自然支持上的同Z/Y1合成机制、DRO/ERM/known-mask/MMoE完整比较。','未知任务兼容规则学习、自然因果不变性与新shift总体未测。'),
5:('E-NATIVE P4;router diagnostics','full_mix/interaction/reliability/domain/axis等原强对照及有限直接调用成本完整迁移。','共同可行函数族下置信度是否超越匹配全局收缩未独立识别。'),
6:('E-NATIVE P5;credit diagnostics','同tower finite单源/当前混合效用、regret/entropy/uniformKL/coalition保留。','有限one-hot探针不等于训练更新；新独立meta总体决策效用仍未测。'),
7:('E-P6 all32;mask/budget diagnostics','原每访问label mask、静态端点/独立CV/残差及实际budget全部迁移。','静态可缓存标签不是真实重复标注成本，持续观测/执行节省未建立。'),
8:('all frozen N19/CM19 native/controlled groups;core S5/S6','核心已显示selected/final及AUC/LL反向；新transfer逐cell/权重/阶段仍待完整。','新部署效用、独立总体、N因果不靠事后选择更有利端点解决。')}
cm={
1:('F-G0/G1 + same-state path/gradient probes','自然ML路径、原生L2/data梯度及factor局部处理按原方法观察。','正则/尺度/几何/旁路竞争仍缺函数保持gauge的独立干预。'),
2:('F-G0/G1 original nominal AdamW/WD0','原float32 lr/WD保留；核实际乘数与模型canonical SHA，不能暗改精度。','真正可表示的正解耦收缩仍未测试，不把名义WD当有效收缩。'),
3:('F-G0/G1/G2/G3 full baselines','恢复路径、即时移除与从头重训区分；逐任务强fixed/PLE对照完整。','活动性不等于有益；一般有害失活及普遍竞争性救援不能预设。'),
4:('all groups seeds2022-2025','新自然人口是探索性复现，优化seed不是独立数据总体。','未选择新候选，不自动新增Ali-CCP/新总体；总体确认仍待独立设计。'),
5:('all source-preserved adapted models','继续忠实旧公开适配实现，HAMUR的batch语义与neutral动作按模型分别报告。','不能由本矩阵判严格paper工业复现或跨模型模块尺度优劣。'),
6:('F-G0M/G0MC + same-task single-task baselines','相同mean-task目标标度；双backbone容量/执行成本单列。','不是等容量纯任务冲突因果，不能把单任务0/1交叉作MDMT宏比较。'),
7:('F-G4/G4LONG +F-M artificial/matchedN/exposure','固定物理d2降比例后的角色重算；自然.25 d2成为最小，旧人工.25仍最大。','单嵌套子集、重复/独立支持/标签特征联合变化；N是否主要原因仍待新结果。'),
8:('E-NATIVE original/original_sample;F-G0 original bridge','保留两个研究原始aux-batch时钟；本轮不把N19和CM19 original误alias。','目标/重叠辅助优化等同时变化，不能唯一拆分贡献。'),
9:('all selected/final plus curve/cost/path receipts','核心S5/S6已见完整预算final相对救援但selected更差；新CM阶段/LL结果尚待。','LL恶化不识别校准/label shift；eval率entropy不是部署可拟合常数。'),
10:('all fixed-probe/seed records and numerical identity checks','精确预测中性限定固定点；四seed点态区间与相关cell如实报告。','不等于数学全局零函数/永久吸收态或新总体确认。'),
11:('F cost profiles39 + local/probe96','保留原meta batch维度、支持算子计数、实际wall time及从头重训成本。','冻结参数不等于物理剪枝；共享主机时间/形状估算不建立生产加速。')}
for family,name,mapping in [('N19','N19_OPEN_PROBLEMS_ORIGINAL.md',n19),('CM19','CM19_OPEN_PROBLEMS_ORIGINAL.md',cm)]:
    path=OUT/name;raw=path.read_bytes();digest=hashlib.sha256(raw).hexdigest();sources.append(dict(path=str(path),sha256=digest))
    count=0
    for lineno,line in enumerate(raw.decode().splitlines(),1):
        if family=='N19' and line.startswith('|Can ') or family=='N19' and line.startswith('|Does ') or family=='N19' and line.startswith('|What ') or family=='N19' and line.startswith('|When ') or family=='N19' and line.startswith('|Why '):
            count+=1;question=line.split('|')[1];original=line
        elif family=='CM19' and re.match(r'^\d+\. \*\*',line):
            count+=1;question=re.search(r'\*\*(.*?)\*\*',line).group(1);original=line
        else:continue
        scope,assessment,remaining=mapping[count]
        rows.append(dict(id=family+'-O'+str(count),family=family,original_question=question,original_record=original,
            source_file=name,source_line=lineno,source_sha256=digest,frozen_scope=scope,
            current_assessment=assessment,unidentified_or_future=remaining,
            new_transfer_result_status='PENDING',overall_natural_study_complete=False))
    assert count==len(mapping)
assert len(rows)==19
with (OUT/'transfer_open_question_crosswalk.csv').open('w',newline='') as h:
    w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(OUT/'transfer_open_question_crosswalk.json').write_text(json.dumps(dict(status='INTERIM_NOT_FINAL',sources=sources,
    original_question_count=19,automatic_new_experiments_triggered=False,questions=rows),ensure_ascii=False,indent=2)+'\n')
print('Preserved original N19 8 + CM19 11 questions; new transfer outcomes pending')

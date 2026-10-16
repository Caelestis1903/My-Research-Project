# DESIGN ADDENDUM — AO04 embedded selected-state recovery

2026-10-16，主代理；在实现前记录。WP1读取发现五个last_resume.pt存在，包含model_state_dict、selected_cpu_state、两optimizer、RNG和批次信息。旧按文件名检查selected_checkpoint.pth的missing索引不能作为选中模型不存在的证明。本补充属于原WP1/WP2/Stage5零训练取证，不新增训练，不修改历史源文件或原实验定义。

对实际存在的五个resume逐一：记录SHA/字段/step/epoch/selected_epoch；config中R2源码hash核验、manifest的data身份核验；strict load selected_cpu_state。按原dataset/setting/seed loader恢复完整validation顺序，检查旧NPZ全部labels及预测；新CPU完整forward与旧pred最大残差≤3e-6才将row binding升级为RECONSTRUCTED_LOADER_AND_FULL_PREDICTIONS_MATCH。该绑定不是原run独立保存row IDs的事实，不改旧NPZ。若原NPZ不存在，只记recovered weights，不补造旧指标。

对有可用B1 q参数的状态（包括未完整训练的seed2023，只作partial诊断）检查selected和final两个实际状态。每域固定validation前16真实row，不按标签选样本；这些确定性探针不是域总体估计。通过原模型forward hooks记录三domain expert输出及各cell gate；逐row autograd得到dlogit/dq、dprob/dq。记录rho、aD(1-bD)、rho(1-rho)、expert差范数、branch norm、gate entropy及数值有限差分（h=.01）。不将低entropy、分支间范数或有限sample重复当作collapse证明。

规划§12把`aD(1-bD)rho(1-rho)(Ej-Er)`标为dF/drho，实为dF/dq（rho=sigmoid(q)）；dF/drho=aD(1-bD)(Ej-Er)。这是导数变量的文字错误，报告两种量，不静默修科研假设。

阻塞：state/config/source不匹配、完整validation预测不匹配即停止该case；不可恢复的全epoch/全部seed敏感度仍INCOMPLETE。实际观察结果不改变B3/B4暂停、Stage4历史S5门及新训练门。

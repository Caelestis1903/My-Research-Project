# F0-B1-ML GPU/资源 preflight

状态：**PASS_FOR_SEARCH_GATE_WITH_FIRST_EPOCH_MEASUREMENTS_PENDING**。无真实数据训练，无test读取；首条search轨迹的首epoch仍用于记录实际epoch耗时、CPU/GPU峰值与checkpoint bytes。

- GPU：2张，usable=[0, 1]，并行上限=min(5, usable)=2。
- Driver/runtime：0, NVIDIA GeForce RTX 3090, GPU-4ac4b341-3045-1093-fec3-492516863e5a, 570.144, 24576 MiB, 4481 MiB, 19655 MiB, 0 %
1, NVIDIA GeForce RTX 3090, GPU-2a59fb49-6bef-5320-74ab-ffb24d056908, 570.144, 24576 MiB, 277 MiB, 23859 MiB, 0 %；Python /home/xzf/miniconda3/envs/phase0-3-env/bin/python，PyTorch 2.9.0+cu126 / CUDA runtime 12.6。
- Deterministic settings：`{'deterministic_algorithms': True, 'cudnn_deterministic': True, 'cudnn_benchmark': False, 'cudnn_allow_tf32': False, 'cuda_matmul_allow_tf32': False, 'amp_enabled_by_protocol': False, 'model_dtype': 'float32', 'cublas_workspace_config': ':4096:8'}`。
- 4096和40960 batch × M3OE/MMoE/CGC-1的6项synthetic FP32 forward/backward smoke均PASS；重复输出/loss/gradient一致，模型权重未变，无optimizer.step。峰值显存见逐GPU JSON。
- 合成探针用冻结模型和相同输入shape，未读取训练/验证/测试样本。真实首epoch资源仍pending；未虚构时间。
- RAM：`{'total_bytes': 67181826048, 'available_bytes': 63424290816}`；输出盘：`{'path': '/home/xzf/Code/phase0-3/experiments/stage_F0_outputs', 'free_bytes': 300342939648, 'total_bytes': 983344672768, 'used_bytes': 632975036416}`。
- GPU0已有外部进程约占4.2GiB；没有干预该进程。GPU分配时每卡最多1个worker。


- 失败尝试已保留在 `batch_memory_smoke_attempts.json` 与 `batch_memory_smoke_gpu*_error.json`：第一次是预检脚本import路径错误，第二次是重置峰值统计前CUDA context未初始化；两者均属明显执行初始化问题。修复后每GPU各一次重试成功，共6项PASS；没有改数据、模型、batch或dtype，也未调用optimizer.step。
- 6/6 queue resource-preflight工件已完成。其 `runtime_preflight` 的初始默认 deterministic/TF32设置记录为环境原态；runner的显式训练设置检查单独记为deterministic=true、TF32=false、AMP=false。真实epoch字段保持pending。

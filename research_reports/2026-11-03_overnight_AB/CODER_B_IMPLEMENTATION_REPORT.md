# B组合实现报告

实现依据为同目录`DESIGN_SPECIFICATION.md`和冻结规划03；实施阶段完成，待独立review及真实GPU batch smoke准入。

## 实现内容

- `optimizers.py`：`SerialCPOptimizer`继承只读原`PairOptimizer`的参数分组、ordinary Adam及RAM状态导入导出。原P01先更新选中h和全部active时钟；二阶矩仅递推一次未投影混合梯度b²，源`project_history`仅修复u=b−lambda*(w/q)*r；修复值进入m，PAIR历史H保留原定义，shared/ordinary参数各更新一次。pair=None不读取H，既有history表保留。
- `objectives.py`：原生A0/C/P/G关闭分支直接调用固定源入口；CP保留原PAIR forward、logit排名目标、采样随机流和backward，然后按源C构建probe，最后一次更新及EMA。CG提供冻结Gaussian目标＋源C修复接口，c_mix与Gaussian的kappa分离，是否启用由运行层冻结配置控制。
- `worker_b.py`：实现共同create/advance/export/restore接口，恰好推进一个完整epoch。保留源初始化、seed+11000的torch shuffle、原batch及尾批，两个估计器独立V选点，selected/E10/E50的完整模型状态、optimizer历史、EMA、loader、RNG只驻RAM。返回原始标量batch trace、成本、数据顺序hash、参数分组/数量/hash及端点身份；不读取test。
- `tests/test_combined.py`：六项直接涉及训练语义的CPU检查。

旧history/pair/vendor和历史报告未修改，未引入依赖，未执行科研训练或结果分析。

## 实际检查

工作目录`/home/xzf/Code/phase0-3_overnight_ab`：

```sh
PYTHONPATH=/home/xzf/Code/phase0-3_overnight_ab:/home/xzf/Code/phase0-3 \
 /home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m pytest \
 experiments/overnight_ab_20261103/tests/test_combined.py -q
```

结果：`6 passed, 1 warning in 2.00s`。warning来自固定源设置TF32旧API，未修改固定源。该次测试在Git取回期间通过PYTHONPATH只读接入历史依赖，未改源码搜索实现。

覆盖：

1. 手算CP的m/v/h/时钟、唯一共享和ordinary更新、耦合L2一次。
2. 非空投影修复真实项、v保持未投影混合梯度。
3. pair=None保留历史但不添加H，以及optimizer RAM往返后后续更新精确一致。
4. 四个关闭分支调用原生入口，CG配置中两种kappa互不覆盖。
5. tiny M3模型上的PAIR+C实际autograd验证probe不改变已写p.grad、一次EMA、一次优化更新及预期forward计数。
6. 连续两epoch与一epoch→RAM导出→重建恢复→一epoch的model/EMA/optimizer/RNG/loader/selected/快照、训练顺序和标量trace精确相同。

## 文件身份

| 文件 | SHA256 |
|---|---|
| optimizers.py | `495ae87c4b120d484ec9ce308ca113a0690c886260af1639a1db68df9b7f7b01` |
| objectives.py | `da93fccd75deac10c948adddabe9630c5a22775256fad67b301756ddfdd4d397` |
| worker_b.py | `1303ecf65e3eca1b444e7b91858dfcf155052eab803a8ca3e13e952a2fdb4a12` |
| tests/test_combined.py | `40cdd6a5621dc595bc83c23d3f41331945a46f3f73f843b9226a64ad1d800c22` |

独立review后，runner需在两数据各执行一个40960原batch的CP smoke，记录实际显存/RAM与有限梯度。正式五seed训练和最终评价由运行层推进。当前B代码无已知CPU工程阻塞。

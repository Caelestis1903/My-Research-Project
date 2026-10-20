# 逐域任务比较数据

完整表以`.csv.gz`提交，`pandas.read_csv`可直接读取；`gzip -dc 表名.csv.gz`可恢复原CSV。脚本同时在执行工作目录生成未压缩表，避免在Git中重复保存同一大表。

`paired_seed_cell_deltas`保留实际模型任务；单任务STAR只配对同任务。`actual_size_roles_and_tradeoffs`同时保留候选/基线的独立main支持排序，不把重放曝光当独立N。支持干预改变排序时，以基线排序解释干预效应，并明确物理域角色跨越；算法内同人口比较不得和数据重分配混合。

来源与配方继承`../transfer/{source_index.csv,run_config_index.csv,comparison_definitions.json}`。点态四seed区间不是独立总体确认。

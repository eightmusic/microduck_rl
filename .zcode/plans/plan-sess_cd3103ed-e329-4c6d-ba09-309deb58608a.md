直接创建 `docs/microduck_velocity_env_cfg_zh.py`，它是当前 `src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py` 的中文详细注释学习副本，不参与任务注册、不修改训练代码。

实现方式：
- 保留原文件的真实配置结构、函数名、参数名、配置 key 和关键代码逻辑，避免把学习内容变成脱离源码的伪配置。
- 在文件开头明确说明：这是学习副本，真实训练仍使用 `src/.../microduck_velocity_env_cfg.py`；不要从 `docs` 导入它。
- 按全局类别在源码中加入中文分区注释：全局开关/常量、rough terrain、传感器、基础模板继承、scene/robot/action、reward、events/domain randomization、observations、commands、pose rewards、terrain、curriculum、PPO。
- 对 `make_microduck_velocity_env_cfg(play=False, rough=False)` 的两个参数、返回对象、每类配置项及其运行时用途写出详细中文注释；重点标注 14D action、61D actor、76D critic、`passive_*` 过滤、command 拼接顺序、event mode、reward 符号、curriculum step 单位。
- 因为这是 Python 副本而不是被执行的配置，必要时将原始实现体保留为教学代码并用注释说明调用关系；不在 `tasks/` 目录创建可被插件误导入的模块。

验证：创建后用 `python -m py_compile docs/microduck_velocity_env_cfg_zh.py` 做语法检查，并用文本检查确认关键函数、分类标题、任务维度和“学习副本不参与注册”说明存在；不运行训练、不修改其他文件。
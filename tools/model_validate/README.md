# Model Validation Skeleton

这个目录预留给“网络预测结果 + ROS2/MoveIt 快速验证”的功能。

当前只提供最基础骨架，便于后续继续扩展，同时保证上传 Git 时不会把验证数据和结果一起提交。

目录约定：

- `configs/robots/<vendor>/<model>/`
  - 验证配置，按机械臂分类。
- `scripts/`
  - 本地和服务器的验证入口脚本。
- `inputs/`
  - 待验证输入数据，例如网络随机采样点、拉回点、预测结果。
  - 默认不上传。
- `outputs/`
  - 验证统计结果、失败样本、可视化产物。
  - 默认不上传。

后续建议的主入口会是：

```bash
ros2 run reachability_cli network_validation_cli --config tools/model_validate/configs/robots/aubo/aubo_i5/validate_default.yaml
```

当前 `network_validation_cli` 还没有实现，所以 runner 只保留环境准备和占位调用逻辑。

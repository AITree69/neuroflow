# NeuroFlow 阶段路线图

> 从 0 到 1 的可执行路线图。每阶段都有**交付物**、**可验证指标**、**退出条件**。

---

## 阶段总览

| 阶段 | 名称 | 时长 | 核心目标 | 状态 |
|---|---|---|---|---|
| **0** | 启动 | M1-M3 | 团队 + 仓库 + RFC | ✅ 当前 |
| **1** | **MVP（神经算子闭环）** | M3-M9 | 训练 FNO → 导出 IR → C++ 推理 → 验证精度 | 🔥 本阶段 |
| 2 | 神经算子覆盖 | M9-M15 | 多种算子 + 量化 + 首个领域 SDK |
| 3 | 数值方法耦合 | M15-M21 | FEM/FVM + 混合求解图 + OpenFOAM 插件 |
| 4 | 分布式 + HPC | M21-M27 | MPI/K8s/Triton + 工业 benchmark |
| 5 | 领域生态 | M27-M36 | 6 领域 SDK + 工业 POC + 商业版 |

---

## 阶段 0：项目启动（M1-M3）

**目标**：把脚手架立起来，圈定创始团队

**交付物**：
- [x] 项目 README（定位/架构/路线图）
- [ ] 创始团队招募（3-5 人）
- [ ] 仓库脚手架：CMake / pyproject / pre-commit / CI
- [ ] 公开 RFC-0001（NeuroIR 设计）

**退出条件**：第一个 PR 合入，CI 跑通

---

## 阶段 1：MVP — 神经算子闭环（M3-M9）🎯

**目标**：验证"训练 → 序列化 → C++ 推理"全链路可工作

**最小可交付范围**：
1. **Python 侧**
   - `FNO1d` 实现（PyTorch）
   - Burgers 1D 数据集（有限差分生成）
   - 标准 Trainer（Adam + L2 loss + lr schedule）
   - `.neuroir` v0 序列化（JSON + base64 权重）

2. **C++ 侧**
   - 轻量 Tensor 类（row-major 连续，支持 float32）
   - 简单 FFT（Cooley-Tukey radix-2，零外部依赖）
   - FNO 推理（线性层 + 频谱卷积 + ReLU + bias）
   - CLI：`nflow_infer --model foo.neuroir --input x.npy --output y.npy`
   - pybind11 绑定：`nf.cpp_runtime.infer(model_path, x)` 即可

3. **演示**
   - 训练 1D Burgers FNO（< 5 分钟训完，CPU 可跑）
   - 导出 IR，C++ 加载
   - 精度验证：与 PyTorch 推理误差 < 1e-4
   - 性能对比：单 batch 推理延迟 baseline

**可验证指标**：
| 指标 | 目标 |
|---|---|
| 训练时间（CPU） | < 5 min（5000 步） |
| L2 相对误差（Burgers test） | < 5% |
| C++ 精度（vs PyTorch） | max abs diff < 1e-4 |
| C++ 推理延迟（CPU） | < 50 ms（batch=1, N=256） |
| .neuroir 文件大小 | < 5 MB（典型 FNO1d） |
| 跨平台 | Linux + Windows MSVC + macOS |

**退出条件**：
- 一个 5 分钟 demo：`python examples/01_train_burgers1d.py && python examples/02_export_and_infer.py`，跑通
- `tests/test_fno.py` + `cpp/tests/test_runtime.cpp` 全绿
- README 增加"Quick Start"章节

---

## 阶段 2：神经算子覆盖（M9-M15）

**目标**：从单 FNO1d 扩展到算子全家族 + 量化

**交付物**：
- FNO2d / FNO3d
- DeepONet
- Transolver / GNO
- 多通道 / 多分辨率
- INT8 / FP8 量化
- 第一个领域 SDK：`heat`（芯片瞬态热，2D FNO2d）

**退出条件**：在 2 个公开 benchmark（NS-2D / Heat-2D）上达到 SOTA baseline 精度

---

## 阶段 3：数值方法耦合（M15-M21）

**目标**：神经算子 + 传统数值方法在同一张计算图

**交付物**：
- `nflow-fem` v0（线弹性 + 稳态热）
- `nflow-fvm` v0（不可压 N-S 简化版）
- 混合求解图（Hybrid Solve Graph）原型
- OpenFOAM `functionObject` 插件
- 第二个领域 SDK：`cfd`

**退出条件**：在 1 个 OpenFOAM benchmark 上展示"FEM+FNO 混合"比纯 FNO 精度高 5x

---

## 阶段 4：分布式 + HPC（M21-M27）

**目标**：从单机到多机、从研究到生产

**交付物**：
- MPI + NCCL 多卡/多机训练
- K8s CRD Operator
- Triton Inference Server Backend
- 性能 benchmark：vs ANSYS 在 3 个工业 case 上达到 100x+ 加速

**退出条件**：1 家 EDA 客户完成付费 POC

---

## 阶段 5：领域生态 + 商业化（M27-M36）

**目标**：6 领域 SDK 全覆盖，ARR 起步

**交付物**：
- 6 大领域 SDK v1：cfd / heat / em / structural / grid / climate
- 工业合作 2-3 家付费 POC
- 顶会论文 5-10 篇
- 社区贡献者 100+
- 商业版：NeuroFlow Enterprise（高级 UQ / 认证模型 / SLA）
- 推动 NeuroIR 标准化（IEEE / INCITS）

**退出条件**：M36 年化经常性收入 ARR 5M USD

---

## 一阶段里程碑详细分解

### 阶段 1 子任务

| 周 | 任务 | 负责人 | 验证方式 |
|---|---|---|---|
| W1 | 仓库脚手架 + CI | C++ Lead | CI 跑通 |
| W2 | FNO1d PyTorch + 单元测试 | ML Lead | `tests/test_fno.py` |
| W3 | Burgers 数据集 + 数据管线 | ML Lead | 可视化样本 |
| W4 | Trainer + 训练脚本 | ML Lead | 收敛曲线 |
| W5 | IR spec v0（JSON） | 系统 Lead | spec 文档 |
| W6 | PyTorch → IR 导出 | ML Lead | roundtrip 测试 |
| W7-W8 | C++ Tensor + FFT + FNO forward | C++ Lead | 单元测试 |
| W9 | pybind11 绑定 | C++ Lead | Python 调通 |
| W10 | CLI + end-to-end demo | 全员 | 5 min 跑通 |
| W11 | 性能 benchmark + 文档 | 全员 | README 更新 |
| W12 | 发布 0.1.0 + RFC-0002 启动 | 全员 | GitHub release |

### 风险与应对

| 风险 | 概率 | 应对 |
|---|---|---|
| 跨平台构建出问题 | 高 | 第 1 周就配 Linux + Windows + macOS 三平台 CI |
| 数值精度不一致 | 中 | IR 强制 FP32，对齐 PyTorch；后续加 FP16/FP8 |
| 训练时间超过 5 分钟 | 中 | 先用小模型（width=32, modes=8）做 baseline |
| 团队招不到人 | 中 | 远程 + 兼职可接受 |

---

## 版本号约定

- **0.1.x**：阶段 1（MVP）— 神经算子闭环
- **0.2.x**：阶段 2 — 算子覆盖
- **0.3.x**：阶段 3 — 数值耦合
- **0.4.x**：阶段 4 — 分布式
- **0.5.x**：阶段 5 — 领域生态
- **1.0.0**：API 稳定 + 商业版就绪

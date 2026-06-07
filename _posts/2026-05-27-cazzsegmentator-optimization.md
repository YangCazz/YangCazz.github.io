---
layout: post
title: WiseSegmentator 工程优化：消除僵尸代码、静默错误与副作用
date: 2026-05-27 23:00:00.000000000 +08:00
categories:
- AI
- 医学图像
- 软件工程
tags:
- 医学图像分割
- nnUNet
- Python
- 代码重构
- CI
excerpt: 从 P0 僵尸抽象和静默吞错，到 P3 CI 烟雾测试——记录 WiseSegmentator 四轮工程优化中踩过的坑、做出的取舍，以及研究代码走向生产级必须解决的六类问题。
image: "/assets/images/covers/ai-dev-tools.jpg"
---


## 引言

[上一篇]({% post_url 2026-05-26-cazzsegmentator-design %})我写了 WiseSegmentator 的架构设计——两级目标体系、ROI 裁剪推理、Predictor 生命周期管理。那些是"看得见"的设计。

这篇文章写的是"看不见"的东西：**一个研究项目型的代码仓库，在架构定型之后、真正达到可维护状态之前，还需要经过哪些打磨。**

我把问题按严重程度分为 P0→P3 四个等级，逐轮修复。回头看，每轮修的都是不同层级的问题——从"代码能不能跑对"到"别人能不能参与贡献"。

---

## 问题分级框架

| 等级 | 定义 | 典型表现 |
|------|------|---------|
| **P0** | 正确性风险 — 代码在特定条件下会静默失败 | 僵尸抽象、异常被吞、环境变量被覆盖 |
| **P1** | 开发者体验 — 功能正确但阻碍调试和维护 | 裸 print、tqdm 与日志冲突、无持久化日志 |
| **P2** | 功能完整性 — 缺少用户可感知的关键功能 | 无缓存/断点续跑、无输入校验、优先级黑盒 |
| **P3** | 质量保障 — 缺少自动化验证手段 | 无 CI、测试覆盖薄弱、mock 不足 |

这个分级来自于一个朴素的原则：**先保证代码不撒谎，再让它易维护，然后补全用户需要的功能，最后用自动化守住底线。**

---

## P0：三类会"说谎"的代码

### ModelManager：一个从未被实现的抽象

打开 `wisesegmentator/models/manager.py`，你能看到一个叫 `ModelManager` 的类，提供了 `load_model()`、`download_model()`、`clear_cache()` 等看起来"很正式"的接口：

```python
class ModelManager:
    def load_model(self, model_name: str):
        ...
        return self._load_model_impl(model_path)

    def _load_model_impl(self, model_path):
        raise NotImplementedError  # 从第一天就不存在
```

它承诺了模型加载、下载、缓存管理——但实际上 `_load_model_impl()` 永远抛出 `NotImplementedError`。真正的模型加载在 `executor/model_executor.py` 里通过 `WiseNNUNetPredictor` 完成，走的完全是另一条路。

**这是一种典型的"研究代码遗留"**：项目初期按想象中的架构搭建骨架，但实际执行路径绕过了它。类留在那里，给后来的维护者一个假象——"这里有一个模型管理器，你可以通过它扩展"。

**处理方式**：直接删除。没有重构，没有"以后可能会用"，没有"留个 TODO"。

在清理研究代码时有一条铁律：**僵尸代码不是技术债务——它是谎言。** 技术债务是有价值但需要改进的代码；谎言是从未履行过承诺的代码。对前者要重构，对后者要删除。

连带被清理的还有 `predict_from_nifti()` 和 `predict_from_array()`——这两个方法在 `WiseNNUNetPredictor` 中被定义但从未被调用，唯一被使用的是 `predict_from_file()`。以及 `spatial_restore.py` 中 ~265 行的废弃函数 `restore_to_original_space` 和相关辅助函数，在重构后只保留了 `get_image_metadata`。

### Monkey-patch 失败静默吞掉

TotalSegmentator 的肺结节模型使用了一个自定义 Trainer `nnUNetTrainer_MOSAIC_1k_QuarterLR_NoMirroring`，它不在标准 nnUNetv2 的类注册表中。为了让 nnUNet 能识别它，需要在模块导入时 monkey-patch `recursive_find_python_class` 函数：

```python
# 修复前
def _inject_custom_trainers():
    try:
        from nnunetv2.utilities.find_class import recursive_find_python_class
        # ... 注入自定义 trainer
    except Exception:
        pass  # 失败了也无所谓？
```

`except Exception: pass` ——没有日志，没有 warning，什么都没有。如果 nnUNetv2 的 API 变了导致 monkey-patch 失败，用户只会在运行到肺结节模型时遇到一个神秘的 `ClassNotFoundError`，而完全不知道根因。

**修复方式**：

```python
# 修复后
def _inject_custom_trainers():
    try:
        from nnunetv2.utilities.find_class import recursive_find_python_class
        # ... 注入自定义 trainer
    except Exception:
        import warnings
        warnings.warn(
            "Failed to inject TotalSegmentator custom trainers. "
            "ts_lung_nodules model may not work.",
            RuntimeWarning
        )
```

`bare except: pass` 是 Python 代码里最常见的反模式之一。如果一段代码的失败值得写 try-except，那它一定值得记录。

### 导入时副作用覆盖环境变量

最隐蔽的 bug 往往和导入顺序有关。

```python
# nnunet_predictor.py 模块级代码
def _setup_nnunet_env():
    os.environ["nnUNet_raw"] = "/tmp/nnunet_raw"
    os.environ["nnUNet_preprocessed"] = "/tmp/nnunet_preprocessed"
    os.environ["nnUNet_results"] = "/tmp/nnunet_results"

_setup_nnunet_env()  # 模块导入即执行
```

这段代码的动机是好的——抑制 nnunetv2 启动时检查环境变量的警告。但它有两个问题：

1. **覆盖用户配置**：如果用户已经设置了 `nnUNet_results` 指向自己的模型目录，导入 WiseSegmentator 后会悄无声息地被覆盖。
2. **时机错误**：调用发生在 `WiseNNUNetPredictor.__init__` 里，但 nnunetv2 在导入时就检查环境变量——太晚了，警告已经打出来了。

**修复**：

```python
def _setup_nnunet_env():
    for var in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
        if var not in os.environ:
            os.environ[var] = "/tmp/nnunet_default"
```

两处改动：检查 `not in` 避免覆盖已有值；调用从类的 `__init__` 提升到模块级别（在 `from nnunetv2 import ...` 之前）。这次改动同时消除了导入阶段的警告和运行时的副作用。

---

## P1：开发体验 —— 你不可能调试看不见的东西

### 日志系统的六个级别

最初的代码里混用 `print()` 和 `logging`，而且 tqdm 进度条和日志消息互相争抢 stdout。对于命令行工具来说，日志输出质量直接决定用户的第一印象。

重写的日志系统在 `utils/logging.py` 中集中管理，核心设计：

| 级别 | 标签 | 颜色 | 用途 |
|------|------|------|------|
| DEBUG | `[DEBUG ]` | 灰色 | shape/spacing/GPU 显存诊断 |
| INFO | `[ INFO ]` | 白色 | 模型开始、流程步骤 |
| SUCCESS (25) | `[  OK  ]` | 绿色 | 模型完成、后处理完成、总耗时 |
| WARNING | `[ WARN ]` | 黄色 | 非致命错误 |
| ERROR | `[ERROR ]` | 红色 | 模型失败 |

SUCCESS 是用 `logging.addLevelName` 注册的自定义级别（25，介于 INFO 和 WARNING 之间）。6-char 固定宽度的级别标签保证输出对齐，ANSI 颜色码只在终端输出时启用（通过 `sys.stderr.isatty()` 判断），写日志文件时不带颜色码。

tqdm 兼容使用标准的 `tqdm.write()` 模式：自定义 handler 在写消息前通过 `tqdm.external_write_mode()` 上下文管理器暂停进度条，避免输出撕裂。

### 为什么这点很重要

修复 P0 问题时我第一次意识到：**如果 monkey-patch 失败时有 warning，这个问题可能一周前就被发现了。** P1 的日志优化不是为了"好看"——它是所有后续调试工作的基础设施。

---

## P2：用户能感知到的功能缺口

### 指纹缓存：为什么重复运行要重新推理？

这是实际使用中暴露的痛点。一个场景：用户跑了 `wiseseg -p all`，发现肝脏分段不太对，想调一下后处理参数。但重新运行时，所有 9 个模型又要全部推理一遍——4 分多钟的等待，只为了改一个后处理阈值。

指纹缓存的思路很简单：

```
输入 Nifti → SHA256 → 检查 models/model-{name}/.fingerprint.json
  ├── 命中: 加载 _segmentation_full.nii.gz → 跳过推理
  └── 未命中: 执行推理 → 保存指纹 + 结果
```

实现细节在 `executor/model_executor.py` 中，三个方法：

- `_compute_input_hash()`：对输入文件分块计算 SHA256（8KB chunks，避免大文件内存问题）
- `_check_fingerprint()`：读取 `.fingerprint.json` 比对 `input_sha256` 和 `model_name`
- `_save_fingerprint()`：保存指纹 JSON + 完整分割结果

缓存按模型粒度而非流水线粒度——这意味着即使你修改了 `task_registry.json` 只影响了一个模型，其他 8 个仍然可以命中缓存。

**一个重要的设计决策**：缓存的 key 是 `input_sha256` 而不是输入文件路径。这意味着同一个文件移到不同目录仍然能命中。反过来，内容不同的文件即使路径相同也会重新推理。

### 输入预校验：早报错 vs 跑到一半才挂

```
原始行为：
用户输入 2D 切片 → 模型执行到一半 → nnUNet 报错 "expected 3D input"
                                → 栈回溯中找不到清晰的错误原因

修复后：
用户输入 2D 切片 → _validate_input() 立即返回: "Input must be 3D, got shape (512, 512)"
```

`_validate_input()` 在流水线入口执行四步检查：文件存在性 → Nifti 有效性 → 最少 3 维 → 每轴至少 2 体素。约 15 行代码，但节省了用户等待模型加载 50s 后才发现输入问题的时间。

### 显式 priority 字段

旧的模型优先级依赖名称前缀匹配：`custom_* = 1, ts_* = 2, cads_* = 3`。添加新模型时如果命名不规范，比如从第三方引入的叫 `external_xxx_*` 的模型，就会回退到 999（最低优先级）。

在 `task_registry.json` 中为每个条目增加 `"priority"` 字段后，`_get_model_priority()` 优先读取 JSON 配置，没配置才回退前缀规则。这比隐式约定更可靠。

---

## P3：CI 烟雾测试 —— 自动化守住底线

在 P0-P2 的修改过程中，我手动跑了无数次 `pytest`。每次提交都担心"前面的改动是不是偷偷弄坏了后面的东西"。

烟雾测试的定位不是在 GPU 上跑真实推理——那属于集成测试，需要硬件支持。烟雾测试的目标是：**用一个 mock 的 nnUNetPredictor，验证整个流水线的调度逻辑在"推理瞬间完成"的假设下是否正确。**

```python
@pytest.fixture
def mock_predictor():
    """mock nnUNetPredictor，返回合成数据而非真实推理"""
    with patch("wisesegmentator.models.nnunet_predictor.nnUNetPredictor") as mock:
        mock.return_value.predict_from_files.return_value = [
            create_dummy_nifti((2, 2, 2))  # 合成 3D 分割结果
        ]
        yield mock
```

10 个测试覆盖了三个维度：

1. **完整流水线**（3 个）：单目标端到端、指纹缓存二次命中跳过、`-s` 二级过滤
2. **指纹缓存单元**（6 个）：SHA256 计算、指纹读写、缓存命中/未命中、损坏文件容错
3. **缓存跳过集成**（1 个）：`_execute_single_model` 命中指纹时跳过推理

有这 10 个测试兜底后，每次改调度逻辑——比如修改分组策略或依赖图——不需要手动跑一遍全量推理来验证。121 个单元测试在 CI 上 69 秒跑完。

---

## 六条经验

回头看这四轮优化，有几条规律值得记录。

### 僵尸代码不是"以后有用"，是"正在说谎"

删除未使用的代码比保留它风险更低。留下的空壳会给后来的维护者一个错误的承诺——"这个接口可以用"。尤其是定义了完整接口但实现只有 `raise NotImplementedError` 的类。

### `except Exception: pass` 是定时炸弹

任何 bare except 都要问自己：如果这段代码失败了，用户应该知道吗？答案几乎总是"应该"。即使一个错误在当前版本下"永远不会发生"，依赖的第三方库会升级、API 会变化。

### 模块导入时的副作用是隐式契约

导入一个库不应该改变全局状态。`_setup_nnunet_env()` 修了两个问题后才明白：在 `import` 时产生副作用是一层隐式契约，所有 import 这个模块的代码都隐式依赖这个副作用。理想情况下应该用延迟初始化或显式 `init()` 函数。

### 日志系统是调试基础设施，值得早期投入

P1 的日志重写改动不大，但它是 P0 问题能被快速定位的前提。如果在修复 ModelManager 错误时还是满屏 `print()` 混 tqdm，排查时间会翻倍。

### 测试不是越"真实"越好

烟雾测试的价值在于它 **mock 了最慢的部分而验证了最关键的部分**。nnUNet 推理在 GPU 上跑一次 4 分钟，mock 掉后 69 秒。但它完整覆盖了从 TargetManager 解析到 ModelExecutor 分组到指纹缓存比对的全链路调度逻辑。

### 清理顺序很重要

P0（正确性）→ P1（可观测性）→ P2（功能）→ P3（自动化）的顺序不是随机的。没有 P1 的日志，P0 的问题很难定位。没有 P3 的 CI，P2 的改动不敢放心交付。每一级为下一级提供工具或信心。

---

## 结语

研究和工程之间的鸿沟，不在于代码多不多、架构好不好，而在于——**当一段代码在你不看着它的时候，它能不能不出错地跑完。**

WiseSegmentator 还有大量值得做的事：ONNX 转换提速、Group 2 按 ROI 体积动态并行、自动模型权重下载。但经过这四轮清理后，它从一个"作者的机器上能跑"的项目变成了一个"可以交给别人用"的工具。

代码仓库：<https://github.com/YangCazz/CazzSegmentator>

---

## 参考文献

<ol class="references">
<li><em>WiseSegmentator 设计文档.</em> 统一多模型调度的 CT 多器官分割框架架构设计.<br><a href="https://github.com/YangCazz/CazzSegmentator/blob/main/docs/架构设计.md">https://github.com/YangCazz/CazzSegmentator</a></li>
<li><em>nnUNetv2.</em> Isensee F, et al. Self-configuring Method for Semantic Segmentation.<br><a href="https://github.com/MIC-DKFZ/nnUNet">https://github.com/MIC-DKFZ/nnUNet</a></li>
<li><em>TotalSegmentator.</em> Wasserthal J, et al. Robust Semantic Segmentation of 104 Anatomical Structures in CT Images.<br><a href="https://github.com/wasserth/TotalSegmentator">https://github.com/wasserth/TotalSegmentator</a></li>
</ol>

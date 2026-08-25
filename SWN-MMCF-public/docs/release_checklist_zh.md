# GitHub 发布前检查清单

当前目录已经可以作为审稿用公开包，但以下内容必须根据论文和真实实验环境补齐。不要直接带着 `TODO` 或 `REPLACE_ME` 发布。

## 必填信息

- [ ] 在 `README.md` 中写出论文对 SWN-MMCF 的正式全称。
- [ ] 在 `CITATION.cff` 中填写作者、论文题目、期刊/会议和 GitHub 地址。
- [ ] 在 `docs/reproducibility.md` 中填写实际使用的 OpenVINS 完整 commit SHA。
- [ ] 填写 ROS、Ubuntu、编译器、Eigen、OpenCV 和硬件版本。
- [ ] 填写实验所用数据集、序列、标定文件哈希和随机种子。
- [ ] 将 `config/swn_mmcf.example.yaml` 的示例值替换为论文实验值，或明确标注各数据集对应配置。
- [ ] 对照论文公式，统一“节点评分、约束构造、置信度、门限”等术语和符号。

## 保密边界检查

- [ ] `include/swn_mmcf/core_api.hpp` 只含函数声明、注释和成员字段，没有函数体。
- [ ] `pseudocode/swn_mmcf_pipeline.pseudo.cpp` 只描述步骤，不包含可直接恢复的核心公式、参数或权重。
- [ ] 未提交模型权重、私有数据、真实设备标定、密钥、token、bag/db3 数据文件。
- [ ] 未复制作者私有 OpenVINS fork 的代码。
- [ ] 未把该仓库描述成“完整可运行/完全可复现代码”。

## 许可证检查

- [ ] 保留根目录 `LICENSE` 和 `third_party/NOTICE.md`。
- [ ] 如果最终仓库实际加入或修改了 OpenVINS 源码，保留所有上游版权头和修改说明。
- [ ] 如果发布与 OpenVINS 组合后的二进制或衍生程序，先确认已满足 GPL-3.0 对相应源代码的提供义务。

## 发布命令示例

确认所有占位符已清除后，在该目录执行：

```bash
git init
git add .
git commit -m "Initial public interface release"
git branch -M main
git remote add origin https://github.com/YOUR_ACCOUNT/SWN-MMCF.git
git push -u origin main
```

建议先创建一个私有 GitHub 仓库供合作者检查，再切换为公开仓库。


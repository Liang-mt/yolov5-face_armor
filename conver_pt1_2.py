import time
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
import warnings
import sys
import gc
import yaml
import os

# 添加YOLOv5根目录到Python路径（根据实际路径调整）
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # 假设脚本在YOLOv5根目录或子目录
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.yolo import Model  # 导入YOLOv5模型类

# 忽略PyTorch权重加载的无关警告
warnings.filterwarnings("ignore", category=UserWarning)


def strip_optimizer_advanced(input_model='best.pt', out_model='out.pt',cfg_path='models/yolov5s.yaml', progress=False):
    """
    高级版：彻底重建模型，剔除所有训练冗余（类比第二个代码的优化逻辑）
    :param input_model: 输入模型路径
    :param out_model: 输出模型路径
    :param cfg_path: 模型配置文件路径（yaml），用于重建纯净模型
    :param progress: 是否显示进度条
    """

    def update_pbar(pbar, step, desc):
        """辅助函数：更新进度条或打印日志"""
        if progress and pbar:
            pbar.set_postfix_str(desc)
            pbar.update(step)
        elif not progress:
            print(f"▶ {desc}")

    try:
        # 1. 前置检查
        input_path = Path(input_model)
        if not input_path.exists():
            raise FileNotFoundError(f"输入模型文件不存在: {input_path.absolute()}")

        cfg_full_path = Path(cfg_path)
        if not cfg_full_path.exists():
            # 尝试相对于YOLOv5根目录查找
            cfg_full_path = ROOT / cfg_path
            if not cfg_full_path.exists():
                raise FileNotFoundError(f"找不到配置文件: {cfg_path}，请确保路径正确")

        # 2. 加载原始模型（阶段1：仅用于提取信息，不保留）
        main_pbar = tqdm(total=6, desc="🛠️ 模型深度优化", unit="step") if progress else None

        ckpt = torch.load(
            input_path,
            map_location=torch.device('cpu'),
            weights_only=False
        )
        update_pbar(main_pbar, 1, "原始模型加载完成")

        # 3. 提取关键信息（不保留原模型对象）
        # 优先使用EMA权重，否则用model
        model_ckpt = ckpt.get('ema', ckpt.get('model'))
        if model_ckpt is None:
            raise ValueError("模型文件中未找到'model'或'ema'键")

        # 确保是nn.Module并转为eval模式
        if hasattr(model_ckpt, 'eval'):
            model_ckpt = model_ckpt.float().eval()

        # 提取类别信息
        nc = getattr(model_ckpt, 'nc', None)
        if nc is None and hasattr(model_ckpt, 'model') and hasattr(model_ckpt.model[-1], 'nc'):
            nc = model_ckpt.model[-1].nc

        # 提取类别名
        names = ckpt.get('names')
        if names is None and hasattr(model_ckpt, 'names'):
            names = model_ckpt.names
        if names is None and nc is not None:
            names = [f'class_{i}' for i in range(nc)]

        update_pbar(main_pbar, 1, f"提取信息完成: {nc} classes")

        # 4. 核心优化：彻底筛选State Dict（剔除aux + 分离梯度）
        state_dict = {}
        aux_count = 0
        total_count = 0

        for k, v in model_ckpt.state_dict().items():
            total_count += 1
            if not k.startswith('aux'):
                # 关键：detach并禁用梯度，确保是独立张量
                state_dict[k] = v.detach().requires_grad_(False)
            else:
                aux_count += 1

        # 立即删除原模型，释放内存
        del ckpt
        del model_ckpt
        gc.collect()

        update_pbar(main_pbar, 1,f"State Dict筛选: 保留{len(state_dict)}/{total_count}, 剔除{aux_count}个aux参数")

        # 5. 重建纯净模型（关键步骤：类比第二个代码）
        with open(cfg_full_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)

        # 使用提取的nc，或从cfg获取
        if nc is not None:
            cfg['nc'] = nc

        # 重建全新模型实例（无训练历史）
        model = Model(cfg, ch=3, nc=nc).float().eval()
        update_pbar(main_pbar, 1, "纯净模型重建完成")

        # 6. 加载筛选后的权重（严格匹配）
        model.load_state_dict(state_dict, strict=True)
        update_pbar(main_pbar, 1, "权重加载完成")

        # 7. Fuse + FP16转换（在no_grad环境下，类比第二个代码）
        with torch.no_grad():
            # Fuse Conv+BN（减少计算量，融合权重）
            model = model.fuse()

            # 确保所有参数都不需要梯度（使用.data绕过leaf检查）
            for param in model.parameters():
                param.data = param.data
                param.requires_grad_(False)

            # 转换为FP16
            model = model.half()

            # 再次确认FP16参数状态
            for param in model.parameters():
                param.requires_grad_(False)

        update_pbar(main_pbar, 1, "Fuse完成 + FP16转换")

        # 8. 处理锚框（分离存储，避免重复）
        with torch.no_grad():
            if hasattr(model, 'anchors'):
                anchors = model.anchors
            elif hasattr(model, 'model') and len(model.model) > 0 and hasattr(model.model[-1], 'anchors'):
                anchors = model.model[-1].anchors
            else:
                # 从cfg获取默认锚框
                anchors = torch.tensor(cfg.get('anchors', [
                    [[10, 13], [16, 30], [33, 23]],
                    [[30, 61], [62, 45], [59, 119]],
                    [[116, 90], [156, 198], [373, 326]]
                ]))

            anchors = anchors.half().detach().requires_grad_(False)

        if main_pbar:
            main_pbar.close()

        # 9. 构建极简保存字典（仅推理必需，彻底清理）
        save_dict = {
            'model': model,  # 纯净推理模型（FP16 + fuse + 无aux）
            'nc': nc,  # 类别数
            'names': names,  # 类别名
            'anchors': anchors,  # 锚框
            'epoch': -1,  # 标记为推理专用
        }

        # 可选：添加stride信息（某些YOLOv5版本需要）
        if hasattr(model, 'stride'):
            save_dict['stride'] = model.stride

        # 10. 紧凑序列化保存
        save_path = Path(out_model) if out_model else input_path
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if progress:
            with tqdm(total=1, desc="💾 保存优化模型") as save_pbar:
                torch.save(save_dict, save_path, _use_new_zipfile_serialization=False)
                save_pbar.update(1)
        else:
            torch.save(save_dict, save_path, _use_new_zipfile_serialization=False)

        # 11. 结果统计
        orig_size = os.path.getsize(input_path) / 1024 / 1024  # 原始大小(MB)
        new_size = os.path.getsize(save_path) / 1024 / 1024  # 优化后大小(MB)
        reduce_ratio = 100 * (orig_size - new_size) / orig_size

        print(f"\n{'=' * 55}")
        print(f"✅ 深度优化完成")
        print(f"📊 体积优化: {orig_size:.2f}MB → {new_size:.2f}MB (↓{reduce_ratio:.1f}%)")
        print(f"🔧 优化措施:")
        print(f"   • 模型重建: 从cfg创建纯净实例，无训练历史")
        print(f"   • Aux剔除: 移除{aux_count}个辅助训练参数")
        print(f"   • FP16转换: 权重精度减半")
        print(f"   • Fuse优化: Conv+BN层融合")
        print(f"   • 紧凑序列: 最小化元数据开销")
        print(f"💾 输出路径: {save_path.absolute()}")
        print(f"{'=' * 55}")

    except Exception as e:
        print(f"\n❌ 优化失败: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='📦 YOLOv5模型深度精简工具：重建 + Aux剔除 + FP16 + Fuse')
    parser.add_argument('--input', type=str, default='runs/train/exp6/weights/best_1.pt',help='输入模型路径')
    parser.add_argument('--output', type=str, default='runs/train/exp6/weights/best_2.pt',help='输出模型路径')
    parser.add_argument('--cfg', type=str, default='models/yolov5s6.yaml',help='模型配置文件路径（用于重建纯净模型）')
    parser.add_argument('--progress', action='store_true', default=False,help='显示进度条')

    args = parser.parse_args()

    print(f"🚀 开始深度优化: {args.input}")
    strip_optimizer_advanced(
        input_model=args.input,
        out_model=args.output,
        cfg_path=args.cfg,
        progress=args.progress
    )
    print("\n🎉 优化完成！模型已准备就绪。")
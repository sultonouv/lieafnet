import argparse
import torch
from thop import profile
from train_supervision import py2cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_path', required=True)
    parser.add_argument('--size', type=int, default=1024)
    args = parser.parse_args()

    config = py2cfg(args.config_path)
    net = config.net.cuda()
    net.eval()

    x = torch.randn(1, 4, args.size, args.size).cuda()

    try:
        macs, params = profile(net, inputs=(x,), verbose=False)
        # NOTE: reported as raw MACs/1e9, not MACs*2/1e9, to match the
        # convention used to produce the paper's Table II GFLOPs figures
        # (4.04 @ 512x512, 16.17 @ 1024x1024).
        gflops = macs / 1e9
        print(f"RESULT model={config.model_name} params={params/1e6:.2f}M gflops={gflops:.2f}")
    except Exception as e:
        print(f"RESULT model={config.model_name} ERROR={e}")


if __name__ == '__main__':
    main()

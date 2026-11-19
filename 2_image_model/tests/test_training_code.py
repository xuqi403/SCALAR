import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from train import parse_args

def test_cli_fixed_to_d_robust():
    args=parse_args(['--manifest','manifest.csv','--dino-repo','dinov3',
                     '--dino-weights','weights.pth','--output-dir','outputs'])
    assert not hasattr(args, 'experiment')
    assert not args.run_test
    assert not args.init_checkpoint

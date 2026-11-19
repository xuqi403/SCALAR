from pathlib import Path
import sys,tempfile
import cv2,numpy as np,pandas as pd,torch
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R));from scalar_infer import model_loader, pipeline
import infer_video
class MockYolo:
    def __init__(self):self.confidence=.4
class FakeLib:
    CLIP_OFFSETS=(-4,-2,0,2,4);CENTER_INDEX=2;PRED_THRESHOLD=.5;YOLO_CONF_THRESHOLD=.4
    def import_yolo_components(self):return MockYolo,None,None,None
    def yolo_detect_boxes(self,yolo,pil,*_):return [{'x1':48,'y1':48,'x2':80,'y2':80,'score':.95}]
    def letterbox_image_and_mask(self,img,mask,out_size):return cv2.resize(img,(out_size,out_size)),None
    def image_tensor(self,img):return torch.from_numpy(img.transpose(2,0,1).copy()).float()/255
    def topology_cleanup(self,outer,lumen):
        m=np.zeros(outer.shape,np.uint8);m[outer]=1;m[lumen & outer]=2
        return m,{}
class FakeModel(torch.nn.Module):
    def extract_frame_features(self,clip):
        b,t,c,h,w=clip.shape
        return torch.ones((b,t,8,7,7),device=clip.device,dtype=clip.dtype)
    def fuse_once(self,feat,weights,center):
        return feat[:,center],{}
    def decoder(self,feat,target):
        b=feat.shape[0]
        y,x=torch.meshgrid(torch.arange(448,device=feat.device),torch.arange(448,device=feat.device),indexing='ij')
        r=((x-224)**2+(y-224)**2).sqrt()
        out=(110-r).float().view(1,1,448,448).expand(b,-1,-1,-1)
        lum=(65-r).float().view(1,1,448,448).expand(b,-1,-1,-1)
        return {'outer_logits':out,'lumen_logits':lum}

import unittest
class VideoPipelineTests(unittest.TestCase):
    def test_simulated_whole_video_pipeline(self):
        with tempfile.TemporaryDirectory() as t:
            t=Path(t)
            video=t/'test.mp4'
            v=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),25,(128,128))
            assert v.isOpened()
            for i in range(12):
                frame=np.zeros((128,128,3),np.uint8)
                cv2.circle(frame,(64,64),35,(75,75,75),-1);cv2.circle(frame,(64,64),20,(15,15,15),-1)
                v.write(frame)
            v.release()
            ckpt=t/'model.pth';ckpt.write_bytes(b'test')
            class Args:pass
            args=Args();args.output=t/'output';args.checkpoint=ckpt;args.overwrite=False;args.training_cache_manifest=None;args.no_overlay=False;args.feature_batch_size=8;args.fusion_batch_size=24
            original=model_loader.load_v5_model
            model_loader.load_v5_model=lambda lib,ckpt,device:FakeModel().to(device)
            try:
                result=pipeline.run_one(video,args,FakeLib())
                df=pd.read_csv(result/'frame_predictions.csv')
                assert len(df)==12
                assert int(df.inference_valid.sum())==12
                assert (result/'overlay.mp4').exists()
                assert (result/'masks_448.npy').exists()
                assert not (result/'physical_frame_measurements.csv').exists()
                assert not (result/'wall_thickness_map_mm.npz').exists()
                assert not any('area' in c or 'thickness' in c for c in df.columns)
                print('END-TO-END MOCK SMOKE PASS',len(df),'frames',int(df.inference_valid.sum()),'valid')
            finally:model_loader.load_v5_model=original

if __name__=="__main__":unittest.main()

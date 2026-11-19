import sys
from pathlib import Path
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dataset import read_manifest, CarotidDataset
from examples.make_synthetic_data import ellipse_points
import json, cv2, numpy as np


def construct(tmp_path):
    rows=[]
    for n, split in enumerate(('train','val','test')):
        p=tmp_path/f'{n}.png'
        cv2.imwrite(str(p),np.zeros((64,64,3),np.uint8))
        j=tmp_path/f'{n}.json'
        j.write_text(json.dumps({'shapes':[
            {'label':'wall','shape_type':'polygon','points':ellipse_points(32,32,24,24)},
            {'label':'lumen','shape_type':'polygon','points':ellipse_points(32,32,13,13)}
        ]}),encoding='utf-8')
        rows.append({'split':split,'annotation_image_path':p.name,
                     'annotation_json_path':j.name,'group_id':f'g{n}'})
    csv=tmp_path/'manifest.csv'
    pd.DataFrame(rows).to_csv(csv,index=False)
    return csv


def test_manifest_and_labelme(tmp_path):
    df=read_manifest(construct(tmp_path))
    ds=CarotidDataset(df[df.split=='train'],False,'none',image_size=64)
    row=ds[0]
    assert tuple(row['image'].shape)==(3,64,64)
    assert set(row['mask'].unique().tolist())=={0,1,2}
    assert not any(k in row for k in ('patient_id','image_path','video_path'))


def test_split_leakage_rejected(tmp_path):
    csv=construct(tmp_path)
    df=pd.read_csv(csv)
    df.loc[1,'group_id']='g0'
    df.to_csv(csv,index=False)
    with pytest.raises(ValueError,match='multiple splits'):
        read_manifest(csv)


def test_duplicate_image_rejected(tmp_path):
    csv=construct(tmp_path)
    df=pd.read_csv(csv)
    df.loc[1,'annotation_image_path']=df.loc[0,'annotation_image_path']
    df.to_csv(csv,index=False)
    with pytest.raises(ValueError,match='same image'):
        read_manifest(csv)

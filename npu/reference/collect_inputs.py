from pathlib import Path
import hashlib,json,tempfile,copy
import numpy as np
import qlib
from qlib.config import C
from qlib.utils import init_instance_by_config
from qlib.data.dataset.handler import DataHandlerLP
from qlib.workflow.record_temp import SignalRecord
from ruamel.yaml import YAML

def arrays(a):
 a=np.asarray(a);assert a.dtype.kind in 'fiub',a.dtype
 exact=hashlib.sha256();numeric=hashlib.sha256();nan=posinf=neginf=0
 for i in range(0,len(a),4096):
  raw=np.ascontiguousarray(a[i:i+4096],dtype=a.dtype.newbyteorder('<'));exact.update(raw.tobytes())
  b=np.array(a[i:i+4096],dtype='<f8',order='C',copy=True);mask=np.isnan(b);nan+=int(mask.sum());posinf+=int(np.isposinf(b).sum());neginf+=int(np.isneginf(b).sum());b[mask]=np.nan;b[b==0]=0.0;numeric.update(b.tobytes())
 return {'shape':list(a.shape),'dtype':str(a.dtype),'exact_sha256':exact.hexdigest(),'numeric_f64_sha256':numeric.hexdigest(),'nan':nan,'posinf':posinf,'neginf':neginf}
def frame(f):
 h=hashlib.sha256()
 for idx in f.index:
  values=idx if isinstance(idx,tuple) else (idx,)
  h.update((json.dumps([str(v) for v in values],ensure_ascii=False,separators=(',',':'))+'\n').encode())
 columns=[str(v) for v in f.columns]
 return {'rows':len(f),'index_names':list(f.index.names),'index_sha256':h.hexdigest(),'columns':len(columns),'column_sha256':hashlib.sha256(json.dumps(columns,ensure_ascii=False,separators=(',',':')).encode()).hexdigest(),'values':arrays(f.to_numpy())}
def collect(config_path):
 cfg=YAML(typ='safe').load(Path(config_path).read_text());exp=copy.deepcopy(C['exp_manager']);exp['kwargs']['uri']='file:'+tempfile.mkdtemp(prefix='lgb_input_mlruns_');qlib.init(**cfg['qlib_init'],exp_manager=exp)
 dataset=init_instance_by_config(cfg['task']['dataset']);model=init_instance_by_config(cfg['task']['model']);sets=model._prepare_data(dataset);summary={}
 for ds,name in sets:
  f=dataset.prepare(name,col_set=['feature','label'],data_key=DataHandlerLP.DK_L);assert np.array_equal(ds.data,f['feature'].to_numpy(),equal_nan=True)
  summary[name]={'features':frame(f['feature']),'labels':frame(f['label']),'lightgbm_label_f32':arrays(np.asarray(ds.label,dtype='<f4')),'weight':None if ds.weight is None else arrays(ds.weight)}
 summary['test']={'features':frame(dataset.prepare('test',col_set='feature',data_key=DataHandlerLP.DK_I)),'labels':frame(SignalRecord.generate_label(dataset))}
 return summary,sets,model.params,dataset
if __name__=='__main__':
 import sys
 s,*_=collect(sys.argv[1] if len(sys.argv)>1 else 'benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml');print('INPUT_SUMMARY_BEGIN');print(json.dumps(s,ensure_ascii=False,indent=2));print('INPUT_SUMMARY_END')

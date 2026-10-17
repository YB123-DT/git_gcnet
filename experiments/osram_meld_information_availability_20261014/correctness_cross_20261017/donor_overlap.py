import json,pathlib,sys,numpy as np
root=pathlib.Path('/data2/yb/remote_experiments/osram_meld_information_availability_20261014/full_68c698f_v2')
sys.path.insert(0,'/data2/yb/remote_experiments/osram_meld_information_availability_20261014/correctness_cross_20261017')
from correctness_cross import load_predictions,align_predictions,sha
rows=[]
manifest={}
for i in range(8):
 name=f'rate_0p{i}';folder=root/name/name
 with np.load(folder/'test_features.npz',allow_pickle=False) as cache:
  full_labels=cache['labels'];flat=cache['prediction'].argmax(1)
 for seed in [66,67,68]:
  donor_path=folder/'probes'/f'donors_seed{seed}.json'
  included=np.asarray(json.loads(donor_path.read_text())['test']['included'])
  base=folder/'probes'/f'seed_{seed}'
  data={k:load_predictions(base/k/'predictions.npz') for k in ['Local','Memory','MemoryDonor','LocalMemory','LocalMemoryDonor']}
  for k in data:manifest[str(base/k/'predictions.npz')]=sha(base/k/'predictions.npz')
  manifest[str(donor_path)]=sha(donor_path)
  y,lp,rp=align_predictions(full_labels,data['Local'],data['Memory'],included)
  for k in ['MemoryDonor','LocalMemory','LocalMemoryDonor']:
   align_predictions(full_labels,data['Local'],data[k],included)
  lc=lp==y;rc=rp==y;dc=data['MemoryDonor']['predictions']==y
  jc=data['LocalMemory']['predictions']==y;jdc=data['LocalMemoryDonor']['predictions']==y
  fc=flat[included]==y
  onlyreal=(~lc)&rc&(~dc);both=(~lc)&rc&dc
  row=dict(rate=f'{i/10:.1f}',probe_seed=seed,N=len(y),local_wrong_N=int((~lc).sum()),
   memory_correct_on_local_wrong=int(((~lc)&rc).sum()),donor_correct_on_local_wrong=int(((~lc)&dc).sum()),
   both_memory_donor_correct_on_local_wrong=int(both.sum()),
   only_real_correct_on_local_wrong=int(onlyreal.sum()),
   only_donor_correct_on_local_wrong=int(((~lc)&dc&(~rc)).sum()),
   real_only_retained_by_LocalMemory=int((onlyreal&jc).sum()),
   real_only_missed_by_LocalMemory=int((onlyreal&(~jc)).sum()),
   real_only_retained_by_OriginalFlat=int((onlyreal&fc).sum()),
   real_only_missed_by_OriginalFlat=int((onlyreal&(~fc)).sum()),
   joint_real_correct_on_local_wrong=int(((~lc)&jc).sum()),
   joint_donor_correct_on_local_wrong=int(((~lc)&jdc).sum()))
  row['all_memory_real_minus_donor_ACC_pp']=100*float(rc.mean()-dc.mean())
  row['local_wrong_memory_real_minus_donor_ACC_pp']=100*(row['memory_correct_on_local_wrong']-row['donor_correct_on_local_wrong'])/row['local_wrong_N']
  row['per_class']=[]
  for c in range(7):
   m=(y==c)&(~lc)
   row['per_class'].append(dict(class_id=c,N=int(m.sum()),real_correct=int((m&rc).sum()),donor_correct=int((m&dc).sum())))
  assert row['memory_correct_on_local_wrong']==row['only_real_correct_on_local_wrong']+row['both_memory_donor_correct_on_local_wrong']
  assert row['donor_correct_on_local_wrong']==row['only_donor_correct_on_local_wrong']+row['both_memory_donor_correct_on_local_wrong']
  assert row['only_real_correct_on_local_wrong']==row['real_only_retained_by_LocalMemory']+row['real_only_missed_by_LocalMemory']
  assert row['only_real_correct_on_local_wrong']==row['real_only_retained_by_OriginalFlat']+row['real_only_missed_by_OriginalFlat']
  rows.append(row)
print(json.dumps(dict(label='INTERNAL DIAGNOSTIC ONLY',missing_protocol='gcnet-legacy-fixed-v1',backbone_seed=66,probe_seeds=[66,67,68],training=False,inference=False,input_root=str(root),input_sha256=manifest,rows=rows)))

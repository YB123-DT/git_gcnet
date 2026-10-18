# Read-only audit commands

No model forward/backward, dataset replay or training. CPU tensor deserialization only. Source artifacts are hashed, opened read-only and rehashed after summarization. Commands print JSON to stdout and do not modify source runs.

## Training logs and retained parameters

```bash
ssh biggpu 'CUDA_VISIBLE_DEVICES="" /data2/yb/conda_envs/cmerc-repro/bin/python -' <<'AUDIT_READ_ONLY'
import hashlib,json,math,statistics,time
from pathlib import Path
import torch
torch.set_num_threads(2)
root=Path("/home/qinjinghui/remote_experiments/osram_meld_related_history_20261018/runs_v1")
prefix="osram.related_history."
source_hashes={}
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
 return h.hexdigest()
def readj(p):
 source_hashes[str(p)]=sha(p)
 return json.loads(p.read_text())
def load(p):
 source_hashes[str(p)]=sha(p)
 return torch.load(p,map_location="cpu",weights_only=False)
def group_norm(state,start):
 return math.sqrt(sum(float(v.double().square().sum()) for k,v in state.items() if k.startswith(start)))
def groups(state,reference=None):
 result={}
 for g in ("query.","key.","value.","speaker_bias.","distance_bias.","availability_bias.","fusion.","output.","relation."):
  if not any(k.startswith(g) for k in state):continue
  row={"norm":group_norm(state,g)}
  if reference is not None:
   delta={k:v-reference[k] for k,v in state.items() if k.startswith(g)}
   row["delta_norm"]=group_norm(delta,g)
   n=group_norm(reference,g)
   row["delta_over_reference_norm"]=row["delta_norm"]/n if n else None
  result[g.rstrip(".")]=row
 return result
result={"status":"complete","training":False,"new_inference":False,"device":"CPU only","source":str(root),"models":{},"created_unix":time.time(),"limits":["No task-validation or per-epoch heldout-relation curves saved.","Historical individual Q/K/V gradient magnitudes are not saved.","Gradient records are branch-level pre-global-clip norms and nonzero tensor counts.","B2 logged classification_loss includes CE+.05 auxiliary BCE.","Checkpoint parameter differences include normal optimizer weight decay.","Task test masks can differ by epoch; no fixed-mask epoch evaluation added.","This audit concerns wd2e-4 B1/B2, not current wd1e-5 baseline."]}
for kind in ("B1","B2"):
 p=root/kind
 summary=readj(p/"SUMMARY.json");history=readj(p/"history.json");last=load(p/"last_training.pt")
 assert summary["status"]=="complete" and len(history)==50 and last["next_epoch"]==50
 assert [r["epoch"] for r in history]==list(range(1,51)) and last["history"]==history and last["identity"]==summary["identity"]
 branch={k[len(prefix):]:v.detach().clone() for k,v in last["model"].items() if k.startswith(prefix)}
 tensor_count=len(branch);epochs=[]
 assert tensor_count==(13 if kind=="B1" else 17)
 for row in history:
  e=row["epoch"];d=readj(p/"diagnostics"/f"epoch_{e:03d}.json");steps=d["steps"];aux=d["auxiliary_relation_losses"]
  assert d["epoch"]==e and len(steps)==row["train"]["optimizer_steps"]==33
  assert all(math.isfinite(s["gradient_norm"]) and s["gradient_norm"]>0 and s["gradient_tensors"]==tensor_count for s in steps)
  assert d["optimizer"]=={"rates":[1e-5]*3,"weight_decay":[2e-4]*3}
  entry={"epoch":e,"logged_train_loss":row["train"]["classification_loss"],"train_ACC":row["train"]["accuracy"]*100,"task_test_CE_mean":statistics.mean(t["loss"] for t in row["test_oracle"].values()),"test_ACC_mean":row["test_oracle_mean_selection_score"]*100,"test_WF1_mean":row["test_oracle_mean_weighted_f1"]*100,"test_per_rate":{r:{"ACC":v["accuracy"]*100,"CE":v["loss"],"WF1":v["weighted_f1"]*100} for r,v in row["test_oracle"].items()},"output_weight_norm":d["branch_output_weight_norm"],"branch_preclip_gradient_mean":statistics.mean(s["gradient_norm"] for s in steps),"branch_preclip_gradient_min":min(s["gradient_norm"] for s in steps),"branch_preclip_gradient_max":max(s["gradient_norm"] for s in steps),"gradient_tensor_count":tensor_count,"all_tensors_nonzero_steps":sum(s["nonzero_tensors"]==tensor_count for s in steps),"optimizer_steps":len(steps),"first_step_nonzero_tensors":steps[0]["nonzero_tensors"],"aux_BCE_batch_mean":statistics.mean(aux) if aux else None}
  assert (len(aux)==33 if kind=="B2" else len(aux)==0)
  epochs.append(entry)
 snapshots=[]
 first=None;previous=None
 for path in sorted((p/"training_versions").glob("*_model.pt")):
  e=int(path.name.split("_")[1]);state=load(path)
  state={k[len(prefix):]:v.detach().clone() for k,v in state.items() if k.startswith(prefix)}
  if first is None:first=state
  snapshots.append({"epoch":e,"file":str(path),"sha256":source_hashes[str(path)],"groups":groups(state,previous),"output_weight_norm":float(state["output.weight"].norm())})
  assert abs(snapshots[-1]["output_weight_norm"]-epochs[e-1]["output_weight_norm"])<1e-5
  previous=state
 snapshots.append({"epoch":50,"file":str(p/"last_training.pt"),"sha256":source_hashes[str(p/"last_training.pt")],"groups":groups(branch,previous),"output_weight_norm":float(branch["output.weight"].norm())})
 assert abs(snapshots[-1]["output_weight_norm"]-epochs[-1]["output_weight_norm"])<1e-5
 windows={}
 for lo,hi in ((1,5),(6,10),(11,20),(21,30),(31,40),(41,50)):
  segment=epochs[lo-1:hi];keys=("logged_train_loss","train_ACC","task_test_CE_mean","test_ACC_mean","test_WF1_mean","branch_preclip_gradient_mean")
  w={k:statistics.mean(v[k] for v in segment) for k in keys}
  if kind=="B2":w["aux_BCE_batch_mean"]=statistics.mean(v["aux_BCE_batch_mean"] for v in segment)
  windows[f"{lo}-{hi}"]=w
 result["models"][kind]={"config":summary["config"],"branch":summary["branch"],"identity":summary["identity"],"environment":summary["environment"],"optimizer_steps":sum(v["optimizer_steps"] for v in epochs),"all_tensors_nonzero_steps":sum(v["all_tensors_nonzero_steps"] for v in epochs),"best_epochs":{r:v["epoch"] for r,v in summary["testbest"]["rates"].items()},"best_test_metrics":summary["testbest"],"epoch_records":epochs,"window_means":windows,"parameter_snapshots":snapshots,"parameter_update_first_retained_to_epoch50":groups(branch,first),"first_retained_epoch":snapshots[0]["epoch"],"validation_recorded":any("val" in k.lower() for row in history for k in row)}
 del last,branch,first,previous
control=Path("/home/qinjinghui/remote_experiments/osram_meld_optimizer_screen_20261018/runs_v1/O6")
h=readj(control/"history.json")
result["matched_original_Flat_wd2e-4"]=[{"epoch":r["epoch"],"logged_train_loss":r["train"]["classification_loss"],"test_CE_mean":statistics.mean(v["loss"] for v in r["test_oracle"].values()),"test_ACC_mean":r["test_oracle_mean_selection_score"]*100} for r in h]
assert all(sha(Path(p))==digest for p,digest in source_hashes.items())
result["source_inputs_unchanged"]=True;result["source_hashes"]=source_hashes
for kind,model in result["models"].items():
 model["identity"]={k:v for k,v in model["identity"].items() if k!="extension"}
 for row in model["epoch_records"]:
  row.pop("test_per_rate")
  row["online_train_CE_approx"]=row["logged_train_loss"]-(.05*row["aux_BCE_batch_mean"] if row["aux_BCE_batch_mean"] is not None else 0.)
 for key,w in model["window_means"].items():
  w["online_train_CE_approx"]=w["logged_train_loss"]-(.05*w["aux_BCE_batch_mean"] if "aux_BCE_batch_mean" in w else 0.)
print("AUDIT_JSON_BEGIN");print(json.dumps(result,ensure_ascii=False,separators=(",",":")))
AUDIT_READ_ONLY
```

## Parameter geometry (not task semantics)

```bash
ssh biggpu 'CUDA_VISIBLE_DEVICES="" /data2/yb/conda_envs/cmerc-repro/bin/python -' <<'AUDIT_GEOMETRY'
import json,math
from pathlib import Path
import torch
torch.set_num_threads(2)
root=Path("/home/qinjinghui/remote_experiments/osram_meld_related_history_20261018/runs_v1")
result={}
for kind in ("B1","B2"):
 p=root/kind
 first=torch.load(next((p/"training_versions").glob("epoch_000008_*_model.pt")),map_location="cpu",weights_only=False)
 last=torch.load(p/"last_training.pt",map_location="cpu",weights_only=False)["model"]
 rows={}
 for group in ("query","key","value","relation"):
  keys=[k for k in first if k.startswith("osram.related_history."+group+".")]
  if not keys:continue
  a=torch.cat([first[k].double().flatten() for k in sorted(keys)])
  b=torch.cat([last[k].double().flatten() for k in sorted(keys)])
  scale=float(a.dot(b)/a.dot(a));ortho=b-scale*a
  rows[group]={"epoch8_norm":float(a.norm()),"epoch50_norm":float(b.norm()),"cosine_epoch8_epoch50":float(a.dot(b)/a.norm()/b.norm()),"best_scalar_fit":scale,"orthogonal_change_over_epoch8_norm":float(ortho.norm()/a.norm()),"norm_ratio":float(b.norm()/a.norm())}
 result[kind]=rows
print(json.dumps(result))
AUDIT_GEOMETRY
```

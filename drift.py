"""Flag shifts in feature means relative to training standard deviation."""
import csv,json,sys,joblib,numpy as np
artifact=joblib.load(sys.argv[1])
with open(sys.argv[2],newline='') as f: rows=list(csv.DictReader(f))
if not rows: raise ValueError('monitoring data is empty')
x=np.array([[float(row[k]) for k in artifact['features']] for row in rows]); mean=x.mean(axis=0); ref=np.array(artifact['reference_mean']); sd=np.array(artifact['reference_std']); z=np.abs(mean-ref)/np.maximum(sd,1e-9)
print(json.dumps({'rows':len(rows),'features':[{'name':name,'standardized_mean_shift':float(score),'flagged':bool(score>0.5)} for name,score in zip(artifact['features'],z)]},indent=2))

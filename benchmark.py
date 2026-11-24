import argparse,json,time,urllib.request,statistics
p=argparse.ArgumentParser();p.add_argument('--url',default='http://127.0.0.1:8000/predict');p.add_argument('--features',required=True,help='comma-separated numeric values');p.add_argument('--n',type=int,default=100);a=p.parse_args()
body=json.dumps({'features':[float(x) for x in a.features.split(',')]}).encode(); timings=[]
for _ in range(a.n):
 start=time.perf_counter(); req=urllib.request.Request(a.url,data=body,headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req) as response: response.read()
 timings.append((time.perf_counter()-start)*1000)
timings.sort(); print(json.dumps({'requests':len(timings),'p50_ms':statistics.median(timings),'p95_ms':timings[min(len(timings)-1,int(.95*len(timings)))], 'max_ms':max(timings)}))

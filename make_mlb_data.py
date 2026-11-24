"""Create chronological MLB pregame training data from prior player games."""
import argparse,csv,datetime as dt
p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('output');a=p.parse_args()
with open(a.source,newline='') as f: rows=sorted(csv.DictReader(f),key=lambda row:row['date'])
previous={};out=[]
for row in rows:
 player=row['player_id'];day=dt.date.fromisoformat(row['date']);prior=previous.get(player)
 if prior and (day-prior[0]).days>0: out.append({'previous_batting_average':prior[1],'days_since_appearance':(day-prior[0]).days,'label':int(float(row['performance'])>0)})
 previous[player]=(day,float(row['performance']))
with open(a.output,'w',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=['previous_batting_average','days_since_appearance','label']);writer.writeheader();writer.writerows(out)
print(f'wrote {len(out)} chronologically ordered rows')

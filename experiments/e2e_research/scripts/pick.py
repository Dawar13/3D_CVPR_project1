import sys,os,csv,collections
d=sys.argv[1]
rows=list(csv.DictReader(open(os.path.join(d,'cdn_raw_metadata.csv'))))
v=[r for r in rows if r['fold']=='Validation' and r['is_in_upsampling']=='True' and r['has_laser_scanner_point_clouds']=='True']
byv=collections.OrderedDict()
for r in v: byv.setdefault(r['visit_id'],[]).append(r['video_id'])
print('visits',len(byv),'videos',len(v))
print(collections.Counter(r['is_in_threedod'] for r in v))
# deterministic pick: first video of every 6th visit sorted
vis=sorted(byv)
pick=[byv[x][0] for x in vis[::max(1,len(vis)//15)]][:15]
print(' '.join(pick))
print(' '.join(byv['421337']))

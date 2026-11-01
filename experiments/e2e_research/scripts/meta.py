import sys,os,csv,collections
d=sys.argv[1]
rows=list(csv.DictReader(open(os.path.join(d,'cdn_raw_metadata.csv'))))
c=collections.Counter((r['fold'],r['is_in_upsampling'],r['has_laser_scanner_point_clouds']) for r in rows)
for k,v in sorted(c.items()): print(k,v)
v=[r for r in rows if r['fold']=='Validation']
vu=[r for r in v if r['is_in_upsampling']=='True']
print('val',len(v),'val upsampling',len(vu),'unique visits',len(set(r['visit_id'] for r in vu)))
print(collections.Counter(r['sky_direction'] for r in vu))
print('val upsampling & laser', sum(r['has_laser_scanner_point_clouds']=='True' for r in vu))
for r in vu[:15]: print(r)
tv=set(r['visit_id'] for r in rows if r['fold']=='Training' and r['visit_id']!='NA'); vv=set(r['visit_id'] for r in rows if r['fold']=='Validation' and r['visit_id']!='NA')
print('visit overlap', len(tv&vv))

import sys,os,csv,collections
d=sys.argv[1]
meta={r['video_id']:r for r in csv.DictReader(open(os.path.join(d,'cdn_raw_metadata.csv')))}
for name in ['test_set.csv','train_val_set.csv']:
    rows=list(csv.DictReader(open(os.path.join(d,'scenefun3d','scenefun3d_benchmark_file_lists_'+name if False else 'benchmark_file_lists_'+name))))
    print(name, 'cols',list(rows[0].keys()),'rows',len(rows),'visits',len(set(r['visit_id'] for r in rows)))
    c=collections.Counter((meta[r['video_id']]['fold'],meta[r['video_id']]['is_in_upsampling'],meta[r['video_id']]['has_laser_scanner_point_clouds']) for r in rows)
    print('  (fold,is_in_upsampling,has_laser)',dict(c))
    if name=='test_set.csv':
        print('  first rows',[ (r['visit_id'],r['video_id']) for r in rows[:6]])
for t in ['val_scenes.txt','test_scenes.txt','train_scenes.txt']:
    ids=open(os.path.join(d,'scenefun3d','benchmark_file_lists_'+t)).read().split()
    folds=collections.Counter()
    vm=collections.defaultdict(set)
    for v in meta.values(): vm[v['visit_id']].add(v['fold'])
    for i in ids: folds[tuple(sorted(vm.get(i,{'?'})))]+=1
    print(t,len(ids),dict(folds))

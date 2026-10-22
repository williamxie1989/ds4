import glob, hashlib, os, re, sys
OUT="/tmp/dspark-ab"
rows=[]
for err in sorted(glob.glob(f"{OUT}/*.err")):
    name=os.path.basename(err)[:-4]
    txt=open(err, errors="replace").read()
    m=re.search(r"prefill: ([0-9.]+) t/s, generation: ([0-9.]+) t/s", txt)
    pre,gen=(m.group(1),m.group(2)) if m else ("-","-")
    st=re.search(r"DSpark stats (.*)", txt)
    stats={}
    if st:
        for k,v in re.findall(r"(\w+)=([^\s]+)", st.group(1)):
            stats[k]=v
    out=open(f"{OUT}/{name}.out","rb").read()
    rows.append((name,pre,gen,hashlib.sha256(out).hexdigest()[:12],stats))
print(f"{'run':10s} {'prefill':>8s} {'gen':>7s} {'out#':12s}  dspark")
for n,p,g,h,s in rows:
    extra=""
    if s:
        extra=(f"cycles={s.get('cycles')} prop={s.get('proposed')} acc={s.get('accepted_draft')} "
               f"rate={s.get('accept_rate')} avg={s.get('avg_accept')} "
               f"no_draft={s.get('no_draft')} skips={s.get('scheduler_skips')} "
               f"tgt={s.get('target')} saved={s.get('saved')} net={s.get('net_saved')} "
               f"prop_ms={s.get('propose')} ver_ms={s.get('verify')}")
    print(f"{n:10s} {p:>8s} {g:>7s} {h:12s}  {extra}")

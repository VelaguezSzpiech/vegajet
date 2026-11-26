#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""CPU-only Ornith TP3 ring-verify dispatch/traffic census; no model or device I/O.

Useful load bytes before L2, rather than measured HBM transactions. The baseline
is resident Q3 GU, sub32 Down, Q8 head, ordinary ring GDN and unsplit Q8 flash.
Kernel variants are identified by logical dispatch; --spec-gdn adds norm dispatch.
"""
import argparse
import csv
import sys


def census(m=12, ctx=128, spec_gdn=False, pair_routes=None):
    rows = []
    d = 3072
    def add(g, layer, kernel, bucket, nbytes, note=""):
        rows.append(dict(rank=g, layer=layer, kernel=kernel, bucket=bucket,
                         bytes=int(nbytes), note=note))
    q4 = lambda r, c: r*c*20//32  # Q4_A8 packed weights + float2 d/m
    tile = 8 if m <= 8 else 12 if m <= 12 else 16
    for g, (nk, nv, nq, nkv, ff, vocab) in enumerate(zip(
            (6,5,5), (24,20,20), (12,10,10), (1,2,1), (384,320,320),
            (91520,78336,78528))):
        add(g,-1,"ornith_embed_q8_kernel","input",m*d*34//32)
        add(g,-1,"ornith_rms_quant_kernel","input",m*d*17)
        for layer in range(48):
            if layer % 4 != 3:
                # GDN in = q,k,v,z,alpha,beta; out uses nv*128 values.
                cdim = (2*nk+nv)*128
                add(g,layer,f"gemm_q4a8_m{tile}","dense",q4(2*nk*128+2*nv*128+2*nv,d))
                add(g,layer,"gdn_conv_ring","conv",(7+2*m+3*min(m,16))*cdim*4+(2*nv+4*m*nv)*4,"weights + conv ring snapshots + activations, approximate")
                state=nv*128*128*4
                if spec_gdn:
                    add(g,layer,"spec_gdn_verify","recur",2*state+m*nv*128*8,"state loaded once; slot0/final stores, approximate")
                    add(g,layer,"spec_gdn_norm","recur",m*nv*128*9)
                else:
                    add(g,layer,"gdn_recur_ring","recur",(m+1)*state,"one full state read + M full state-ring writes; ignores Q/K/V/Z")
                add(g,layer,f"gemm_q4a8_m{tile}","dense",q4(d,nv*128))
            else:
                add(g,layer,f"gemm_q4a8_m{tile}","dense",q4((2*nq+2*nkv)*256,d))
                add(g,layer,"attn_prep","prep",m*((2*nq+2*nkv)*256*4+2*nkv*272),"approx, Q/K norms + rope + KV write")
                add(g,layer,"attn_flash_q8","flash",m*nq*(ctx+(m+1)/2)*272*2,"per-query-head requested K/V; cache shares KV heads")
                add(g,layer,f"gemm_q4a8_m{tile}","dense",q4(d,nq*256))
            add(g,layer,"p2p_ar_n_fp16","AR",m*d*2*4,"two peer sends + two peer receives; prepush moves sends into GEMM")
            add(g,layer,"moe_router_logits","router",((m+7)//8)*257*d*2,"8-row tile, approx weights")
            add(g,layer,f"gemm_q4a8_m{tile}","dense",q4(2*ff,d))
            add(g,layer,"moe_top8","top8",m*(256*4+8*8))
            add(g,layer,"swiglu_q8","shared",m*ff*13)
            routes=m*8 if pair_routes is None else pair_routes
            add(g,layer,"vj_moe_gu_q3_kernel" if pair_routes is None else "vj_verify_gu_q3_pair_kernel",
                "GU",routes*ff*(2304+240),"96 routes at M12; Q3 scales include 192 B subscales + 48 B fp16 pairs")
            add(g,layer,f"gemm_q4a8_m{tile}","dense",q4(d,ff))
            add(g,layer,"moe_down_dot8_sub32","Down",m*8*d*(ff//32)*20,"one read for each routed expert per row")
            add(g,layer,"p2p_ar_n_fp16","AR",m*d*2*4)
        add(g,48,"vocab_argmax_b12" if 7 <= m <= 12 else "vocab_argmax_b16" if m>=13 else "vocab_argmax_b6",
            "LM",vocab*d*34//32*(1 if m>=7 else (m+5)//6),"approx vocabulary shard; read exact vocab rows from model for precise total")
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rows',type=int,default=12,choices=range(2,17))
    p.add_argument('--ctx',type=int,default=128)
    p.add_argument('--spec-gdn',action='store_true')
    p.add_argument('--pair-routes',type=int,help='sum ceil(expert_count/2) from VERIFY-UNION; applies GU only')
    p.add_argument('--dispatches',action='store_true',help='CSV of every dispatch, all 48 layers and all ranks')
    p.add_argument('--aggregate-gbs',type=float,default=1100.)
    a=p.parse_args()
    rows=census(a.rows,a.ctx,a.spec_gdn,a.pair_routes)
    if a.dispatches:
        w=csv.DictWriter(sys.stdout,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows);return
    print(f'M={a.rows} TP=3 ctx={a.ctx}; requested useful bytes BEFORE cache; no GPU access')
    for bucket in dict.fromkeys(r['bucket'] for r in rows):
        counts=[sum(r['rank']==g and r['bucket']==bucket for r in rows) for g in range(3)]
        bs=[sum(r['bytes'] for r in rows if r['rank']==g and r['bucket']==bucket) for g in range(3)]
        print(f'{bucket:8} calls/rank={counts} GB/rank='+' / '.join(f'{b/1e9:.4f}' for b in bs)+f' aggregate floor={sum(bs)/a.aggregate_gbs/1e6:.3f} ms')
    print(f'dispatches={len(rows)//3}/rank, {len(rows)} total; expected independent expert union={256*(1-(248/256)**a.rows):.2f}')
    print('Small state/activation, flash and router figures are traffic estimates, not measured HBM.')

if __name__=='__main__':
    main()

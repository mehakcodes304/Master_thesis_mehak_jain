#!/usr/bin/env python3
"""
plot_contact_heatmap.py
───────────────────────
Single script for all contact matrix visualisations.
Takes the sparse TSV + stems2 CSV and produces three outputs:

  OUTPUT 1 — Full annotated heatmap (PNG)
  OUTPUT 2 — Per-stem zoomed PNGs
  OUTPUT 3 — Interactive HTML (Plotly, opens in any browser)

The HTML file is written to --out/<rna>_interactive.html
Open it with:  xdg-open <out_dir>/<rna>_interactive.html   (Linux)
               open <out_dir>/<rna>_interactive.html        (Mac)
               or double-click it in your file manager

Usage
─────
conda activate analysis

  pip install plotly matplotlib pandas numpy

  python plot_contact_heatmap.py \
      --sparse  .../rRNA_5S_URS..._contact_matrix_sparse.tsv \
      --stems   .../stems2_ecoli_rRNA.csv \
      --rna     rRNA_5S \
      --length  120 \
      --out     ./plots

python plot_contact_heatmap.py   --sparse  /mnt/ceph/mjain/software/envs/bio/contact_matrices/ecoli_rRNA/rRNA_23S_URS000077EFF8_511145_contact_matrix_sparse.tsv   --stems   /mnt/ceph/mjain/software/envs/bio/stems/ecoli_rRNA/stems2_ecoli_rRNA.csv   --chimeric /mnt/ceph/mjain/software/envs/bio/chimeric_reads/ecoli_rRNA/ecoli_rRNA_chimeric_reads.csv   --rna     rRNA_23S   --length  2932   --out     /mnt/ceph/mjain/software/envs/bio/contact_matrices/ecoli_rRNA/plots  --no_zoom   --min_support 5

python plot_contact_heatmap.py   --sparse  /mnt/ceph/mjain/software/envs/bio/contact_matrices/ecoli_rRNA/rRNA_5S_URS0000049E57_511145_contact_matrix_sparse.tsv   --stems   /mnt/ceph/mjain/software/envs/bio/stems/ecoli_rRNA/stems2_ecoli_rRNA.csv   --chimeric /mnt/ceph/mjain/software/envs/bio/chimeric_reads/ecoli_rRNA/ecoli_rRNA_chimeric_reads.csv   --rna     rRNA_5S   --length  120   --out     /mnt/ceph/mjain/software/envs/bio/contact_matrices/ecoli_rRNA/plots

python plot_contact_heatmap.py   --sparse  /mnt/ceph/mjain/software/envs/bio/contact_matrices/ecoli_rRNA/rRNA_16S_URS0000801AEA_511145_contact_matrix_sparse.tsv   --stems   /mnt/ceph/mjain/software/envs/bio/stems/ecoli_rRNA/stems2_ecoli_rRNA.csv   --chimeric /mnt/ceph/mjain/software/envs/bio/chimeric_reads/ecoli_rRNA/ecoli_rRNA_chimeric_reads.csv   --rna     rRNA_16S   --length  1556   --out     /mnt/ceph/mjain/software/envs/bio/contact_matrices/ecoli_rRNA/plots  --no_zoom   --min_support 5

# Each tRNA is its own RNA — run one per tRNA type
# First check what RNA names are in your stems CSV:
python3 -c "import pandas as pd; df=pd.read_csv('/mnt/ceph/mjain/software/envs/bio/stems2_ecoli_tRNA.csv'); print(df['name'].unique())"

python plot_contact_heatmap.py \
  --sparse  /mnt/ceph/mjain/software/envs/bio/contact_matrices/ecoli_tRNA/<rna_name>_URS*_contact_matrix_sparse.tsv \
  --stems   /mnt/ceph/mjain/software/envs/bio/stems/ecoli_tRNA/stems2_ecoli_tRNA.csv \
  --rna     <rna_name> \
  --length  <length> \
  --out     /mnt/ceph/mjain/software/envs/bio/contact_matrices/ecoli_tRNA/plots \
  --no_zoom

  # 5S rRNA
python plot_contact_heatmap.py \
  --sparse  /mnt/ceph/mjain/software/envs/bio/contact_matrices/human_rRNA/rRNA_5S_1_8_URS*_contact_matrix_sparse.tsv \
  --stems   /mnt/ceph/mjain/software/envs/bio/stems/human_rRNA/stems2_human_rRNA.csv \
  --rna     rRNA_5S_1_8 \
  --length  120 \
  --out     /mnt/ceph/mjain/software/envs/bio/contact_matrices/human_rRNA/plots/5S_rRNA 

# 5.8S rRNA
python plot_contact_heatmap.py \
  --sparse  /mnt/ceph/mjain/software/envs/bio/contact_matrices/human_rRNA/rRNA_5_8S_N_1_3_URS*_contact_matrix_sparse.tsv \
  --stems   /mnt/ceph/mjain/software/envs/bio/stems/human_rRNA/stems2_human_rRNA.csv \
  --rna     rRNA_5_8S_N_1_3 \
  --length  157 \
  --out     /mnt/ceph/mjain/software/envs/bio/contact_matrices/human_rRNA/plots/5_8S_rRNA 

# 18S rRNA
python plot_contact_heatmap.py \
  --sparse  /mnt/ceph/mjain/software/envs/bio/contact_matrices/human_rRNA/rRNA_18S_N_2_4_URS*_contact_matrix_sparse.tsv \
  --stems   /mnt/ceph/mjain/software/envs/bio/stems/human_rRNA/stems2_human_rRNA.csv \
  --rna     rRNA_18S_N_2_4 \
  --length  1869 \
  --out     /mnt/ceph/mjain/software/envs/bio/contact_matrices/human_rRNA/plots/18S_rRNA \
  --min_support 5

# 28S rRNA (large — skip zooms)
python plot_contact_heatmap.py \
  --sparse  /mnt/ceph/mjain/software/envs/bio/contact_matrices/human_rRNA/rRNA_28S_N5_URS*_contact_matrix_sparse.tsv \
  --stems   /mnt/ceph/mjain/software/envs/bio/stems/human_rRNA/stems2_human_rRNA.csv \
  --rna     rRNA_28S_N5 \
  --length  5070 \
  --out     /mnt/ceph/mjain/software/envs/bio/contact_matrices/human_rRNA/plots/28S_rRNA \
  --no_zoom \
  --min_support 5


  #to get length from bam headers
  samtools view -H /mnt/ceph/mjain/software/envs/bio/rnanue_run/human_rRNA/output/03_detect/treatment/condition1/condition1_splits.bam | grep "^@SQ"

  # Skip specific outputs
  python plot_contact_heatmap.py ... --no_zoom
  python plot_contact_heatmap.py ... --no_html
  python plot_contact_heatmap.py ... --no_full
"""

import argparse, math, re, os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import numpy as np, pandas as pd
import plotly.graph_objects as go

COLORS = [
    "#E74C3C","#2980B9","#27AE60","#F39C12","#8E44AD",
    "#16A085","#C0392B","#1A5276","#D35400","#117A65",
    "#884EA0","#1E8449","#B7950B","#154360","#6E2F1A",
]
CMAP = mcolors.LinearSegmentedColormap.from_list(
    "contact",["white","#AED6F1","#2874A6","#1A5276","#922B21","#E74C3C"])

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sparse",      required=True)
    p.add_argument("--stems",       required=True)
    p.add_argument("--rna",         required=True,
                   help="Must match 'name' column in stems CSV, e.g. rRNA_16S")
    p.add_argument("--length",      required=True, type=int,
                   help="Reference sequence length in nt")
    p.add_argument("--out",         required=True)
    p.add_argument("--pad",         type=int, default=15)
    p.add_argument("--min_support", type=int, default=1)
    p.add_argument("--chimeric", default=None,
                   help="Chimeric reads CSV to exclude zero-read stems from "
                        "annotation strips (optional but recommended)")
    p.add_argument("--no_full",  action="store_true")
    p.add_argument("--no_zoom",  action="store_true")
    p.add_argument("--no_html",  action="store_true")
    return p.parse_args()

def parse_arms(val):
    m = re.findall(r"\((\d+)-(\d+)\)", str(val))
    if len(m) >= 2: return (int(m[0][0]),int(m[0][1])),(int(m[1][0]),int(m[1][1]))
    return None

def icolor(i): return COLORS[(i-1) % len(COLORS)]

def cbar_ticks(maxv):
    raw = [v for v in [0,1,5,10,25,50,100,200,maxv] if v <= maxv]
    return [np.log1p(v) for v in raw], [str(v) for v in raw]

def build_matrix(df, n):
    mat = np.zeros((n,n), dtype=np.int32)
    for _, row in df.iterrows():
        i,j,c = int(row["pos_arm1"])-1, int(row["pos_arm2"])-1, int(row["count"])
        if 0<=i<n and 0<=j<n:
            mat[i][j]=c
    return mat

# ── Output 1: full annotated heatmap ─────────────────────────────────────────

def draw_arm1_strip(ax, stems_df, n):
    # Bars only — no tick labels or ylabel (both now on ax_main)
    ax.set_xlim(0,1); ax.set_ylim(0.5,n+0.5); ax.set_xticks([])
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(left=False, labelleft=False, right=False, labelright=False)
    handles=[]
    for _,row in stems_df.iterrows():
        arms=parse_arms(row["sub_interaction_stem"])
        if arms is None: continue
        a1,_=arms; i=int(row["interaction_number"]); c=icolor(i)
        h=a1[1]-a1[0]+1; cy=(a1[0]+a1[1])/2
        ax.barh(y=cy,width=0.80,height=h,left=0.10,color=c,alpha=0.90,
                linewidth=0.5,edgecolor="white")
        if h>=3: ax.text(0.50,cy,f"int{i}",ha="center",va="center",
                         fontsize=5.5,color="white",fontweight="bold")
        handles.append(mpatches.Patch(color=c,alpha=0.90,
                       label=f"int{i}: {row['sub_interaction_stem']}"))
    return handles

def draw_arm2_strip(ax, stems_df, n, large=False):
    # Labels on the TOP edge (inner, touching the plot); bars sit below
    ax.set_ylim(0,1); ax.set_xlim(0.5,n+0.5); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    x_rot = 45 if large else 0
    ax.tick_params(bottom=False, labelbottom=False,
                   top=True, labeltop=True, labelsize=7,
                   length=0, pad=3, rotation=x_rot)
    ax.set_xlabel("Arm 2 position (1-based)", fontsize=9, labelpad=6)
    for _,row in stems_df.iterrows():
        arms=parse_arms(row["sub_interaction_stem"])
        if arms is None: continue
        _,a2=arms; i=int(row["interaction_number"]); c=icolor(i)
        w=a2[1]-a2[0]+1; cx=(a2[0]+a2[1])/2
        ax.bar(x=cx,height=0.80,width=w,bottom=0.10,color=c,alpha=0.90,
               linewidth=0.5,edgecolor="white")
        if w>=3: ax.text(cx,0.50,f"int{i}",ha="center",va="center",
                         fontsize=5.5,color="white",fontweight="bold")

def plot_full(mat, stems_df, rna, out_dir):
    n=mat.shape[0]; logmat=np.log1p(mat.astype(float))
    n_int=len(stems_df)
    ncol=max(1,min(4,n_int//10+1)); n_rows=-(-n_int//ncol)
    # Each legend row needs ~0.30" at fontsize 6.5, plus 1.2" for x-axis strip/label/gap
    legend_h_in  = n_rows * 0.30 + 1.2
    plot_h_in    = 11.0
    total_h_in   = plot_h_in + legend_h_in
    fig=plt.figure(figsize=(11, total_h_in))
    gs=gridspec.GridSpec(2,3,width_ratios=[0.06,1,0.04],
                         height_ratios=[1,0.06],hspace=0.08,wspace=0.09)
    ax_main=fig.add_subplot(gs[0,1])
    ax_a1  =fig.add_subplot(gs[0,0],sharey=ax_main)
    ax_a2  =fig.add_subplot(gs[1,1],sharex=ax_main)
    ax_cb  =fig.add_subplot(gs[0,2])
    fig.add_subplot(gs[1,0]).axis("off")

    im=ax_main.imshow(logmat,cmap=CMAP,origin="lower",aspect="auto",
                      interpolation="nearest",
                      extent=[0.5,n+0.5,0.5,n+0.5])
    step=5 if n<=200 else 25 if n<=500 else 100
    ticks=list(range(1,n+1,step))
    ax_main.set_xticks(ticks)
    ax_main.set_yticks(ticks)
    # Y-axis numbers on ax_main's left edge — they appear in the wspace gap
    # between the bar strip (ax_a1) and the heatmap: bars → numbers → heatmap
    # X-axis numbers on ax_a2's top edge: heatmap → numbers → bars
    ax_main.tick_params(left=True, labelleft=True, labelsize=7, length=0, pad=2,
                        bottom=False, labelbottom=False)
    ax_main.set_ylabel("Arm 1 position (1-based)", fontsize=9)
    ax_main.set_title(f"{rna}  —  Intramolecular contact frequency\n"
                      f"colour = log₁₊ₓ(read count)  |  white = no contact",
                      fontsize=10,pad=8)
    cb=fig.colorbar(im,cax=ax_cb)
    tv,tl=cbar_ticks(int(mat.max()))
    cb.set_ticks(tv); cb.set_ticklabels(tl,fontsize=7)
    cb.set_label("read count (raw)",fontsize=8)

    handles=draw_arm1_strip(ax_a1,stems_df,n)
    ax_a1.set_ylabel("Arm 1 position (1-based)", fontsize=9, labelpad=4)
    draw_arm2_strip(ax_a2,stems_df,n,large=(n>500))

    # Legend anchored just below ax_a2 in figure coordinates.
    # bbox_inches="tight" expands the canvas to include it — no margin hacks needed.
    # plot occupies top plot_h_in inches; legend occupies bottom legend_h_in inches
    plot_bottom_frac  = legend_h_in / total_h_in   # fraction from bottom where plot starts
    legend_top_frac   = (legend_h_in - 0.4) / total_h_in  # legend top just inside its area

    fig.legend(handles=handles, loc="upper center", ncol=ncol, fontsize=6.5,
               framealpha=0.95, facecolor="white", edgecolor="#cccccc",
               title="Interactions", title_fontsize=8,
               borderpad=0.5, handlelength=1.2, columnspacing=0.8,
               handletextpad=0.4,
               bbox_to_anchor=(0.55, legend_top_frac),
               bbox_transform=fig.transFigure)

    fig.subplots_adjust(bottom=plot_bottom_frac, hspace=0.01)

    out=out_dir/f"{rna}_heatmap_full.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.2)
    plt.close(); return out

# ── Output 2: per-stem zoomed PNGs ───────────────────────────────────────────

def plot_zooms(mat, stems_df, rna, pad, out_dir):
    n=mat.shape[0]; written=0
    for _,row in stems_df.iterrows():
        arms=parse_arms(row["sub_interaction_stem"])
        if arms is None: continue
        a1,a2=arms; i=int(row["interaction_number"]); c=icolor(i)
        r1s=max(0,a1[0]-1-pad); r1e=min(n,a1[1]+pad)
        r2s=max(0,a2[0]-1-pad); r2e=min(n,a2[1]+pad)
        sub=np.log1p(mat[r1s:r1e,r2s:r2e].astype(float))
        fig,ax=plt.subplots(figsize=(7,5))
        im=ax.imshow(sub,cmap=CMAP,origin="lower",aspect="auto",
                     extent=[r2s+0.5,r2e+0.5,r1s+0.5,r1e+0.5])
        ax.add_patch(Rectangle((a2[0]-0.5,a1[0]-0.5),a2[1]-a2[0]+1,a1[1]-a1[0]+1,
                                fill=False,edgecolor=c,linewidth=2,linestyle="--",
                                label=f"int{i} core arms"))
        st=max(1,(r1e-r1s+r2e-r2s)//16)
        ax.set_xticks(range(r2s+1,r2e+1,st))
        ax.set_xticklabels(range(r2s+1,r2e+1,st),rotation=90,fontsize=7)
        ax.set_yticks(range(r1s+1,r1e+1,st))
        ax.set_yticklabels(range(r1s+1,r1e+1,st),fontsize=7)
        ax.set_xlabel(f"Arm 2 position (1-based)  core: [{a2[0]}–{a2[1]}]",fontsize=9)
        ax.set_ylabel(f"Arm 1 position (1-based)  core: [{a1[0]}–{a1[1]}]",fontsize=9)
        ax.set_title(f"{rna}  int{i}: {row['sub_interaction_stem']}\n"
                     f"colour = log₁₊ₓ(read count)  |  dashed = core arm region",fontsize=9)
        mv=int(mat[r1s:r1e,r2s:r2e].max())
        tv,tl=cbar_ticks(mv); cb=plt.colorbar(im,ax=ax,fraction=0.025)
        cb.set_ticks(tv); cb.set_ticklabels(tl,fontsize=7); cb.set_label("read count (raw)",fontsize=8)
        ax.legend(fontsize=7,loc="upper right")
        plt.tight_layout()
        plt.savefig(out_dir/f"{rna}_int{i:03d}_zoom.png",dpi=130,bbox_inches="tight")
        plt.close(); written+=1
    return written

# ── Output 3: interactive HTML ────────────────────────────────────────────────

def plot_html(sparse_df, stems_df, rna, n, min_sup, out_path):
    hi=sparse_df[sparse_df["count"]>=min_sup].copy()
    x_all=hi["pos_arm2"].tolist()+hi["pos_arm1"].tolist()
    y_all=hi["pos_arm1"].tolist()+hi["pos_arm2"].tolist()
    c_all=hi["count"].tolist()*2
    log_c=[math.log1p(v) for v in c_all]
    hover=[f"arm1={y}, arm2={x}<br>count={c}" for x,y,c in zip(x_all,y_all,c_all)]
    cs=[[0.0,"white"],[0.15,"#AED6F1"],[0.40,"#2874A6"],
        [0.65,"#1A5276"],[0.82,"#922B21"],[1.0,"#E74C3C"]]
    ml=math.log1p(hi["count"].max()) if len(hi)>0 else 1.0
    rt=[v for v in [0,1,5,10,25,50,100,200,240] if v<=hi["count"].max()]
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=x_all,y=y_all,mode="markers",
        marker=dict(color=log_c,colorscale=cs,cmin=0,cmax=ml,size=3,symbol="square",
                    colorbar=dict(title="read count",
                                  tickvals=[math.log1p(v) for v in rt],
                                  ticktext=[str(v) for v in rt],len=0.5,x=1.02),
                    showscale=True),
        text=hover,hovertemplate="%{text}<extra></extra>",name="contacts",showlegend=False))
    shapes,leg=[],[]
    for _,row in stems_df.iterrows():
        arms=parse_arms(row["sub_interaction_stem"])
        if arms is None: continue
        a1,a2=arms; i=int(row["interaction_number"]); c=icolor(i)
        shapes.append(dict(type="rect",x0=a2[0]-0.5,x1=a2[1]+0.5,
                           y0=a1[0]-0.5,y1=a1[1]+0.5,
                           line=dict(color=c,width=1.5,dash="dash"),
                           fillcolor="rgba(0,0,0,0)",layer="above"))
        leg.append(go.Scatter(x=[None],y=[None],mode="markers",
                              marker=dict(symbol="square",color=c,size=10),
                              name=f"int{i}: {row['sub_interaction_stem']}",showlegend=True))
    for t in leg: fig.add_trace(t)
    ts=10 if n<=200 else 50 if n<=500 else 100
    fig.update_layout(shapes=shapes,
        title=dict(text=(f"{rna}  —  Intramolecular contact frequency<br>"
                         f"<sup>colour = log₁₊ₓ(read count)  |  dashed = core stem arm regions  "
                         f"|  zoom/pan with mouse  |  click legend to toggle</sup>"),font=dict(size=13)),
        xaxis=dict(title="Arm 2 position (1-based)",range=[0.5,n+0.5],
                   showgrid=False,zeroline=False,tickmode="linear",dtick=ts),
        yaxis=dict(title="Arm 1 position (1-based)",range=[0.5,n+0.5],
                   showgrid=False,zeroline=False,tickmode="linear",dtick=ts),
        width=1100,height=1000,plot_bgcolor="white",paper_bgcolor="white",
        legend=dict(title="Interactions<br><sub>(click to highlight)</sub>",
                    font=dict(size=8),itemsizing="constant",x=1.10,y=1,
                    bgcolor="rgba(255,255,255,0.9)",bordercolor="#cccccc",borderwidth=1),
        hovermode="closest")
    fig.write_html(str(out_path),include_plotlyjs="cdn",full_html=True)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args=parse_args()
    out_dir=Path(args.out); out_dir.mkdir(parents=True,exist_ok=True)
    print(f"\n{'='*55}\n  Contact Matrix Visualisation\n{'='*55}")
    print(f"  RNA:      {args.rna}  ({args.length} nt)")
    print(f"  Output:   {out_dir}")
    sparse_df=pd.read_csv(args.sparse,sep="\t")
    stems_df=pd.read_csv(args.stems)
    stems_df=stems_df[stems_df["name"]==args.rna].copy()

    if args.chimeric:
        chim=pd.read_csv(args.chimeric,usecols=["name","interaction_number"])
        valid=set(zip(chim["name"],chim["interaction_number"]))
        before=len(stems_df)
        stems_df=stems_df[stems_df.apply(
            lambda r:(r["name"],r["interaction_number"]) in valid,axis=1)].copy()
        print(f"  Stems filtered: {before} → {len(stems_df)} "
              f"({before-len(stems_df)} excluded — got 0 reads)")
    print(f"  Pairs:    {len(sparse_df):,}  |  Stems: {len(stems_df)}  |  "
          f"Count: {sparse_df['count'].min()}–{sparse_df['count'].max()}")

    mat=None
    if not args.no_full or not args.no_zoom:
        print(f"\n  Building {args.length}×{args.length} matrix "
              f"({args.length**2*4//1024//1024} MB)...")
        mat=build_matrix(sparse_df,args.length)

    if not args.no_full:
        print("\n[1/3] Full annotated heatmap...")
        p=plot_full(mat,stems_df,args.rna,out_dir)
        print(f"  → {p.name}")

    if not args.no_zoom:
        print(f"\n[2/3] Per-stem zoomed PNGs (pad={args.pad} nt)...")
        n=plot_zooms(mat,stems_df,args.rna,args.pad,out_dir)
        print(f"  → {n} PNGs written")

    if not args.no_html:
        print(f"\n[3/3] Interactive HTML...")
        hp=out_dir/f"{args.rna}_interactive.html"
        plot_html(sparse_df,stems_df,args.rna,args.length,args.min_support,hp)
        print(f"  → {hp.name}  ({os.path.getsize(hp)//1024} KB)")
        print(f"  Open:  xdg-open {hp}")

    print(f"\n{'='*55}\n  Done. All outputs in: {out_dir}\n{'='*55}\n")

if __name__=="__main__":
    main()
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

C_STAGE='#dbe6f2'; E_STAGE='#3f6d9e'
C_AUX='#fbe5d0';  E_AUX='#c8792f'
C_EVAL='#f4d9d9'; E_EVAL='#b23b3b'
INK='#222222'

fig,ax=plt.subplots(figsize=(12,7.0),dpi=200); ax.set_xlim(0,12); ax.set_ylim(0,7.05); ax.axis('off')

def box(x,y,w,h,t,fc,ec,fs=11,wt='normal'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.03,rounding_size=0.14",fc=fc,ec=ec,lw=1.8))
    if t: ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=fs,weight=wt,color=INK)

def arr(x1,y1,x2,y2,c=INK,ls='-',lw=2.0,t=None,dx=0.15,rad=0.0):
    cs=f'arc3,rad={rad}' if rad else 'arc3'
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=18,color=c,lw=lw,ls=ls,shrinkA=3,shrinkB=3,connectionstyle=cs))
    if t: ax.text((x1+x2)/2,(y1+y2)/2+dx,t,ha='center',va='bottom',fontsize=9,color=c)

# ---- stage boxes -----------------------------------------------------------
sy,sh,sw=2.75,1.35,3.2
box(0.15,sy,sw,sh,"Stage 1\nSupervised training\nAnti-UAV-RGBT (IR)",C_STAGE,E_STAGE,11,'bold')
box(4.40,sy,sw,sh,"Stage 2\nKnowledge distillation\nAnti-UAV410 (IR)",C_STAGE,E_STAGE,11,'bold')
box(8.65,sy,sw,sh,"Stage 3\nNaive / SSH / random-strat.\nCST Anti-UAV (IR)",C_STAGE,E_STAGE,11,'bold')
arr(3.40,sy+sh/2,4.38,sy+sh/2,t='init $\\theta$')
arr(7.65,sy+sh/2,8.63,sy+sh/2,t='init $\\theta$')

# ---- auxiliary boxes -------------------------------------------------------
ay,ah=4.85,1.05
box(4.40,ay,sw,ah,"Frozen Stage-1 teacher\nKD, $\\lambda_{kd}=1.0$",C_AUX,E_AUX,10)
arr(6.00,ay,6.00,sy+sh,c=E_AUX)
box(8.65,ay,sw,ah,"Stratified replay buffer\nSSH (herding) or random selection\n300 exemplars, 75 per stratum",C_AUX,E_AUX,9.5)
arr(10.25,ay,10.25,sy+sh,c=E_AUX)

# ---- provenance arcs from Stage 1 -----------------------------------------
arr(2.50,sy+sh+0.05,4.36,ay+0.45,c=E_AUX,lw=1.3,rad=-0.25)
ax.text(3.10,4.40,"frozen copy",fontsize=8.5,color=E_AUX,ha='center')
arr(1.10,sy+sh+0.05,8.85,ay+ah+0.06,c=E_AUX,lw=1.3,ls=(0,(5,3)),rad=-0.30)
ax.text(4.70,6.52,"exemplars from T1 train split",fontsize=8.5,color=E_AUX,ha='center')

# ---- evaluation box --------------------------------------------------------
ey,eh=0.50,1.10
box(1.5,ey,9.0,eh,None,C_EVAL,E_EVAL)
ax.text(6.0,ey+eh*0.64,"Evaluate Task-1 retention on Anti-UAV-RGBT val  $\\Rightarrow$  Forgetting Measure (FM)",
        ha='center',va='center',fontsize=11,weight='bold',color=INK)
ax.text(6.0,ey+eh*0.28,"overall + per size stratum (tiny / small / normal / large)",
        ha='center',va='center',fontsize=9.5,color=INK)
for cx in (1.75,6.0,10.25):
    arr(cx,sy,cx,ey+eh+0.03,c=E_EVAL,ls=(0,(4,3)),lw=1.4)

ax.text(6,6.80,"Single-stream YOLOMG (motion channel $=0$ throughout) — input modality held constant at thermal IR",
        ha='center',fontsize=9.5,style='italic',color='#555')
plt.tight_layout()
out='fig_methodology_pipeline.png'
fig.savefig(out,dpi=200,bbox_inches='tight'); print('saved',out)

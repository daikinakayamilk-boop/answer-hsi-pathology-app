import io
import math
import os
import shutil
import tempfile
import zipfile
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from skimage.color import rgb2hed
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu
from skimage.measure import regionprops
from skimage.morphology import binary_closing, binary_opening, disk, remove_small_holes, remove_small_objects
from skimage.segmentation import watershed

st.set_page_config(page_title="ANSWER | Hyperspectral Pathology", page_icon="🔬", layout="wide")

st.markdown("""
<style>
[data-testid="stSidebar"] { background: linear-gradient(180deg,#132431,#1a2e3e); }
[data-testid="stSidebar"] * { color:#f4f8fc; }
.block-container { max-width: 1500px; padding-top: 0.8rem; }
.answer-logo { font-size: 2.1rem; font-weight: 800; letter-spacing: .12em; color: #f8fbff; margin: .2rem 0 0; }
.answer-sub { color:#a9bbca; font-size:.8rem; margin-bottom:1rem; }
.small { color:#64748b; font-size:.86rem; }
</style>
""", unsafe_allow_html=True)

@dataclass
class HSIConfig:
    width: int = 1920
    height: int = 1080
    bands: int = 141
    header_bytes: int = 1000000
    interleave: str = "BIL"
    wl_start: float = 350.0
    wl_step: float = 5.0
    max_width: int = 480

    @property
    def wavelengths(self):
        return self.wl_start + self.wl_step * np.arange(self.bands, dtype=np.float32)

def save_upload(upload, folder):
    os.makedirs(folder, exist_ok=True)
    name = os.path.basename(upload.name)
    raw = os.path.join(folder, name)
    upload.seek(0)
    with open(raw, "wb") as f:
        shutil.copyfileobj(upload, f, length=8 * 1024 * 1024)
    if name.lower().endswith(".zip"):
        with zipfile.ZipFile(raw) as z:
            members = [m for m in z.namelist() if m.lower().endswith(".hsd") and not m.endswith("/")]
            if not members:
                raise ValueError("ZIP内に .hsd ファイルが見つかりません。")
            member = members[0]
            target = os.path.join(folder, os.path.basename(member))
            with z.open(member) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
            return target
    return raw

def infer_default(path, cfg):
    size = os.path.getsize(path)
    expected = cfg.header_bytes + 1920 * 1080 * 141 * 2
    if size == expected:
        cfg.width, cfg.height, cfg.bands = 1920, 1080, 141
        cfg.interleave = "BIL"
        cfg.wl_start, cfg.wl_step = 350.0, 5.0
    return cfg

def open_cube(path, cfg):
    shape_bil = (cfg.height, cfg.bands, cfg.width)
    if cfg.interleave == "BIL":
        return np.memmap(path, dtype="<u2", mode="r", offset=cfg.header_bytes, shape=shape_bil).transpose(0,2,1)
    if cfg.interleave == "BIP":
        return np.memmap(path, dtype="<u2", mode="r", offset=cfg.header_bytes, shape=(cfg.height,cfg.width,cfg.bands))
    return np.memmap(path, dtype="<u2", mode="r", offset=cfg.header_bytes, shape=(cfg.bands,cfg.height,cfg.width)).transpose(1,2,0)

def downsample(cube, max_width):
    factor = max(1, int(math.ceil(cube.shape[1] / max_width)))
    return cube[::factor, ::factor, :].astype(np.float32), factor

def white_correct(sample, white, wavelengths):
    if white is not None:
        if white.shape[:2] == sample.shape[:2]:
            return np.clip(sample / np.maximum(white, 1.0), 0, 4).astype(np.float32)
        ws = np.maximum(white.reshape(-1, white.shape[-1]).mean(axis=0), 1.0)
        return np.clip(sample / ws[None,None,:], 0, 4).astype(np.float32)
    idx = [int(np.argmin(np.abs(wavelengths-w))) for w in (450,550,650)]
    vis = sample[...,idx].mean(axis=2)
    th = np.percentile(vis, 99)
    bright = sample[vis >= th]
    ws = bright.mean(axis=0) if len(bright) else sample.reshape(-1,sample.shape[-1]).mean(axis=0)
    return np.clip(sample / np.maximum(ws,1.0)[None,None,:], 0, 4).astype(np.float32)

def xbar(l):
    t1=(l-442.0)*np.where(l<442.0,0.0624,0.0374)
    t2=(l-599.8)*np.where(l<599.8,0.0264,0.0323)
    t3=(l-501.1)*np.where(l<501.1,0.0490,0.0382)
    return 0.362*np.exp(-0.5*t1*t1)+1.056*np.exp(-0.5*t2*t2)-0.065*np.exp(-0.5*t3*t3)

def ybar(l):
    t1=(l-568.8)*np.where(l<568.8,0.0213,0.0247)
    t2=(l-530.9)*np.where(l<530.9,0.0613,0.0322)
    return 0.821*np.exp(-0.5*t1*t1)+0.286*np.exp(-0.5*t2*t2)

def zbar(l):
    t1=(l-437.0)*np.where(l<437.0,0.0845,0.0278)
    t2=(l-459.0)*np.where(l<459.0,0.0385,0.0725)
    return 1.217*np.exp(-0.5*t1*t1)+0.681*np.exp(-0.5*t2*t2)

def stretch(arr, low=.8, high=99.2, gamma=.92):
    out=np.empty_like(arr,dtype=np.float32)
    for c in range(arr.shape[-1]):
        lo,hi=np.percentile(arr[...,c],[low,high])
        x=np.clip((arr[...,c]-lo)/(hi-lo+1e-6),0,1)
        out[...,c]=x**gamma
    return out

def cie_rgb(cube,w):
    cmf=np.stack([xbar(w),ybar(w),zbar(w)],axis=1).astype(np.float32)
    cmf[(w<380)|(w>780)]=0
    cmf/=cmf[:,1].sum()+1e-8
    xyz=(cube.reshape(-1,cube.shape[-1])@cmf).reshape(cube.shape[0],cube.shape[1],3)
    M=np.array([[3.2406,-1.5372,-0.4986],[-0.9689,1.8758,0.0415],[0.0557,-0.2040,1.0570]],np.float32)
    lin=np.clip(xyz@M.T,0,None)
    a=.055
    rgb=np.where(lin<=.0031308,12.92*lin,(1+a)*np.power(lin,1/2.4)-a)
    return (stretch(np.clip(rgb,0,2),.5,99.7,.95)*255).astype(np.uint8)

def three_band(cube,w,bands_nm):
    ids=[int(np.argmin(np.abs(w-x))) for x in bands_nm]
    arr=cube[...,[ids[2],ids[1],ids[0]]]
    return (stretch(arr,.8,99.2,.86)*255).astype(np.uint8)

def pca_image(cube,w):
    valid=(w>=400)&(w<=900)
    X=cube[...,valid].reshape(-1,valid.sum())
    rng=np.random.default_rng(0)
    train=X[rng.choice(len(X),min(30000,len(X)),replace=False)]
    model=PCA(n_components=1,random_state=0).fit(train)
    pc=model.transform(X)[:,0].reshape(cube.shape[:2])
    lo,hi=np.percentile(pc,[1,99])
    n=np.clip((pc-lo)/(hi-lo+1e-6),0,1)
    stops=np.array([[0,48,18,59],[.25,35,140,200],[.5,70,200,100],[.75,245,200,40],[1,180,30,30]],np.float32)
    out=np.zeros((*n.shape,3),np.float32)
    for a,b in zip(stops[:-1],stops[1:]):
        m=(n>=a[0])&(n<=b[0]); t=(n[m]-a[0])/(b[0]-a[0]+1e-8)
        out[m]=a[1:]+(b[1:]-a[1:])*t[:,None]
    return out.astype(np.uint8)

def cluster_image(cube,w,k):
    valid=(w>=420)&(w<=800)
    X=cube[...,valid].reshape(-1,valid.sum())
    rng=np.random.default_rng(1)
    train=X[rng.choice(len(X),min(25000,len(X)),replace=False)]
    mu=train.mean(0); sd=train.std(0)+1e-6
    km=MiniBatchKMeans(n_clusters=k,random_state=0,n_init=3,batch_size=2048).fit((train-mu)/sd)
    labels=km.predict((X-mu)/sd).reshape(cube.shape[:2])
    palette=np.array([[45,106,150],[91,160,180],[123,196,130],[240,183,72],[211,86,67],[140,90,170],[85,85,85]],np.uint8)
    return palette[labels%len(palette)]

def tissue_mask(rgb):
    x=rgb.astype(np.float32)/255
    sat=x.max(2)-x.min(2); val=x.mean(2)
    m=(sat>.06)&(val<.985)
    m=binary_closing(m,disk(2))
    m=remove_small_holes(m,area_threshold=150)
    return remove_small_objects(m,min_size=150)

def cell_count(rgb):
    tissue=tissue_mask(rgb)
    H=rgb2hed(rgb.astype(np.float32)/255)[...,0]
    vals=H[tissue]
    if len(vals)<20:
        return rgb.copy(),0
    th=threshold_otsu(vals)
    nuc=(H>th*.90)&tissue
    nuc=binary_opening(nuc,disk(1))
    nuc=remove_small_objects(nuc,min_size=4)
    D=ndi.distance_transform_edt(nuc)
    peaks=peak_local_max(D,min_distance=2,threshold_abs=.8,labels=nuc)
    markers=np.zeros_like(nuc,np.int32)
    for i,(r,c) in enumerate(peaks,1): markers[r,c]=i
    labels=watershed(-D,ndi.label(markers>0)[0],mask=nuc)
    boxes=[]
    for reg in regionprops(labels):
        if 4<=reg.area<=700:
            minr,minc,maxr,maxc=reg.bbox
            boxes.append((minc,minr,maxc,maxr))
    gray=np.dot(rgb[...,:3],[.299,.587,.114]).astype(np.uint8)
    out=np.stack([gray,gray,gray],2)
    im=Image.fromarray(out); d=ImageDraw.Draw(im)
    for x0,y0,x1,y1 in boxes:
        d.rectangle((max(0,x0-2),max(0,y0-2),min(out.shape[1]-1,x1+2),min(out.shape[0]-1,y1+2)),outline=(255,0,0),width=2)
    return np.array(im),len(boxes)

def supervised(cube,rgb,w):
    tissue=tissue_mask(rgb)
    H=rgb2hed(rgb.astype(np.float32)/255)[...,0]
    vals=H[tissue]
    hi=np.percentile(vals,82); lo=np.percentile(vals,35)
    pos=(H>=hi)&tissue; neg=(H<=lo)&tissue
    valid=(w>=400)&(w<=900)
    Xall=cube[...,valid].reshape(-1,valid.sum())
    pi=np.flatnonzero(pos.ravel()); ni=np.flatnonzero(neg.ravel())
    rng=np.random.default_rng(42)
    n=min(5000,len(pi),len(ni))
    if n<100: raise ValueError("教師データが不足しています。")
    ids=np.concatenate([rng.choice(pi,n,False),rng.choice(ni,n,False)])
    y=np.concatenate([np.ones(n,np.uint8),np.zeros(n,np.uint8)])
    sc=StandardScaler(); Xt=sc.fit_transform(Xall[ids])
    clf=RandomForestClassifier(n_estimators=60,max_depth=10,min_samples_leaf=4,random_state=42,n_jobs=1,class_weight="balanced").fit(Xt,y)
    p=np.zeros(len(Xall),np.float32)
    for s in range(0,len(Xall),30000):
        e=min(len(Xall),s+30000); p[s:e]=clf.predict_proba(sc.transform(Xall[s:e]))[:,1]
    p=p.reshape(cube.shape[:2]); p[~tissue]=0
    pred=(p>=.58)&tissue
    stops=np.array([[0,0,0,0],[.15,0,25,180],[.42,0,215,255],[.70,255,235,0],[1,255,30,0]],np.float32)
    heat=np.zeros((*p.shape,3),np.float32)
    for a,b in zip(stops[:-1],stops[1:]):
        m=(p>=a[0])&(p<=b[0]); t=(p[m]-a[0])/(b[0]-a[0]+1e-8)
        heat[m]=a[1:]+(b[1:]-a[1:])*t[:,None]
    heat[~tissue]=0
    ps=cube[pred].mean(0) if pred.any() else cube[pos].mean(0)
    ns=cube[tissue&(~pred)].mean(0)
    return heat.astype(np.uint8),ps,ns

def png_bytes(arr):
    b=io.BytesIO(); Image.fromarray(arr).save(b,format="PNG"); return b.getvalue()

with st.sidebar:
    st.markdown('<div class="answer-logo">ANSWER</div><div class="answer-sub">Hyperspectral AI for a Visible Future</div>',unsafe_allow_html=True)
    st.markdown("### 解析メニュー")
    st.markdown("• CIE補正RGB")
    st.markdown("• 3バンド表示")
    st.markdown("• PCA")
    st.markdown("• クラスタリング")
    st.markdown("• セルカウント")
    st.markdown("• 教師あり予測")
    st.caption("研究・デモ用途")

st.title("ANSWER 病理ハイパースペクトル解析")
st.caption("顕微鏡HSIをアップロードすると、6種類の画像処理とスペクトル比較を実行します。")

a,b=st.columns(2)
sample=a.file_uploader("ハイパースペクトルデータ",type=["hsd","zip"])
white=b.file_uploader("白色参照データ（任意）",type=["hsd","zip"])

with st.expander("HSD設定"):
    c1,c2,c3,c4=st.columns(4)
    width=int(c1.number_input("Width",value=1920,min_value=1))
    height=int(c2.number_input("Height",value=1080,min_value=1))
    bands=int(c3.number_input("Bands",value=141,min_value=1))
    header=int(c4.number_input("Header bytes",value=1000000,min_value=0,step=1000))
    c5,c6,c7,c8=st.columns(4)
    interleave=c5.selectbox("Interleave",["BIL","BIP","BSQ"])
    wl_start=float(c6.number_input("開始波長",value=350.0))
    wl_step=float(c7.number_input("波長間隔",value=5.0))
    max_width=int(c8.slider("解析幅",360,720,480,40))

c1,c2,c3,c4=st.columns(4)
w1=float(c1.number_input("短波長",value=450.0,step=5.0))
w2=float(c2.number_input("中波長",value=600.0,step=5.0))
w3=float(c3.number_input("長波長",value=700.0,step=5.0))
k=int(c4.slider("クラスタ数",3,8,5))

if st.button("▶ 解析実行",type="primary",use_container_width=True):
    if sample is None:
        st.error("HSIデータをアップロードしてください。")
        st.stop()
    cfg=HSIConfig(width,height,bands,header,interleave,wl_start,wl_step,max_width)
    try:
        with tempfile.TemporaryDirectory(prefix="answer_") as td:
            with st.status("解析中",expanded=True) as status:
                sp=save_upload(sample,os.path.join(td,"sample"))
                cfg=infer_default(sp,cfg)
                cube=open_cube(sp,cfg)
                small,factor=downsample(cube,cfg.max_width)
                ws=None
                if white is not None:
                    wp=save_upload(white,os.path.join(td,"white"))
                    wc=open_cube(wp,cfg)
                    ws=wc[::factor,::factor,:].astype(np.float32)
                w=cfg.wavelengths
                corr=white_correct(small,ws,w)
                st.write("RGB・3バンド・PCA")
                rgb=cie_rgb(corr,w)
                three=three_band(corr,w,(w1,w2,w3))
                pca=pca_image(corr,w)
                st.write("クラスタリング")
                cl=cluster_image(corr,w,k)
                st.write("セルカウント")
                count_img,ncells=cell_count(rgb)
                st.write("教師あり核ピクセル予測")
                heat,ps,ns=supervised(corr,rgb,w)
                status.update(label="解析完了",state="complete",expanded=False)

            m1,m2,m3=st.columns(3)
            m1.metric("検出核数（簡易）",f"{ncells:,}")
            m2.metric("解析解像度",f"{rgb.shape[1]}×{rgb.shape[0]}")
            m3.metric("波長帯",f"{w[0]:.0f}–{w[-1]:.0f} nm")

            r1,r2=st.columns(2)
            r1.markdown("### 01 CIE補正RGB"); r1.image(rgb,use_container_width=True)
            r2.markdown(f"### 02 3バンド {w1:.0f}/{w2:.0f}/{w3:.0f} nm"); r2.image(three,use_container_width=True)
            r3,r4=st.columns(2)
            r3.markdown("### 03 PCA 第1主成分"); r3.image(pca,use_container_width=True)
            r4.markdown(f"### 04 クラスタリング K={k}"); r4.image(cl,use_container_width=True)
            r5,r6=st.columns(2)
            r5.markdown("### 05 セルカウント"); r5.image(count_img,use_container_width=True)
            r6.markdown("### 06 教師あり核ピクセル予測"); r6.image(heat,use_container_width=True)

            st.markdown("### スペクトル比較")
            fig=go.Figure()
            fig.add_trace(go.Scatter(x=w,y=ps,mode="lines",name="核（予測領域）"))
            fig.add_trace(go.Scatter(x=w,y=ns,mode="lines",name="非核領域"))
            fig.update_layout(height=360,xaxis_title="波長 (nm)",yaxis_title="補正反射強度",margin=dict(l=40,r=20,t=20,b=40))
            st.plotly_chart(fig,use_container_width=True)

            outputs={
                "01_CIE_RGB.png":rgb,
                "02_3band.png":three,
                "03_PCA_PC1.png":pca,
                "04_clustering.png":cl,
                "05_cell_count.png":count_img,
                "06_supervised_probability.png":heat,
            }
            spec=pd.DataFrame({"wavelength_nm":w,"nucleus_predicted":ps,"non_nucleus":ns})
            bundle=io.BytesIO()
            with zipfile.ZipFile(bundle,"w",zipfile.ZIP_DEFLATED) as z:
                for name,arr in outputs.items(): z.writestr(name,png_bytes(arr))
                z.writestr("spectra.csv",spec.to_csv(index=False).encode("utf-8-sig"))
            st.download_button("解析結果をZIPで保存",bundle.getvalue(),file_name="ANSWER_analysis_results.zip",mime="application/zip",type="primary")
            st.warning("研究・デモ用途の簡易解析プロトタイプです。病理診断には使用しないでください。")
    except Exception as e:
        st.exception(e)
else:
    st.info("HSIファイルをアップロードして「解析実行」を押してください。")

"""
Stage 3 manual review annotator (tkinter). Play an episode like a video and press
number keys to stamp / adjust the 6 CRITICAL POINTS (subtask transitions).

This is the REVIEW stage of the 3-stage pipeline: the timeline shows the VLM
(Stage 1) and state (Stage 2) proposals plus the fused result, the human marks are
pre-seeded from the fused annotation, and the points the two machines disagreed on
(fused review_points) are flagged with red ▲ — so you only adjust those and save.

Critical points (keys 1..6):
  1 抓试管 (grasp tube)  2 开始倒 (start pour)  3 放试管 (release tube)
  4 抓杵 (grasp pestle)  5 开始磨 (start grind) 6 抬杵 (lift pestle)
=> 7 subtasks. Episode start (0) and end (N-1) are implicit.

Saves human labels to mvt_annotations_human/.

Run:  & "C:\\Users\\jerry\\miniconda3\\envs\\vlm\\python.exe" annotate_gui.py --ep 0
Keys: Space play/pause  <- -> step  , . jump10  Home/End ends  +/- speed
      1..6 mark point   Shift+1..6 clear   0 clear all   f seed-from-fused
      s save   n/p ep   q quit
"""
import argparse, io, os, json, glob, sys
import numpy as np, pandas as pd
import tkinter as tk
from PIL import Image, ImageTk, ImageEnhance, ImageDraw

from batch_annotate import LABELS   # 7 subtask labels (single source of truth)

# 6 critical points marked by keys 1..6 (= start frame of S1..S6; S0 starts at 0)
BND = ["B1 抓试管 S0→S1", "B2 开始倒 S1→S2", "B3 放试管 S2→S3",
       "B4 抓杵 S3→S4", "B5 开始磨 S4→S5", "B6 抬杵 S5→S6"]
COLORS = ["#ff5555", "#ffb000", "#ff7a00", "#a070ff", "#00dc5a", "#50a0ff", "#ff66cc"]
NB = len(BND)          # 6 critical points
NS = NB + 1            # 7 subtasks

ap = argparse.ArgumentParser()
ap.add_argument("--ep", type=int, default=0)
ap.add_argument("--data", default=r"C:\Intern\black_smash_07\data\chunk-000")
ap.add_argument("--out", default=r"C:\Intern\mvt_annotations_human")
ap.add_argument("--fused", default=r"C:\Intern\mvt_annotations_fused")
ap.add_argument("--state", default=r"C:\Intern\mvt_annotations")
ap.add_argument("--vlm", default=r"C:\Intern\mvt_annotations_vlm")
ap.add_argument("--meta", default=r"C:\Intern\black_smash_07\meta\tasks.jsonl")
ap.add_argument("--layout", default="all", choices=["all", "both", "cam1", "cam0"])
ap.add_argument("--check", action="store_true")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

TASK = "Pour the black powder into the mortar and grind."
try: TASK = json.loads(open(args.meta).readline())["task"]
except Exception: pass

EPS = sorted(int(os.path.basename(f).split("_")[1].split(".")[0])
             for f in glob.glob(os.path.join(args.data, "episode_*.parquet")))

C0, C1 = "observation.images.camera0", "observation.images.camera1"
TL0, TR0 = "observation.images.tactile_left_0", "observation.images.tactile_right_0"
TL1, TR1 = "observation.images.tactile_left_1", "observation.images.tactile_right_1"

TL_L = 40          # left margin in the timeline canvas (row labels)
TL_H = 78          # timeline canvas height


def get_layout(layout):
    if layout == "all":
        SC, ST, G, LH = 260, 130, 6, 16
        items = [(C0, 0, LH, SC, "cam0"), (C1, SC + G, LH, SC, "cam1"),
                 (TL0, 0, LH + SC + LH, ST, "tac L0"), (TR0, ST + G, LH + SC + LH, ST, "tac R0"),
                 (TL1, 2 * (ST + G), LH + SC + LH, ST, "tac L1"), (TR1, 3 * (ST + G), LH + SC + LH, ST, "tac R1")]
        W = max(2 * SC + G, 4 * ST + 3 * G); H = LH + SC + LH + ST
    elif layout == "both":
        S, G, LH = 360, 6, 16
        items = [(C0, 0, LH, S, "cam0"), (C1, S + G, LH, S, "cam1")]; W = 2 * S + G; H = LH + S
    else:
        cam = C1 if layout == "cam1" else C0; S, LH = 480, 16
        items = [(cam, 0, LH, S, layout)]; W = S; H = LH + S
    return sorted({it[0] for it in items}), items, W, H


STREAMS, ITEMS, COMP_W, COMP_H = get_layout(args.layout)


def enh(im):
    a = np.asarray(im).astype(np.float32); lo, hi = np.percentile(a, 1), np.percentile(a, 99)
    a = np.clip((a - lo) / (hi - lo + 1e-6), 0, 1) * 255
    o = Image.fromarray(a.astype(np.uint8))
    return ImageEnhance.Contrast(ImageEnhance.Color(o).enhance(1.35)).enhance(1.15)


def composite(dec):
    cv = Image.new("RGB", (COMP_W, COMP_H), (8, 8, 8)); d = ImageDraw.Draw(cv)
    for stream, x, y, size, label in ITEMS:
        cv.paste(enh(dec[stream]).resize((size, size)), (x, y))
        d.text((x + 2, y - 14), label, fill=(0, 230, 120))
    return cv


def load_frames(ep):
    df = pd.read_parquet(os.path.join(args.data, f"episode_{ep:06d}.parquet"), columns=STREAMS)
    n = len(df)
    print(f"loading ep{ep:03d}: {n} frames x {len(STREAMS)} streams ...", flush=True)
    out = []
    for i in range(n):
        dec = {s: Image.open(io.BytesIO(df[s].iloc[i]["bytes"])).convert("RGB") for s in STREAMS}
        out.append(composite(dec))
        if i % 150 == 0 and i: print(f"  {i}/{n}", flush=True)
    print("  done.", flush=True)
    return out, n


def load_cps(d, ep):
    fp = os.path.join(d, f"ep{ep:03d}_subtasks.json")
    if not os.path.exists(fp):
        return None, None
    try:
        j = json.load(open(fp))
        cps = j.get("critical_points")
        return (cps if cps and len(cps) == NB else None), j.get("review_points")
    except Exception:
        return None, None


class Annotator:
    def __init__(self, root):
        self.root = root; self.fps = 30; self.speed = 1.0
        self.ep_pos = EPS.index(args.ep) if args.ep in EPS else 0
        self.playing = False; self.cur = 0
        self.build(); self.load(EPS[self.ep_pos]); self.tick()

    def build(self):
        self.root.title(f"Subtask Review — 全画面 · {NB} 临界点 / {NS} 段")
        self.root.configure(bg="#111")
        self.info = tk.Label(self.root, font=("Consolas", 13), fg="#eee", bg="#111", anchor="w")
        self.info.pack(fill="x", padx=8, pady=(6, 2))
        self.img_lbl = tk.Label(self.root, bg="#000"); self.img_lbl.pack(padx=8)
        self.tl = tk.Canvas(self.root, width=COMP_W, height=TL_H, bg="#000", highlightthickness=0)
        self.tl.pack(padx=8, pady=4)
        self.marks_lbl = tk.Label(self.root, font=("Consolas", 11), fg="#ddd", bg="#111", anchor="w")
        self.marks_lbl.pack(fill="x", padx=8)
        h = ("空格 播放/暂停   <-/-> 单帧   ,/. 跳10   Home/End 首尾   +/- 速度   f 重置为融合值\n"
             "1=抓试管 2=开始倒 3=放试管 4=抓杵 5=开始磨 6=抬杵   Shift+1~6 清除   0 清空   s 保存   n/p 换集   q 退出")
        tk.Label(self.root, text=h, font=("Consolas", 10), fg="#888", bg="#111", justify="left",
                 anchor="w").pack(fill="x", padx=8, pady=(2, 6))
        for seq in ["<space>", "<Left>", "<Right>", "<comma>", "<period>", "<Home>", "<End>",
                    "<plus>", "<minus>", "<equal>", "s", "n", "p", "q", "<Escape>", "0", "f"]:
            self.root.bind(seq, self.on_key)
        for d in range(1, NB + 1):
            self.root.bind(str(d), self.on_key); self.root.bind(f"<Shift-Key-{d}>", self.on_key)

    def load(self, ep):
        self.ep = ep
        self.frames, self.N = load_frames(ep)
        self.marks = {i: None for i in range(1, NB + 1)}
        self.cur = 0; self.playing = False
        self.photo = ImageTk.PhotoImage(self.frames[0]); self.img_lbl.config(image=self.photo)
        # references
        self.ref_vlm, _ = load_cps(args.vlm, ep)
        self.ref_state, _ = load_cps(args.state, ep)
        self.ref_fused, self.review = load_cps(args.fused, ep)
        self.review = self.review or []
        self.seed_marks(ep)
        self.display()

    def seed_marks(self, ep):
        # priority: existing human file > fused > state
        for src in (args.out, args.fused, args.state):
            cps, _ = load_cps(src, ep)
            if cps:
                for i in range(1, NB + 1): self.marks[i] = cps[i - 1]
                print(f"  seeded marks from {src}")
                return

    def starts(self):
        return [0] + [self.marks[i] for i in range(1, NB + 1)]

    def display(self):
        self.photo.paste(self.frames[self.cur])
        self.info.config(text=f"ep{self.ep:03d} ({self.ep_pos+1}/{len(EPS)})  frame {self.cur:4d}/{self.N-1}"
                              f"  t={self.cur/self.fps:6.2f}s  {'PLAY' if self.playing else 'PAUSE'} x{self.speed:g}")
        self.marks_lbl.config(text="临界点: " + "  ".join(
            ("★" if i in self.review else "") +
            f"{BND[i-1]}={'%d' % self.marks[i] if self.marks[i] is not None else '--'}"
            for i in range(1, NB + 1)) + ("   (★=两法不一致,重点看)" if self.review else ""))
        self.draw_timeline()

    def _bar(self, y, h, starts, label):
        W = COMP_W; N = self.N
        self.tl.create_text(2, y, text=label, fill="#aaa", anchor="nw", font=("Consolas", 8))
        def X(fr): return int(TL_L + fr / N * (W - TL_L))
        for i in range(NS):
            s = starts[i] if i < len(starts) else None
            if s is None: continue
            e = next((starts[j] for j in range(i + 1, NS) if j < len(starts) and starts[j] is not None), N)
            self.tl.create_rectangle(X(s), y, X(e), y + h, fill=COLORS[i], outline="#111")
            if X(e) - X(s) > 13:
                self.tl.create_text(X(s) + 2, y + 1, text=f"S{i}", fill="#000", anchor="nw", font=("Consolas", 7))

    def draw_timeline(self):
        self.tl.delete("all"); W = COMP_W; N = self.N
        def X(fr): return int(TL_L + fr / N * (W - TL_L))
        if self.ref_vlm:   self._bar(14, 8, [0] + self.ref_vlm, "VLM")
        if self.ref_state: self._bar(26, 8, [0] + self.ref_state, "state")
        self._bar(42, 18, self.starts(), "人工")
        # review markers (red triangles) at fused positions of disagreed points
        for p in self.review:
            cp = self.ref_fused[p - 1] if self.ref_fused else None
            if cp is not None:
                x = X(cp); self.tl.create_polygon(x - 4, 62, x + 4, 62, x, 70, fill="#ff3030")
        x = X(self.cur); self.tl.create_line(x, 0, x, TL_H, fill="#fff", width=2)

    def clamp(self, i): return max(0, min(self.N - 1, i))

    def on_key(self, e):
        k = e.keysym; shift = (e.state & 0x0001) != 0
        if k == "space": self.playing = not self.playing
        elif k == "Left": self.playing = False; self.cur = self.clamp(self.cur - 1)
        elif k == "Right": self.playing = False; self.cur = self.clamp(self.cur + 1)
        elif k == "comma": self.playing = False; self.cur = self.clamp(self.cur - 10)
        elif k == "period": self.playing = False; self.cur = self.clamp(self.cur + 10)
        elif k == "Home": self.cur = 0
        elif k == "End": self.cur = self.N - 1
        elif k in ("plus", "equal"): self.speed = min(4.0, self.speed * 1.5)
        elif k == "minus": self.speed = max(0.25, self.speed / 1.5)
        elif k in ("q", "Escape"): self.root.destroy(); return
        elif k == "s": self.save()
        elif k == "n": self.switch(+1)
        elif k == "p": self.switch(-1)
        elif k == "f":
            cps = self.ref_fused
            if cps:
                for i in range(1, NB + 1): self.marks[i] = cps[i - 1]
        elif k == "0": self.marks = {i: None for i in range(1, NB + 1)}
        elif k.isdigit() and 1 <= int(k) <= NB:
            self.marks[int(k)] = None if shift else self.cur
        self.display()

    def switch(self, delta):
        if any(self.marks[i] is not None for i in range(1, NB + 1)): self.save(quiet=True)
        self.ep_pos = (self.ep_pos + delta) % len(EPS); self.load(EPS[self.ep_pos])

    def save(self, quiet=False):
        cps = [self.marks[i] for i in range(1, NB + 1)]
        if any(c is None for c in cps):
            self.info.config(text=f"!! {NB} 个临界点未标全,先标完再保存"); return
        if cps != sorted(cps):
            self.info.config(text="!! 临界点不是递增顺序,先修正"); return
        st = [0] + cps
        subs = []
        for i in range(NS):
            a = st[i]; b = (st[i + 1] - 1) if i < NS - 1 else self.N - 1
            subs.append({"subtask_id": i, "label": LABELS[i], "start_frame": a, "end_frame": b,
                         "start_t": round(a / self.fps, 2), "end_t": round(b / self.fps, 2),
                         "n_frames": b - a + 1, "dur_s": round((b - a + 1) / self.fps, 2)})
        doc = {"episode_index": self.ep, "task": TASK, "n_frames": self.N, "fps": self.fps,
               "annotator": "human-gui", "critical_points": cps, "subtask_starts": st,
               "n_subtasks": NS, "subtasks": subs}
        fp = os.path.join(args.out, f"ep{self.ep:03d}_subtasks.json")
        json.dump(doc, open(fp, "w"), indent=2)
        if not quiet: self.info.config(text=f"saved -> {fp}")
        print("saved", fp, flush=True)

    def tick(self):
        if self.playing:
            if self.cur < self.N - 1: self.cur += 1
            else: self.playing = False
            self.display()
        self.root.after(int(1000 / (self.fps * self.speed)), self.tick)


if args.check:
    print(f"layout={args.layout} composite={COMP_W}x{COMP_H} streams={len(STREAMS)} points={NB} subtasks={NS}")
    df = pd.read_parquet(os.path.join(args.data, f"episode_{args.ep:06d}.parquet"), columns=STREAMS).head(5)
    fr = [composite({s: Image.open(io.BytesIO(df[s].iloc[i]["bytes"])).convert("RGB") for s in STREAMS}) for i in range(5)]
    r = tk.Tk(); a = Annotator.__new__(Annotator)
    a.root = r; a.fps = 30; a.speed = 1.0; a.ep_pos = 0; a.playing = False; a.cur = 0
    a.build(); a.frames = fr; a.N = 5; a.ep = args.ep; a.marks = {i: None for i in range(1, NB + 1)}
    a.ref_vlm = a.ref_state = a.ref_fused = None; a.review = []
    a.photo = ImageTk.PhotoImage(a.frames[0]); a.img_lbl.config(image=a.photo); a.display()
    r.update(); r.destroy(); print("CHECK OK"); sys.exit(0)

root = tk.Tk()
Annotator(root)
root.mainloop()

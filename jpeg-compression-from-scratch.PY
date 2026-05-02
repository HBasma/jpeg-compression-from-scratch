import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import math
import io
import heapq
from collections import Counter
# -----------------------
# HARROUCHE BASMA
# -----------------------


# ===== RGB <-> YCbCr =====
def rgb_to_ycbcr(img):
    R = img[:, :, 0].astype(float)
    G = img[:, :, 1].astype(float)
    B = img[:, :, 2].astype(float)
    Y  =  0.299  * R + 0.587  * G + 0.114  * B
    Cb = -0.1687 * R - 0.3313 * G + 0.5    * B + 128
    Cr =  0.5    * R - 0.4187 * G - 0.0813 * B + 128
    return Y, Cb, Cr

def ycbcr_to_rgb(Y, Cb, Cr):
    R = Y + 1.402   * (Cr - 128)
    G = Y - 0.34414 * (Cb - 128) - 0.71414 * (Cr - 128)
    B = Y + 1.772   * (Cb - 128)
    return np.clip(np.stack([R, G, B], axis=2), 0, 255).astype(np.uint8)


# SOUS ECHANTILLONNAGE 4:2:0 
def subsample_420(Cb, Cr):
    return Cb[::2, ::2], Cr[::2, ::2]

def upsample_420(Cb_sub, Cr_sub, shape):
    h, w = shape
    Cb = np.repeat(np.repeat(Cb_sub, 2, axis=0), 2, axis=1)
    Cr = np.repeat(np.repeat(Cr_sub, 2, axis=0), 2, axis=1)
    # padding 
    if Cb.shape[0] < h or Cb.shape[1] < w:
        Cb_out = np.zeros((h, w))
        Cr_out = np.zeros((h, w))
        Cb_out[:Cb.shape[0], :Cb.shape[1]] = Cb
        Cr_out[:Cr.shape[0], :Cr.shape[1]] = Cr
        return Cb_out, Cr_out
    return Cb[:h, :w], Cr[:h, :w]


#  BLOCS 8×8 
def split_blocks(channel):
    h, w = channel.shape
    new_h = (h + 7) // 8 * 8
    new_w = (w + 7) // 8 * 8
    padded = np.zeros((new_h, new_w))
    padded[:h, :w] = channel
    blocks = [padded[i:i+8, j:j+8]
              for i in range(0, new_h, 8)
              for j in range(0, new_w, 8)]
    return blocks, new_h, new_w

def merge_blocks(blocks, h, w):
    img = np.zeros((h, w))
    idx = 0
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            img[i:i+8, j:j+8] = blocks[idx]
            idx += 1
    return img


#  DCT / IDCT 
def dct2(block):
    N = 8
    result = np.zeros((N, N))
    for u in range(N):
        for v in range(N):
            s = 0
            for x in range(N):
                for y in range(N):
                    s += block[x, y] * \
                         np.cos((2*x+1)*u*np.pi/16) * \
                         np.cos((2*y+1)*v*np.pi/16)
            cu = 1/np.sqrt(2) if u == 0 else 1
            cv = 1/np.sqrt(2) if v == 0 else 1
            result[u, v] = 0.25 * cu * cv * s
    return result

def idct2(block):
    N = 8
    result = np.zeros((N, N))
    for x in range(N):
        for y in range(N):
            s = 0
            for u in range(N):
                for v in range(N):
                    cu = 1/np.sqrt(2) if u == 0 else 1
                    cv = 1/np.sqrt(2) if v == 0 else 1
                    s += cu * cv * block[u, v] * \
                         np.cos((2*x+1)*u*np.pi/16) * \
                         np.cos((2*y+1)*v*np.pi/16)
            result[x, y] = 0.25 * s
    return result


# QUANTIFICATION 
Q = np.array([
    [16, 11, 10, 16, 24,  40,  51,  61],
    [12, 12, 14, 19, 26,  58,  60,  55],
    [14, 13, 16, 24, 40,  57,  69,  56],
    [14, 17, 22, 29, 51,  87,  80,  62],
    [18, 22, 37, 56, 68, 109, 103,  77],
    [24, 35, 55, 64, 81, 104, 113,  92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103,  99],
])

def quantize(block, step):
    return np.round(block / (Q * step))

def dequantize(block, step):
    return block * (Q * step)


#  ZIG-ZAG 
zigzag_idx = sorted(
    [(i, j) for i in range(8) for j in range(8)],
    key=lambda x: (x[0]+x[1], -x[0] if (x[0]+x[1]) % 2 else x[0])
)

def zigzag(block):
    return [block[i][j] for i, j in zigzag_idx]

def inverse_zigzag(arr):
    block = np.zeros((8, 8))
    for k, (i, j) in enumerate(zigzag_idx):
        block[i][j] = arr[k]
    return block


#  RLE 
def rle_encode(arr):
    result = []
    count = 0
    for x in arr:
        if x == 0:
            count += 1
        else:
            result.append((count, x))
            count = 0
    result.append((0, 0))   # EOB
    return result

def rle_decode(data):
    arr = []
    for run, val in data:
        if run == 0 and val == 0:
            break
        arr.extend([0] * run)
        arr.append(val)
    arr += [0] * (64 - len(arr))
    return arr


#  HUFFMAN 
class Node:
    def __init__(self, symbol, freq):
        self.symbol = symbol
        self.freq   = freq
        self.left   = None
        self.right  = None

    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman(data):
    freq = Counter(data)
    heap = [Node(sym, f) for sym, f in freq.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        a, b = heapq.heappop(heap), heapq.heappop(heap)
        merged = Node(None, a.freq + b.freq)
        merged.left, merged.right = a, b
        heapq.heappush(heap, merged)
    return heap[0]

def build_codes(node, prefix="", code=None):   # FIX: no mutable default
    if code is None:
        code = {}
    if node is None:
        return code
    if node.symbol is not None:
        code[node.symbol] = prefix
    build_codes(node.left,  prefix + "0", code)
    build_codes(node.right, prefix + "1", code)
    return code

def huffman_encode(data, code):
    return ''.join(code[s] for s in data)

def huffman_decode(bits, root):
    result, node = [], root
    for b in bits:
        node = node.left if b == '0' else node.right
        if node.symbol is not None:
            result.append(node.symbol)
            node = root
    return result


#  COMPRESSION COMPLETE 
def compress(img, step):
    Y, Cb, Cr = rgb_to_ycbcr(img)
    Cb_sub, Cr_sub = subsample_420(Cb, Cr)
    blocks, H, W = split_blocks(Y)

    rle_blocks = []
    for block in blocks:
        d  = dct2(block - 128)
        q  = quantize(d, step)
        zz = zigzag(q)
        rle_blocks.append(rle_encode(zz))

    # aplatir en liste de strings pour Huffman
    flat = [str(x) for block in rle_blocks for x in block]

    huff_root = build_huffman(flat)
    codes      = build_codes(huff_root)
    bitstream  = huffman_encode(flat, codes)

    # حجم البيتستريم بالبايت (1 bit → 1 char هنا)
    compressed_bytes = len(bitstream) / 8

    return {
        "bitstream":        bitstream,
        "huff_root":        huff_root,
        "shape":            (H, W),
        "orig_shape":       img.shape[:2],
        "chroma":           (Cb_sub, Cr_sub),
        "compressed_bytes": compressed_bytes,
    }


# ===== DECOMPRESSION COMPLETE 
def decompress(comp, step):
    bitstream = comp["bitstream"]
    huff_root = comp["huff_root"]
    H, W      = comp["shape"]
    Cb_sub, Cr_sub = comp["chroma"]

    decoded = huffman_decode(bitstream, huff_root)

    # regrouper par bloc (separateur EOB = "(0, 0)")
    data, temp = [], []
    for x in decoded:
        temp.append(eval(x))        # str → tuple / int
        if x == "(0, 0)":
            data.append(temp)
            temp = []

    blocks = []
    for r in data:
        arr   = rle_decode(r)
        block = inverse_zigzag(arr)
        dq    = dequantize(block, step)
        rec   = idct2(dq) + 128
        blocks.append(rec)

    Y      = merge_blocks(blocks, H, W)
    orig_h = comp["orig_shape"][0]
    orig_w = comp["orig_shape"][1]
    Y_crop = Y[:orig_h, :orig_w]
    Cb, Cr = upsample_420(Cb_sub, Cr_sub, (orig_h, orig_w))
    return ycbcr_to_rgb(Y_crop, Cb, Cr)


# ===== UTILITAIRES =====
def psnr(original, reconstructed):
    mse = np.mean((original.astype(float) - reconstructed.astype(float)) ** 2)
    if mse == 0:
        return 100.0
    return 20 * math.log10(255 / math.sqrt(mse))

def image_size_kb(img_array):
    buf = io.BytesIO()
    Image.fromarray(img_array.astype(np.uint8)).save(buf, format="PNG")
    return buf.tell() / 1024

def show_image(img_array, label, size=(300, 300)):
    img    = Image.fromarray(img_array.astype(np.uint8)).resize(size, Image.LANCZOS)
    imgtk  = ImageTk.PhotoImage(img)
    label.config(image=imgtk, width=size[0], height=size[1])
    label.image = imgtk


# ===== GUI =====
class App:
    def __init__(self, root):
        self.root         = root
        self.root.title("JPEG Compression Project")
        self.root.resizable(False, False)

        self.original_img = None   # np.uint8 array
        self.comp_data    = None   # dict returned by compress()
        self.step         = 1

        self._build_ui()

    def _build_ui(self):
        # ── Barre de contrôle ─────────────────────────────────
        ctrl = tk.Frame(self.root, pady=8)
        ctrl.pack(fill="x", padx=10)

        tk.Button(ctrl, text="📂 Choisir Image", width=18,
                  command=self.load_image).pack(side="left", padx=4)

        tk.Label(ctrl, text="Pas :").pack(side="left", padx=(12, 2))
        self.step_var = tk.StringVar(value="1")
        tk.Entry(ctrl, textvariable=self.step_var, width=5).pack(side="left")

        tk.Button(ctrl, text="🗜 Compresser", width=16,
                  bg="#4CAF50", fg="white",
                  command=self.do_compress).pack(side="left", padx=8)

        tk.Button(ctrl, text="🔓 Décompresser", width=16,
                  bg="#2196F3", fg="white",
                  command=self.do_decompress).pack(side="left", padx=4)

        # ── Barre d'info ──────────────────────────────────────
        self.info_var = tk.StringVar(value="Chargez une image pour commencer.")
        tk.Label(self.root, textvariable=self.info_var,
                 font=("Courier", 10), fg="#333", pady=4).pack()

        # ── Panneaux images ───────────────────────────────────
        panels = tk.Frame(self.root)
        panels.pack(padx=10, pady=6)

        for col, title in enumerate(["Image Originale", "Image Décompressée"]):
            tk.Label(panels, text=title,
                     font=("Arial", 10, "bold")).grid(row=0, column=col, padx=12)

        self.lbl_original     = tk.Label(panels, bg="#ddd")
        self.lbl_original.grid(row=1, column=0, padx=12, pady=4)

        self.lbl_decompressed = tk.Label(panels, bg="#ddd")
        self.lbl_decompressed.grid(row=1, column=1, padx=12, pady=4)

        # ── Résultats ─────────────────────────────────────────
        self.compress_result = tk.Label(self.root, text="",
                                        fg="#2e7d32", font=("Arial", 9), pady=2)
        self.compress_result.pack()

        self.decompress_result = tk.Label(self.root, text="",
                                          fg="#1565c0", font=("Arial", 9), pady=2)
        self.decompress_result.pack()

    # ── Actions ───────────────────────────────────────────────

    def load_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tiff")])
        if not path:
            return

        self.original_img = np.array(Image.open(path).convert("RGB"))
        self.comp_data    = None

        show_image(self.original_img, self.lbl_original)
        self.lbl_decompressed.config(image="")
        self.compress_result.config(text="")
        self.decompress_result.config(text="")

        size_kb = image_size_kb(self.original_img)
        h, w    = self.original_img.shape[:2]
        self.info_var.set(
            f"Image chargée — {w}×{h} px  |  Taille : {size_kb:.1f} Ko"
        )

    def do_compress(self):
        if self.original_img is None:
            messagebox.showwarning("Attention", "Chargez d'abord une image !")
            return
        try:
            self.step = max(1, int(self.step_var.get()))
        except ValueError:
            messagebox.showerror("Erreur", "Le pas doit être un entier ≥ 1.")
            return

        self.info_var.set("Compression en cours… (peut prendre quelques secondes)")
        self.root.update()

        # ── appel correct à compress() ────────────────────────
        self.comp_data = compress(self.original_img.astype(np.float32), self.step)

        size_before       = image_size_kb(self.original_img)
        compressed_kb     = self.comp_data["compressed_bytes"] / 1024
        ratio             = size_before / compressed_kb if compressed_kb > 0 else 0

        self.compress_result.config(
            text=(f"[COMPRESSION]  Pas = {self.step}  |  "
                  f"Avant : {size_before:.1f} Ko  |  "
                  f"Après (bitstream) : {compressed_kb:.1f} Ko  |  "
                  f"Taux : {ratio:.2f}×")
        )
        self.decompress_result.config(text="")
        self.info_var.set("Compression effectuée — cliquez sur Décompresser.")

    def do_decompress(self):
        if self.comp_data is None:
            messagebox.showwarning("Attention", "Effectuez d'abord la compression !")
            return

        self.info_var.set("Décompression en cours…")
        self.root.update()

        # ── appel correct à decompress() ─────────────────────
        recon = decompress(self.comp_data, self.step)

        show_image(recon, self.lbl_decompressed)

        p           = psnr(self.original_img, recon)
        size_recon  = image_size_kb(recon)

        self.decompress_result.config(
            text=(f"[DÉCOMPRESSION]  Taille reconstruite : {size_recon:.1f} Ko  |  "
                  f"PSNR : {p:.2f} dB")
        )
        self.info_var.set("Décompression effectuée.")


# ===== MAIN =====
root = tk.Tk()
App(root)
root.mainloop()
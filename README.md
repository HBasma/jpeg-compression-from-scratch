# JPEG Compression from Scratch

## Description
This project implements a complete JPEG-like image compression pipeline from scratch using Python.

It includes all main steps of JPEG compression: color space conversion, chroma subsampling, DCT transformation, quantization, zigzag scan, run-length encoding (RLE), and Huffman coding.

The image is then decompressed and evaluated using PSNR.

---

## Features
- RGB to YCbCr conversion
- Chroma subsampling (4:2:0)
- Block splitting (8×8)
- Discrete Cosine Transform (DCT)
- Quantization using standard matrix
- Zigzag scanning
- Run-Length Encoding (RLE)
- Huffman encoding and decoding
- Full compression and decompression pipeline
- PSNR evaluation
- Graphical interface using Tkinter

---

## How it works
The compression pipeline follows these steps:

1. Convert image from RGB to YCbCr
2. Apply chroma subsampling (4:2:0)
3. Split image into 8×8 blocks
4. Apply DCT on each block
5. Quantize frequency coefficients
6. Apply zigzag scan
7. Compress data using RLE
8. Encode using Huffman coding

For decompression:
- Huffman decoding
- RLE decoding
- Inverse zigzag
- Dequantization
- IDCT reconstruction
- Conversion back to RGB

---

## Key concepts
- JPEG compression pipeline
- Frequency domain processing
- Entropy coding (Huffman)
- Lossy compression
- PSNR (Peak Signal-to-Noise Ratio)

---

## How to run

```bash
pip install numpy pillow
python jpeg-compression-from-scratch.py

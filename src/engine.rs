//! Inference engine.
//!
//! - `MappedModel` memory-maps the GGUF file. Pages load on demand, so the
//!   whole model never has to fit in resident memory at once.
//! - Q4_0 dot-product kernels: AVX2/FMA with runtime detection, plus a scalar fallback.
//! - `matvec_q4_0` parallelizes output rows across a rayon pool.
//! - `run_repl` drives token generation through llama-cli, which runs the full forward pass.

#![allow(dead_code)]

use std::fs::File;
use std::path::Path;
use std::process::Command;

use anyhow::{anyhow, bail, Context, Result};
use memmap2::Mmap;
use rayon::prelude::*;

use crate::detector::{read_gguf_meta, GgufMeta};
use crate::quantizer::find_tool;

/// Q4_0 packs 32 weights per block: a 2-byte f16 scale followed by 16 bytes of nibbles.
pub const QK4_0: usize = 32;
pub const Q4_0_BLOCK_BYTES: usize = 18;

/// Q8_0 packs 32 weights per block: a 2-byte f16 scale followed by 32 signed bytes.
pub const QK8_0: usize = 32;
pub const Q8_0_BLOCK_BYTES: usize = 34;

/// A read-only, memory-mapped GGUF file.
pub struct MappedModel {
    mmap: Mmap,
    pub meta: GgufMeta,
}

impl MappedModel {
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
        // SAFETY: the mapping is read-only. Undefined behavior would require
        // another process to truncate the file while it is mapped, which is
        // outside this program's control; users should not edit models in place.
        let mmap = unsafe { Mmap::map(&file) }
            .with_context(|| format!("memory-mapping {}", path.display()))?;
        let meta = read_gguf_meta(path)?;
        Ok(Self { mmap, meta })
    }

    pub fn bytes(&self) -> &[u8] {
        &self.mmap
    }
}

/// Number of worker threads to use when the user does not specify one.
/// `available_parallelism` reports logical CPUs; SMT siblings are included.
pub fn default_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

/// Configure the global rayon pool. Returns the thread count in use.
pub fn init_thread_pool(threads: Option<usize>) -> Result<usize> {
    let n = threads.unwrap_or_else(default_threads).max(1);
    rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build_global()
        .map_err(|e| anyhow!("failed to initialize thread pool: {e}"))?;
    Ok(n)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kernel {
    #[cfg(target_arch = "x86_64")]
    Avx2Fma,
    #[cfg(target_arch = "aarch64")]
    Neon,
    Scalar,
}

/// Pick the fastest kernel the running CPU supports.
pub fn detect_kernel() -> Kernel {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return Kernel::Avx2Fma;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return Kernel::Neon;
    }
    Kernel::Scalar
}

/// Convert an IEEE 754 half-precision value (as stored in GGUF) to f32.
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;

    let out = if exp == 0 {
        if mant == 0 {
            sign << 31
        } else {
            // Subnormal: normalize the mantissa and adjust the exponent.
            let mut e: i32 = -14;
            let mut m = mant;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            let m = m & 0x3FF;
            (sign << 31) | (((e + 127) as u32) << 23) | (m << 13)
        }
    } else if exp == 0x1F {
        (sign << 31) | (0xFF << 23) | (mant << 13)
    } else {
        (sign << 31) | ((exp + 112) << 23) | (mant << 13)
    };
    f32::from_bits(out)
}

fn dot_block_scalar(d: f32, qs: &[u8], x: &[f32]) -> f32 {
    let mut sum = 0f32;
    for j in 0..16 {
        let lo = (qs[j] & 0x0F) as i32 - 8;
        let hi = (qs[j] >> 4) as i32 - 8;
        sum += lo as f32 * x[j] + hi as f32 * x[j + 16];
    }
    d * sum
}

/// AVX2/FMA kernel for one Q4_0 block (32 weights against 32 activations).
///
/// # Safety
/// Caller must ensure the CPU supports AVX2 and FMA, `qs.len() == 16`, and
/// `x.len() == 32`. `detect_kernel` guarantees the first condition.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_block_avx2(d: f32, qs: &[u8], x: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    debug_assert!(qs.len() == 16 && x.len() == QK4_0);

    let raw = _mm_loadu_si128(qs.as_ptr() as *const __m128i);
    let mask = _mm_set1_epi8(0x0F);
    // Low nibbles map to x[0..16], high nibbles to x[16..32].
    let lo = _mm_and_si128(raw, mask);
    let hi = _mm_and_si128(_mm_srli_epi16::<4>(raw), mask);

    let bias = _mm256_set1_epi32(8);
    let mut acc = _mm256_setzero_ps();

    for (nibbles, base) in [(lo, 0usize), (hi, 16usize)] {
        // Each pass widens 8 bytes to 8 lanes; two passes cover 16 nibbles.
        for half in 0..2usize {
            let bytes = if half == 0 {
                nibbles
            } else {
                _mm_srli_si128::<8>(nibbles)
            };
            let ints = _mm256_sub_epi32(_mm256_cvtepu8_epi32(bytes), bias);
            let w = _mm256_cvtepi32_ps(ints);
            let a = _mm256_loadu_ps(x.as_ptr().add(base + half * 8));
            acc = _mm256_fmadd_ps(w, a, acc);
        }
    }

    // Horizontal sum of 8 lanes.
    let low = _mm256_castps256_ps128(acc);
    let high = _mm256_extractf128_ps::<1>(acc);
    let s = _mm_add_ps(low, high);
    let s = _mm_hadd_ps(s, s);
    let s = _mm_hadd_ps(s, s);
    d * _mm_cvtss_f32(s)
}

/// ARM NEON kernel for one Q4_0 block (32 weights against 32 activations).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_block_neon(d: f32, qs: &[u8], x: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    debug_assert!(qs.len() == 16 && x.len() == QK4_0);

    let raw = vld1q_u8(qs.as_ptr());
    let mask = vdupq_n_u8(0x0F);
    let lo_u8 = vandq_u8(raw, mask);
    let hi_u8 = vandq_u8(vshrq_n_u8::<4>(raw), mask);

    let bias = vdupq_n_s8(8);
    let lo_s8 = vsubq_s8(vreinterpretq_s8_u8(lo_u8), bias);
    let hi_s8 = vsubq_s8(vreinterpretq_s8_u8(hi_u8), bias);

    let mut acc = vdupq_n_f32(0.0);

    let lo_s16_l = vmovl_s8(vget_low_s8(lo_s8));
    let lo_s16_h = vmovl_high_s8(lo_s8);
    let f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo_s16_l)));
    let f1 = vcvtq_f32_s32(vmovl_high_s16(lo_s16_l));
    let f2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo_s16_h)));
    let f3 = vcvtq_f32_s32(vmovl_high_s16(lo_s16_h));

    acc = vfmaq_f32(acc, f0, vld1q_f32(x.as_ptr().add(0)));
    acc = vfmaq_f32(acc, f1, vld1q_f32(x.as_ptr().add(4)));
    acc = vfmaq_f32(acc, f2, vld1q_f32(x.as_ptr().add(8)));
    acc = vfmaq_f32(acc, f3, vld1q_f32(x.as_ptr().add(12)));

    let hi_s16_l = vmovl_s8(vget_low_s8(hi_s8));
    let hi_s16_h = vmovl_high_s8(hi_s8);
    let f4 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi_s16_l)));
    let f5 = vcvtq_f32_s32(vmovl_high_s16(hi_s16_l));
    let f6 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi_s16_h)));
    let f7 = vcvtq_f32_s32(vmovl_high_s16(hi_s16_h));

    acc = vfmaq_f32(acc, f4, vld1q_f32(x.as_ptr().add(16)));
    acc = vfmaq_f32(acc, f5, vld1q_f32(x.as_ptr().add(20)));
    acc = vfmaq_f32(acc, f6, vld1q_f32(x.as_ptr().add(24)));
    acc = vfmaq_f32(acc, f7, vld1q_f32(x.as_ptr().add(28)));

    let sum = vaddvq_f32(acc);
    d * sum
}

/// Dot product of one Q4_0-encoded row with an f32 activation vector.
pub fn dot_q4_0_row(row: &[u8], x: &[f32], kernel: Kernel) -> f32 {
    let mut acc = 0f32;
    for (block, xb) in row
        .chunks_exact(Q4_0_BLOCK_BYTES)
        .zip(x.chunks_exact(QK4_0))
    {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let qs = &block[2..];
        acc += match kernel {
            #[cfg(target_arch = "x86_64")]
            // SAFETY: `Kernel::Avx2Fma` is only produced by `detect_kernel`
            // after confirming AVX2 and FMA support.
            Kernel::Avx2Fma => unsafe { dot_block_avx2(d, qs, xb) },
            #[cfg(target_arch = "aarch64")]
            Kernel::Neon => unsafe { dot_block_neon(d, qs, xb) },
            Kernel::Scalar => dot_block_scalar(d, qs, xb),
        };
    }
    acc
}

/// `y = W · x`, where `W` is `rows × cols` in row-major Q4_0 layout.
/// Output rows are computed in parallel on the rayon pool.
pub fn matvec_q4_0(w: &[u8], rows: usize, cols: usize, x: &[f32], y: &mut [f32]) -> Result<()> {
    if cols % QK4_0 != 0 {
        bail!("column count {cols} is not a multiple of {QK4_0}");
    }
    let row_bytes = cols / QK4_0 * Q4_0_BLOCK_BYTES;
    let expected_bytes = rows * row_bytes;
    if w.len() < expected_bytes {
        bail!(
            "weight buffer too short: got {}, expected at least {expected_bytes}",
            w.len()
        );
    }
    if x.len() < cols {
        bail!(
            "input vector too short: got {}, expected at least {cols}",
            x.len()
        );
    }
    if y.len() < rows {
        bail!(
            "output vector too short: got {}, expected at least {rows}",
            y.len()
        );
    }

    let kernel = detect_kernel();
    y[..rows].par_iter_mut().enumerate().for_each(|(r, out)| {
        let row_slice = &w[r * row_bytes..(r + 1) * row_bytes];
        *out = dot_q4_0_row(row_slice, &x[..cols], kernel);
    });

    Ok(())
}

fn dot_block_q8_0_scalar(d: f32, qs: &[i8], x: &[f32]) -> f32 {
    let mut sum = 0f32;
    for j in 0..32 {
        sum += qs[j] as f32 * x[j];
    }
    d * sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_block_q8_0_avx2(d: f32, qs: &[i8], x: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    debug_assert!(qs.len() == 32 && x.len() == QK8_0);

    let raw = _mm256_loadu_si256(qs.as_ptr() as *const __m256i);
    let low128 = _mm256_castsi256_si128(raw);
    let high128 = _mm256_extracti128_si256::<1>(raw);

    let i0 = _mm256_cvtepi8_epi32(low128);
    let i1 = _mm256_cvtepi8_epi32(_mm_srli_si128::<8>(low128));
    let i2 = _mm256_cvtepi8_epi32(high128);
    let i3 = _mm256_cvtepi8_epi32(_mm_srli_si128::<8>(high128));

    let mut acc = _mm256_setzero_ps();
    acc = _mm256_fmadd_ps(
        _mm256_cvtepi32_ps(i0),
        _mm256_loadu_ps(x.as_ptr().add(0)),
        acc,
    );
    acc = _mm256_fmadd_ps(
        _mm256_cvtepi32_ps(i1),
        _mm256_loadu_ps(x.as_ptr().add(8)),
        acc,
    );
    acc = _mm256_fmadd_ps(
        _mm256_cvtepi32_ps(i2),
        _mm256_loadu_ps(x.as_ptr().add(16)),
        acc,
    );
    acc = _mm256_fmadd_ps(
        _mm256_cvtepi32_ps(i3),
        _mm256_loadu_ps(x.as_ptr().add(24)),
        acc,
    );

    let low = _mm256_castps256_ps128(acc);
    let high = _mm256_extractf128_ps::<1>(acc);
    let s = _mm_add_ps(low, high);
    let s = _mm_hadd_ps(s, s);
    let s = _mm_hadd_ps(s, s);
    d * _mm_cvtss_f32(s)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_block_q8_0_neon(d: f32, qs: &[i8], x: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    debug_assert!(qs.len() == 32 && x.len() == QK8_0);

    let raw0 = vld1q_s8(qs.as_ptr());
    let raw1 = vld1q_s8(qs.as_ptr().add(16));

    let mut acc = vdupq_n_f32(0.0);

    let s16_0 = vmovl_s8(vget_low_s8(raw0));
    let s16_1 = vmovl_high_s8(raw0);
    let f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(s16_0)));
    let f1 = vcvtq_f32_s32(vmovl_high_s16(s16_0));
    let f2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(s16_1)));
    let f3 = vcvtq_f32_s32(vmovl_high_s16(s16_1));

    acc = vfmaq_f32(acc, f0, vld1q_f32(x.as_ptr().add(0)));
    acc = vfmaq_f32(acc, f1, vld1q_f32(x.as_ptr().add(4)));
    acc = vfmaq_f32(acc, f2, vld1q_f32(x.as_ptr().add(8)));
    acc = vfmaq_f32(acc, f3, vld1q_f32(x.as_ptr().add(12)));

    let s16_2 = vmovl_s8(vget_low_s8(raw1));
    let s16_3 = vmovl_high_s8(raw1);
    let f4 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(s16_2)));
    let f5 = vcvtq_f32_s32(vmovl_high_s16(s16_2));
    let f6 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(s16_3)));
    let f7 = vcvtq_f32_s32(vmovl_high_s16(s16_3));

    acc = vfmaq_f32(acc, f4, vld1q_f32(x.as_ptr().add(16)));
    acc = vfmaq_f32(acc, f5, vld1q_f32(x.as_ptr().add(20)));
    acc = vfmaq_f32(acc, f6, vld1q_f32(x.as_ptr().add(24)));
    acc = vfmaq_f32(acc, f7, vld1q_f32(x.as_ptr().add(28)));

    let sum = vaddvq_f32(acc);
    d * sum
}

/// Dot product of one Q8_0-encoded row with an f32 activation vector.
pub fn dot_q8_0_row(row: &[u8], x: &[f32], kernel: Kernel) -> f32 {
    let mut acc = 0f32;
    for (block, xb) in row
        .chunks_exact(Q8_0_BLOCK_BYTES)
        .zip(x.chunks_exact(QK8_0))
    {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let qs: &[i8] = unsafe { std::slice::from_raw_parts(block[2..].as_ptr() as *const i8, 32) };
        acc += match kernel {
            #[cfg(target_arch = "x86_64")]
            Kernel::Avx2Fma => unsafe { dot_block_q8_0_avx2(d, qs, xb) },
            #[cfg(target_arch = "aarch64")]
            Kernel::Neon => unsafe { dot_block_q8_0_neon(d, qs, xb) },
            Kernel::Scalar => dot_block_q8_0_scalar(d, qs, xb),
        };
    }
    acc
}

/// `y = W · x`, where `W` is `rows × cols` in row-major Q8_0 layout.
/// Output rows are computed in parallel on the rayon pool.
pub fn matvec_q8_0(w: &[u8], rows: usize, cols: usize, x: &[f32], y: &mut [f32]) -> Result<()> {
    if cols % QK8_0 != 0 {
        bail!("column count {cols} is not a multiple of {QK8_0}");
    }
    let row_bytes = cols / QK8_0 * Q8_0_BLOCK_BYTES;
    let expected_bytes = rows * row_bytes;
    if w.len() < expected_bytes {
        bail!(
            "weight buffer too short: got {}, expected at least {expected_bytes}",
            w.len()
        );
    }
    if x.len() < cols {
        bail!(
            "input vector too short: got {}, expected at least {cols}",
            x.len()
        );
    }
    if y.len() < rows {
        bail!(
            "output vector too short: got {}, expected at least {rows}",
            y.len()
        );
    }

    let kernel = detect_kernel();
    y[..rows].par_iter_mut().enumerate().for_each(|(r, out)| {
        let row_slice = &w[r * row_bytes..(r + 1) * row_bytes];
        *out = dot_q8_0_row(row_slice, &x[..cols], kernel);
    });

    Ok(())
}

/// Start an interactive session. Token generation is delegated to llama-cli,
/// which runs the full transformer forward pass. Kernels in this module serve
/// as the native path for benchmarking and for future in-process inference.
pub fn run_repl(model: &Path, threads: usize) -> Result<()> {
    let mapped = MappedModel::open(model)?;
    let arch = mapped
        .meta
        .architecture
        .clone()
        .unwrap_or_else(|| "unknown".into());
    let quant = mapped.meta.quant_name();
    let version = mapped.meta.version;
    let tensors = mapped.meta.tensor_count;
    println!(
        "loaded {} (gguf v{version}, arch: {arch}, quant: {quant}, tensors: {tensors}, kernel: {:?}, threads: {threads})",
        model.display(),
        detect_kernel()
    );
    // Release our mapping before the child process loads its own copy.
    drop(mapped);

    let cli = find_tool("llama-cli", "LLM_COMPRESS_CLI_BIN")?;
    let status = Command::new(&cli)
        .arg("-m")
        .arg(model)
        .arg("-t")
        .arg(threads.to_string())
        .arg("-cnv") // interactive conversation mode
        .status()
        .with_context(|| format!("launching {}", cli.display()))?;

    if !status.success() {
        bail!("llama-cli exited with {status}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_known_values() {
        assert_eq!(f16_to_f32(0x3C00), 1.0);
        assert_eq!(f16_to_f32(0xC000), -2.0);
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24)); // smallest subnormal
    }

    #[test]
    fn matvec_matches_scalar_reference() {
        let rows = 5;
        let cols = 96;
        let row_bytes = cols / QK4_0 * Q4_0_BLOCK_BYTES;

        let mut w = vec![0u8; rows * row_bytes];
        for block in w.chunks_exact_mut(Q4_0_BLOCK_BYTES) {
            block[0..2].copy_from_slice(&0x3C00u16.to_le_bytes()); // scale = 1.0
            for (j, q) in block[2..].iter_mut().enumerate() {
                *q = (j * 37 + 7) as u8;
            }
        }
        let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.37).sin()).collect();

        let mut y = vec![0f32; rows];
        matvec_q4_0(&w, rows, cols, &x, &mut y).unwrap();

        for r in 0..rows {
            let expected = dot_q4_0_row(&w[r * row_bytes..(r + 1) * row_bytes], &x, Kernel::Scalar);
            assert!(
                (y[r] - expected).abs() < 1e-3,
                "row {r}: got {}, expected {expected}",
                y[r]
            );
        }
    }

    #[test]
    fn matvec_q8_0_matches_scalar_reference() {
        let rows = 4;
        let cols = 64;
        let row_bytes = cols / QK8_0 * Q8_0_BLOCK_BYTES;

        let mut w = vec![0u8; rows * row_bytes];
        for block in w.chunks_exact_mut(Q8_0_BLOCK_BYTES) {
            block[0..2].copy_from_slice(&0x3C00u16.to_le_bytes()); // scale = 1.0
            for (j, q) in block[2..].iter_mut().enumerate() {
                *q = ((j as i32 * 23 - 50) % 127) as u8;
            }
        }
        let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.25).cos()).collect();

        let mut y = vec![0f32; rows];
        matvec_q8_0(&w, rows, cols, &x, &mut y).unwrap();

        for r in 0..rows {
            let expected = dot_q8_0_row(&w[r * row_bytes..(r + 1) * row_bytes], &x, Kernel::Scalar);
            assert!(
                (y[r] - expected).abs() < 1e-3,
                "row {r}: got {}, expected {expected}",
                y[r]
            );
        }
    }
}

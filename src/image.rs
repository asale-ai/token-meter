//! Image tokens, from the image's own dimensions.
//!
//! Every provider prices an image by its geometry, and the three do it
//! differently — Claude by area, OpenAI by 512-pixel tiles, Gemini by 768-pixel
//! tiles with a flat rate for small images. A single flat constant for "an
//! image" is therefore wrong by more than an order of magnitude at the top end: a
//! 1024×1024 screenshot is ~1400 tokens on Claude and ~85 on the flat-rate
//! assumption that a naive estimator makes.
//!
//! Dimensions come from the file header, which is the first few dozen bytes of a
//! PNG or GIF and a short scan into a JPEG. This module parses those headers
//! directly — no image decoding, no pixel buffers, no dependencies — and gives up
//! rather than guessing when the format is unrecognised.

/// Pixel dimensions of an image.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageDims {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
}

/// The token cost assumed for an image whose dimensions could not be read.
///
/// The low-detail OpenAI rate, and the floor for every provider — deliberately
/// the smallest defensible number, so an unreadable header under-counts rather
/// than inventing a large charge.
pub const UNKNOWN_IMAGE_TOKENS: i64 = 85;

impl ImageDims {
    /// Read dimensions from an image file's leading bytes.
    ///
    /// Supports PNG, JPEG, GIF and WebP. Returns `None` for anything else, or
    /// for a truncated header.
    #[must_use]
    pub fn parse(bytes: &[u8]) -> Option<Self> {
        png(bytes).or_else(|| gif(bytes)).or_else(|| webp(bytes)).or_else(|| jpeg(bytes))
    }

    /// Read dimensions from base64 data, with or without a `data:` URL prefix.
    ///
    /// Only the leading bytes are decoded — enough to reach a JPEG's frame
    /// header — so this stays cheap on a multi-megabyte payload.
    #[must_use]
    pub fn parse_base64(b64: &str) -> Option<Self> {
        let payload = match b64.find("base64,") {
            Some(i) => &b64[i + "base64,".len()..],
            None => b64,
        };
        let head = decode_base64_prefix(payload, HEADER_SCAN_BYTES);
        Self::parse(&head)
    }

    /// Tokens Claude bills for this image: roughly area over 750.
    ///
    /// Images above 1568 pixels on their long edge are resized down by the API
    /// before counting, so that cap is applied here too.
    #[must_use]
    pub fn claude_tokens(self) -> i64 {
        const MAX_EDGE: f64 = 1568.0;
        let (w, h) = (f64::from(self.width), f64::from(self.height));
        if w <= 0.0 || h <= 0.0 {
            return UNKNOWN_IMAGE_TOKENS;
        }
        let scale = (MAX_EDGE / w.max(h)).min(1.0);
        let (w, h) = (w * scale, h * scale);
        ((w * h / 750.0).ceil() as i64).max(1)
    }

    /// Tokens the GPT family bills at `detail: "high"`: a base charge plus 170
    /// per 512×512 tile, after the image is fitted to 2048×2048 and its short
    /// edge scaled to 768.
    #[must_use]
    pub fn openai_tokens(self) -> i64 {
        const BASE: i64 = 85;
        const PER_TILE: i64 = 170;
        let (mut w, mut h) = (f64::from(self.width), f64::from(self.height));
        if w <= 0.0 || h <= 0.0 {
            return BASE;
        }
        // Fit inside 2048×2048.
        let fit = (2048.0 / w.max(h)).min(1.0);
        w *= fit;
        h *= fit;
        // Scale the short edge to 768.
        let short = (768.0 / w.min(h)).min(1.0);
        w *= short;
        h *= short;
        let tiles = (w / 512.0).ceil() * (h / 512.0).ceil();
        BASE + PER_TILE * (tiles as i64).max(1)
    }

    /// Tokens Gemini bills: a flat rate up to 384×384, then 258 per 768×768
    /// tile.
    #[must_use]
    pub fn gemini_tokens(self) -> i64 {
        const PER_TILE: i64 = 258;
        let (w, h) = (self.width, self.height);
        if w == 0 || h == 0 {
            return PER_TILE;
        }
        if w <= 384 && h <= 384 {
            return PER_TILE;
        }
        let tiles_w = w.div_ceil(768).max(1) as i64;
        let tiles_h = h.div_ceil(768).max(1) as i64;
        PER_TILE * tiles_w * tiles_h
    }

    /// Tokens for a given family.
    #[must_use]
    pub fn tokens_for(self, family: crate::Family) -> i64 {
        match family {
            crate::Family::Claude => self.claude_tokens(),
            crate::Family::Gpt => self.openai_tokens(),
            crate::Family::Gemini => self.gemini_tokens(),
            // Unknown provider: the most common convention, and the cheapest of
            // the three at typical sizes.
            crate::Family::Other => self.openai_tokens(),
        }
    }
}

/// How far into a file to decode when hunting for a header. A JPEG's frame
/// marker sits behind whatever EXIF and colour-profile segments came first,
/// which is usually a few kilobytes at most.
const HEADER_SCAN_BYTES: usize = 8 * 1024;

// ── header parsers ──────────────────────────────────────────────────

fn be_u32(b: &[u8]) -> u32 {
    u32::from_be_bytes([b[0], b[1], b[2], b[3]])
}

fn png(b: &[u8]) -> Option<ImageDims> {
    // 8-byte signature, then an IHDR chunk whose payload opens with the size.
    if b.len() < 24 || &b[0..8] != b"\x89PNG\r\n\x1a\n" || &b[12..16] != b"IHDR" {
        return None;
    }
    Some(ImageDims { width: be_u32(&b[16..20]), height: be_u32(&b[20..24]) })
}

fn gif(b: &[u8]) -> Option<ImageDims> {
    if b.len() < 10 || (&b[0..6] != b"GIF87a" && &b[0..6] != b"GIF89a") {
        return None;
    }
    // Logical screen descriptor: little-endian, unlike everything else here.
    Some(ImageDims {
        width: u32::from(u16::from_le_bytes([b[6], b[7]])),
        height: u32::from(u16::from_le_bytes([b[8], b[9]])),
    })
}

fn webp(b: &[u8]) -> Option<ImageDims> {
    if b.len() < 30 || &b[0..4] != b"RIFF" || &b[8..12] != b"WEBP" {
        return None;
    }
    match &b[12..16] {
        // Lossy: a 3-byte sync code, then 14-bit dimensions.
        b"VP8 " => {
            if b.len() < 30 || b[23..26] != [0x9d, 0x01, 0x2a] {
                return None;
            }
            Some(ImageDims {
                width: u32::from(u16::from_le_bytes([b[26], b[27]]) & 0x3fff),
                height: u32::from(u16::from_le_bytes([b[28], b[29]]) & 0x3fff),
            })
        }
        // Lossless: a signature byte, then 14-bit (dimension - 1) pairs packed
        // across four bytes.
        b"VP8L" => {
            if b.len() < 25 || b[20] != 0x2f {
                return None;
            }
            let bits = u32::from_le_bytes([b[21], b[22], b[23], b[24]]);
            Some(ImageDims {
                width: (bits & 0x3fff) + 1,
                height: ((bits >> 14) & 0x3fff) + 1,
            })
        }
        // Extended: 24-bit canvas dimensions, also stored one less than actual.
        b"VP8X" => {
            if b.len() < 30 {
                return None;
            }
            let w = u32::from_le_bytes([b[24], b[25], b[26], 0]) + 1;
            let h = u32::from_le_bytes([b[27], b[28], b[29], 0]) + 1;
            Some(ImageDims { width: w, height: h })
        }
        _ => None,
    }
}

fn jpeg(b: &[u8]) -> Option<ImageDims> {
    if b.len() < 4 || b[0] != 0xFF || b[1] != 0xD8 {
        return None;
    }
    let mut i = 2usize;
    while i + 3 < b.len() {
        // Segments are introduced by one or more 0xFF fill bytes.
        if b[i] != 0xFF {
            i += 1;
            continue;
        }
        let marker = b[i + 1];
        i += 2;
        match marker {
            // Padding and standalone markers carry no length field.
            0xFF | 0x01 | 0xD0..=0xD9 => continue,
            // Start-of-frame variants — every one of them opens the same way.
            // 0xC4 (DHT), 0xC8 and 0xCC (JPG/DAC) are not frames.
            0xC0..=0xC3 | 0xC5..=0xC7 | 0xC9..=0xCB | 0xCD..=0xCF => {
                // Reaches b[i + 6]; a frame header that stops short of the
                // width field is truncated, not a zero-width image.
                if i + 6 >= b.len() {
                    return None;
                }
                // length(2) precision(1) height(2) width(2)
                return Some(ImageDims {
                    width: u32::from(u16::from_be_bytes([b[i + 5], b[i + 6]])),
                    height: u32::from(u16::from_be_bytes([b[i + 3], b[i + 4]])),
                });
            }
            _ => {
                if i + 1 >= b.len() {
                    return None;
                }
                let len = usize::from(u16::from_be_bytes([b[i], b[i + 1]]));
                if len < 2 {
                    return None;
                }
                i += len;
            }
        }
    }
    None
}

// ── base64 ──────────────────────────────────────────────────────────

/// Decode at most `max_bytes` of standard base64.
///
/// Hand-rolled to keep this crate dependency-free for what amounts to twenty
/// lines of table lookup. Tolerates whitespace and both alphabets, stops at the
/// first byte it cannot read, and never allocates more than the cap.
fn decode_base64_prefix(s: &str, max_bytes: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(max_bytes.min(s.len() / 4 * 3 + 3));
    let mut acc: u32 = 0;
    let mut bits: u32 = 0;

    for c in s.bytes() {
        let v = match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' | b'-' => 62,
            b'/' | b'_' => 63,
            b'=' => break,
            b'\n' | b'\r' | b' ' | b'\t' => continue,
            _ => break,
        };
        acc = (acc << 6) | u32::from(v);
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push(((acc >> bits) & 0xFF) as u8);
            if out.len() >= max_bytes {
                break;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Family;

    fn png_bytes(w: u32, h: u32) -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(b"\x89PNG\r\n\x1a\n");
        v.extend_from_slice(&13u32.to_be_bytes());
        v.extend_from_slice(b"IHDR");
        v.extend_from_slice(&w.to_be_bytes());
        v.extend_from_slice(&h.to_be_bytes());
        v.extend_from_slice(&[8, 6, 0, 0, 0]);
        v
    }

    fn jpeg_bytes(w: u16, h: u16) -> Vec<u8> {
        let mut v = vec![0xFF, 0xD8];
        // An APP0 segment first, so the scan has to skip something real.
        v.extend_from_slice(&[0xFF, 0xE0, 0x00, 0x10]);
        v.extend_from_slice(b"JFIF\0\x01\x02\0\0\x01\0\x01\0\0");
        // SOF0
        v.extend_from_slice(&[0xFF, 0xC0, 0x00, 0x11, 0x08]);
        v.extend_from_slice(&h.to_be_bytes());
        v.extend_from_slice(&w.to_be_bytes());
        v
    }

    #[test]
    fn reads_png_dimensions() {
        let d = ImageDims::parse(&png_bytes(1024, 768)).unwrap();
        assert_eq!(d, ImageDims { width: 1024, height: 768 });
    }

    #[test]
    fn reads_jpeg_dimensions_past_a_leading_segment() {
        let d = ImageDims::parse(&jpeg_bytes(1920, 1080)).unwrap();
        assert_eq!(d, ImageDims { width: 1920, height: 1080 });
    }

    #[test]
    fn reads_gif_dimensions_little_endian() {
        let mut v = b"GIF89a".to_vec();
        v.extend_from_slice(&640u16.to_le_bytes());
        v.extend_from_slice(&480u16.to_le_bytes());
        let d = ImageDims::parse(&v).unwrap();
        assert_eq!(d, ImageDims { width: 640, height: 480 });
    }

    #[test]
    fn unrecognised_bytes_give_up_rather_than_guess() {
        assert!(ImageDims::parse(b"not an image at all, really").is_none());
        assert!(ImageDims::parse(&[]).is_none());
        // A truncated PNG header is not a 0x0 image.
        assert!(ImageDims::parse(&png_bytes(10, 10)[..12]).is_none());
    }

    #[test]
    fn decodes_base64_with_and_without_a_data_url_prefix() {
        let raw = png_bytes(800, 600);
        let b64 = to_base64(&raw);
        assert_eq!(ImageDims::parse_base64(&b64).unwrap().width, 800);

        let url = format!("data:image/png;base64,{b64}");
        assert_eq!(ImageDims::parse_base64(&url).unwrap().height, 600);
    }

    #[test]
    fn base64_decoding_stops_at_the_cap() {
        let big = to_base64(&vec![0u8; 100_000]);
        let out = decode_base64_prefix(&big, 64);
        assert_eq!(out.len(), 64, "never decodes the whole payload to read a header");
    }

    #[test]
    fn a_large_image_costs_far_more_than_the_flat_constant() {
        let d = ImageDims { width: 1024, height: 1024 };
        // The failure this module exists to fix: 85 tokens for a screenshot
        // that Claude bills at well over a thousand.
        assert!(d.claude_tokens() > 1_000, "got {}", d.claude_tokens());
        assert!(d.openai_tokens() > UNKNOWN_IMAGE_TOKENS);
        assert!(d.gemini_tokens() > UNKNOWN_IMAGE_TOKENS);
    }

    #[test]
    fn claude_caps_the_long_edge_before_counting() {
        let huge = ImageDims { width: 8000, height: 8000 };
        let capped = ImageDims { width: 1568, height: 1568 };
        assert_eq!(huge.claude_tokens(), capped.claude_tokens());
    }

    #[test]
    fn gemini_charges_one_tile_for_a_thumbnail() {
        assert_eq!(ImageDims { width: 200, height: 200 }.gemini_tokens(), 258);
        assert_eq!(ImageDims { width: 384, height: 384 }.gemini_tokens(), 258);
        assert!(ImageDims { width: 385, height: 385 }.gemini_tokens() >= 258);
    }

    #[test]
    fn every_family_has_an_opinion_and_a_small_image_is_cheap() {
        let small = ImageDims { width: 64, height: 64 };
        for f in [Family::Claude, Family::Gpt, Family::Gemini, Family::Other] {
            let t = small.tokens_for(f);
            assert!(t > 0 && t < 1_000, "{f:?} gave {t}");
        }
    }

    fn to_base64(bytes: &[u8]) -> String {
        const ALPHA: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut out = String::new();
        for chunk in bytes.chunks(3) {
            let b = [chunk[0], *chunk.get(1).unwrap_or(&0), *chunk.get(2).unwrap_or(&0)];
            let n = (u32::from(b[0]) << 16) | (u32::from(b[1]) << 8) | u32::from(b[2]);
            for i in 0..4 {
                if i <= chunk.len() {
                    out.push(ALPHA[((n >> (18 - 6 * i)) & 0x3F) as usize] as char);
                } else {
                    out.push('=');
                }
            }
        }
        out
    }
}

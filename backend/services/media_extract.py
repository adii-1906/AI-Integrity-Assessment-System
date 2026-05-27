"""
Media Text Extraction Service
Extracts text from images and videos for AICES evaluation.

- Images: PIL + Tesseract OCR
- Videos: ffmpeg frame extraction → OCR each frame → combine unique text
"""

import base64
import io
import os
import re
import subprocess
import tempfile
from typing import Dict, Any

from PIL import Image, ImageFilter, ImageEnhance
import pytesseract


class MediaExtractor:
    """
    Extracts readable text from images and videos.
    No API calls — fully local using Tesseract + ffmpeg.
    """

    SUPPORTED_IMAGE_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp", "image/gif", "image/bmp"}
    SUPPORTED_VIDEO_TYPES = {"video/mp4", "video/mpeg", "video/quicktime", "video/x-msvideo", "video/webm"}

    MAX_IMAGE_SIZE_MB = 10
    MAX_VIDEO_SIZE_MB = 50
    MAX_VIDEO_FRAMES = 8   # max frames to OCR per video
    MIN_TEXT_LENGTH = 30   # minimum chars to consider extraction successful

    def extract(self, file_data: str, mime_type: str, filename: str = "") -> Dict[str, Any]:
        """
        Main entry point. Takes base64-encoded file data.

        Returns:
            {
                "success": bool,
                "text": str,
                "method": str,
                "media_type": "image" | "video",
                "details": {...},
                "error": str (only if success=False)
            }
        """
        try:
            raw_bytes = base64.b64decode(file_data)
            size_mb = len(raw_bytes) / (1024 * 1024)

            mime_lower = mime_type.lower()

            if mime_lower in self.SUPPORTED_IMAGE_TYPES:
                if size_mb > self.MAX_IMAGE_SIZE_MB:
                    return {"success": False, "error": f"Image too large ({size_mb:.1f}MB). Max 10MB."}
                return self._extract_from_image(raw_bytes, filename)

            elif mime_lower in self.SUPPORTED_VIDEO_TYPES:
                if size_mb > self.MAX_VIDEO_SIZE_MB:
                    return {"success": False, "error": f"Video too large ({size_mb:.1f}MB). Max 50MB."}
                return self._extract_from_video(raw_bytes, filename)

            else:
                return {"success": False, "error": f"Unsupported file type: {mime_type}. Supported: JPEG, PNG, WebP, MP4, MOV, AVI, WebM"}

        except Exception as e:
            return {"success": False, "error": f"Extraction failed: {str(e)}"}

    # ── IMAGE ──────────────────────────────────────────────────────────

    def _extract_from_image(self, raw_bytes: bytes, filename: str) -> Dict[str, Any]:
        """OCR a single image using Tesseract."""
        try:
            img = Image.open(io.BytesIO(raw_bytes))

            # Convert to RGB if needed (handles RGBA, palette, etc)
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")

            # Preprocessing: resize if very small, enhance contrast
            w, h = img.size
            if w < 300 or h < 300:
                scale = max(300 / w, 300 / h)
                img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

            # Try multiple OCR configs and pick best result
            results = []

            # Config 1: default
            text1 = pytesseract.image_to_string(img, config='--psm 3')
            results.append(text1.strip())

            # Config 2: enhanced contrast
            enhanced = ImageEnhance.Contrast(img).enhance(2.0)
            text2 = pytesseract.image_to_string(enhanced, config='--psm 3')
            results.append(text2.strip())

            # Config 3: grayscale + sharpen
            gray = img.convert('L').filter(ImageFilter.SHARPEN)
            text3 = pytesseract.image_to_string(gray, config='--psm 6')
            results.append(text3.strip())

            # Pick the longest result
            best_text = max(results, key=len)
            best_text = self._clean_text(best_text)

            if len(best_text) < self.MIN_TEXT_LENGTH:
                return {
                    "success": False,
                    "error": "Could not extract sufficient text from image. The image may not contain readable text, or text may be too small/blurry.",
                    "media_type": "image",
                    "details": {"extracted_chars": len(best_text), "raw_preview": best_text[:100]}
                }

            return {
                "success": True,
                "text": best_text,
                "method": "Tesseract OCR (v5)",
                "media_type": "image",
                "details": {
                    "image_size": f"{w}×{h}",
                    "chars_extracted": len(best_text),
                    "words_extracted": len(best_text.split()),
                    "filename": filename
                }
            }

        except Exception as e:
            return {"success": False, "error": f"Image OCR failed: {str(e)}", "media_type": "image"}

    # ── VIDEO ──────────────────────────────────────────────────────────

    def _extract_from_video(self, raw_bytes: bytes, filename: str) -> Dict[str, Any]:
        """Extract frames from video, OCR each, combine unique text."""
        with tempfile.TemporaryDirectory() as tmp:
            # Write video to temp file
            ext = self._guess_video_ext(filename)
            video_path = os.path.join(tmp, f"input{ext}")
            with open(video_path, "wb") as f:
                f.write(raw_bytes)

            # Get video duration
            duration = self._get_video_duration(video_path)

            # Calculate frame interval to get MAX_VIDEO_FRAMES frames
            if duration and duration > 0:
                interval = max(1, duration / self.MAX_VIDEO_FRAMES)
                fps_arg = f"1/{int(interval)}"
            else:
                fps_arg = "1"  # 1 frame per second fallback

            # Extract frames
            frame_pattern = os.path.join(tmp, "frame_%04d.png")
            result = subprocess.run([
                "ffmpeg", "-i", video_path,
                "-vf", f"fps={fps_arg},scale=1280:-1",
                "-frames:v", str(self.MAX_VIDEO_FRAMES),
                "-y", frame_pattern
            ], capture_output=True, text=True, timeout=60)

            # Find extracted frames
            frames = sorted([
                os.path.join(tmp, f)
                for f in os.listdir(tmp)
                if f.startswith("frame_") and f.endswith(".png")
            ])

            if not frames:
                return {
                    "success": False,
                    "error": "Could not extract frames from video. File may be corrupted or in an unsupported format.",
                    "media_type": "video"
                }

            # OCR each frame
            all_texts = []
            frame_results = []
            for i, frame_path in enumerate(frames):
                try:
                    img = Image.open(frame_path).convert("RGB")
                    text = pytesseract.image_to_string(img, config='--psm 3').strip()
                    text = self._clean_text(text)
                    if len(text) > 20:
                        all_texts.append(text)
                        frame_results.append({"frame": i + 1, "chars": len(text)})
                except Exception:
                    continue

            if not all_texts:
                return {
                    "success": False,
                    "error": "No readable text found in video frames. The video may not contain on-screen text.",
                    "media_type": "video",
                    "details": {"frames_extracted": len(frames), "frames_with_text": 0}
                }

            # Deduplicate and combine
            combined = self._deduplicate_texts(all_texts)
            combined = self._clean_text(combined)

            if len(combined) < self.MIN_TEXT_LENGTH:
                return {
                    "success": False,
                    "error": "Extracted text is too short for meaningful analysis.",
                    "media_type": "video",
                    "details": {"frames_with_text": len(all_texts)}
                }

            return {
                "success": True,
                "text": combined,
                "method": f"ffmpeg frame extraction ({len(frames)} frames) + Tesseract OCR",
                "media_type": "video",
                "details": {
                    "duration_seconds": round(duration, 1) if duration else "unknown",
                    "frames_extracted": len(frames),
                    "frames_with_text": len(all_texts),
                    "chars_extracted": len(combined),
                    "words_extracted": len(combined.split()),
                    "filename": filename
                }
            }

    # ── HELPERS ────────────────────────────────────────────────────────

    def _get_video_duration(self, video_path: str) -> float:
        """Get video duration in seconds using ffprobe."""
        try:
            result = subprocess.run([
                "ffprobe", "-v", "quiet",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path
            ], capture_output=True, text=True, timeout=10)
            return float(result.stdout.strip())
        except Exception:
            return 0.0

    def _guess_video_ext(self, filename: str) -> str:
        """Guess video extension from filename."""
        if not filename:
            return ".mp4"
        ext = os.path.splitext(filename)[1].lower()
        return ext if ext in (".mp4", ".mov", ".avi", ".webm", ".mpeg") else ".mp4"

    def _clean_text(self, text: str) -> str:
        """Clean OCR output: remove noise, normalize whitespace."""
        # Remove non-printable characters
        text = re.sub(r'[^\x20-\x7E\n]', ' ', text)
        # Remove very short lines (OCR noise)
        lines = [line.strip() for line in text.split('\n')]
        lines = [line for line in lines if len(line) > 3]
        text = ' '.join(lines)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _deduplicate_texts(self, texts: list) -> str:
        """Combine texts from multiple frames, removing near-duplicate lines."""
        seen_lines = set()
        unique_lines = []

        for text in texts:
            for line in text.split('. '):
                line = line.strip()
                if len(line) < 10:
                    continue
                # Normalize for comparison
                normalized = re.sub(r'\s+', ' ', line.lower())
                if normalized not in seen_lines:
                    seen_lines.add(normalized)
                    unique_lines.append(line)

        return '. '.join(unique_lines)
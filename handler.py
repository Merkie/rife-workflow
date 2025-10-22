#!/usr/bin/env python3
"""
RunPod Serverless Handler for RIFE Video Interpolation
Handles video frame interpolation using RIFE NCNN Vulkan

Optimized to use ephemeral /tmp storage for all intermediate files
(frames, temp videos) and persistent volume for final output only.
"""

import os
import sys
import json
import subprocess
import shutil
import glob
from pathlib import Path
from datetime import datetime
import runpod
import traceback

# --- Configuration ---
# Path to the persistent volume for FINAL outputs
VOLUME_BASE = Path(f"/workspace/ComfyUI-Storage/rife-workflow")
# Path to the RIFE binary built by the Dockerfile
RIFE_BIN = "/app/rife-ncnn-vulkan"
# Base for ephemeral (temporary) job data
EPHEMERAL_BASE = Path("/tmp")


def run_command(command, description="Running command"):
    """Execute a shell command and return output"""
    print(f"[INFO] {description}")
    print(f"[CMD] {command}")

    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        if result.stdout:
            print(f"[OUT] {result.stdout}")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Command failed: {e}")
        print(f"[STDERR] {e.stderr}")
        raise


def get_video_info(video_path):
    """Extract FPS and frame count from video"""
    # Get FPS
    fps_cmd = f"ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of default=noprint_wrappers=1:nokey=1 {video_path}"
    fps_string = run_command(fps_cmd, "Getting video FPS").strip()

    # Calculate FPS from fraction
    if '/' in fps_string:
        num, den = fps_string.split('/')
        original_fps = float(num) / float(den)
    else:
        original_fps = float(fps_string)

    # Get frame count
    frames_cmd = f"ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames -of default=noprint_wrappers=1:nokey=1 {video_path}"
    original_frames = int(run_command(frames_cmd, "Getting frame count").strip())

    return original_fps, original_frames


def calculate_interpolation_params(original_fps, original_frames, target_fps):
    """Calculate interpolation parameters"""
    multiplier = int(round(target_fps / original_fps))
    frames_to_generate = (original_frames * multiplier) - (multiplier - 1)
    total_frames_needed = int(round((original_frames / original_fps) * target_fps))
    frames_to_pad = total_frames_needed - frames_to_generate

    print(f"[INFO] Original FPS: {original_fps}")
    print(f"[INFO] Original Frames: {original_frames}")
    print(f"[INFO] Target FPS: {target_fps}")
    print(f"[INFO] Multiplier: {multiplier}x")
    print(f"[INFO] AI will generate: {frames_to_generate} frames")
    print(f"[INFO] Final video needs: {total_frames_needed} frames")
    print(f"[INFO] Padding with: {frames_to_pad} hold frames")

    return multiplier, frames_to_generate, total_frames_needed, frames_to_pad


def setup_job_workspace(job_id):
    """Setup unique ephemeral workspace for processing and persistent dir for output"""
    job_workspace = EPHEMERAL_BASE / f"job_{job_id}"
    input_frames_dir = job_workspace / "input_frames"
    output_frames_dir = job_workspace / "output_frames"

    print(f"[INFO] Setting up ephemeral job workspace: {job_workspace}")
    job_workspace.mkdir(parents=True, exist_ok=True)
    input_frames_dir.mkdir(parents=True, exist_ok=True)
    output_frames_dir.mkdir(parents=True, exist_ok=True)

    # Also prepare the final persistent output directory
    persistent_output_dir = VOLUME_BASE / f"job_{job_id}"
    print(f"[INFO] Preparing persistent output dir: {persistent_output_dir}")
    persistent_output_dir.mkdir(parents=True, exist_ok=True)
    
    return job_workspace, input_frames_dir, output_frames_dir, persistent_output_dir


def cleanup_job_workspace(job_workspace):
    """Clean up ephemeral files from /tmp"""
    if not job_workspace or not job_workspace.exists():
        return
        
    print(f"[INFO] Cleaning up ephemeral workspace: {job_workspace}")
    try:
        shutil.rmtree(job_workspace)
        print("[INFO] Ephemeral cleanup complete.")
    except Exception as e:
        print(f"[WARNING] Ephemeral cleanup encountered an error: {e}")


def deduplicate_video(input_path, output_path):
    """Remove duplicate frames from video"""
    cmd = f'ffmpeg -i {input_path} -vf "mpdecimate,setpts=N/FRAME_RATE/TB" -an {output_path} -y'
    run_command(cmd, "Deduplicating video frames")


def extract_frames(video_path, output_dir):
    """Extract frames from video"""
    cmd = f"ffmpeg -i {video_path} {output_dir}/%08d.png"
    run_command(cmd, "Extracting video frames")


def run_rife_interpolation(input_dir, output_dir, ai_model, frames_to_generate, gpu_id=0):
    """Run RIFE interpolation"""
    cmd = (
        f"{RIFE_BIN} "
        f"-i {input_dir} "
        f"-o {output_dir} "
        f"-m {ai_model} "
        f"-n {frames_to_generate} "
        f"-g {gpu_id} "
        f"-j 4:8:4 "
        f"-x -z"
    )
    run_command(cmd, "Running RIFE AI interpolation")


def pad_frames(output_frames_dir, frames_to_pad):
    """Pad final frames by duplicating the last frame"""
    if frames_to_pad <= 0:
        print("[INFO] No padding needed")
        return

    print(f"[INFO] Padding {frames_to_pad} frames...")
    
    frame_files = sorted(glob.glob(str(output_frames_dir / "*.png")))
    if not frame_files:
        raise ValueError("No output frames found for padding")

    last_frame = frame_files[-1]
    last_frame_number = int(Path(last_frame).stem)

    for i in range(1, frames_to_pad + 1):
        new_frame_number = last_frame_number + i
        new_frame_name = f"{new_frame_number:08d}.png"
        new_frame_path = output_frames_dir / new_frame_name
        shutil.copy2(last_frame, new_frame_path)


def build_final_video(output_frames_dir, target_fps, input_video_with_audio, output_video_path):
    """Build final video from interpolated frames"""
    cmd = (
        f"ffmpeg -framerate {target_fps} "
        f"-i {output_frames_dir}/%08d.png "
        f"-i {input_video_with_audio} "
        f"-c:v libx264 -pix_fmt yuv420p "
        f"-c:a copy "
        f"-map 0:v:0 -map '1:a:0?' "
        f"{output_video_path} -y"
    )
    run_command(cmd, "Building final video")


def handler(event):
    """
    RunPod serverless handler function
    """
    job_workspace = None  # This will be the ephemeral /tmp path

    try:
        print("[INFO] Starting RIFE interpolation job")
        print(f"[INFO] Event: {json.dumps(event, indent=2)}")

        # Parse input
        job_input = event.get("input", {})
        video_url = job_input.get("video_url")
        video_path = job_input.get("video_path")
        target_fps = job_input.get("target_fps", 240)
        ai_model = job_input.get("ai_model", "rife-v4.6")
        output_filename = job_input.get("output_filename")

        # Validate input
        if not video_url and not video_path:
            return {"error": "Either video_url or video_path must be provided"}

        # Generate unique job ID
        job_id = event.get("id", f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}")
        print(f"[INFO] Job ID: {job_id}")

        # Setup ephemeral workspace and persistent output dir
        job_workspace, input_frames_dir, output_frames_dir, persistent_output_dir = \
            setup_job_workspace(job_id)

        # Define ephemeral video paths
        original_input_video = job_workspace / "input.mp4"
        deduped_video = job_workspace / "input-deduped.mp4"

        # Download or copy input video to *ephemeral* workspace
        if video_url:
            print(f"[INFO] Downloading video to: {original_input_video}")
            run_command(f"wget -O {original_input_video} '{video_url}'", "Downloading video")
        elif video_path:
            # Check if path is on the persistent volume
            if not Path(video_path).exists():
                 return {"error": f"Provided video_path does not exist: {video_path}"}
            print(f"[INFO] Copying video from: {video_path} to {original_input_video}")
            shutil.copy2(video_path, original_input_video)

        # Get video info
        original_fps, original_frames = get_video_info(original_input_video)

        # Calculate parameters
        multiplier, frames_to_generate, total_frames_needed, frames_to_pad = \
            calculate_interpolation_params(original_fps, original_frames, target_fps)

        # Deduplicate frames (ephemeral -> ephemeral)
        deduplicate_video(original_input_video, deduped_video)

        # Extract frames (ephemeral -> ephemeral)
        extract_frames(deduped_video, input_frames_dir)

        # Run RIFE interpolation (ephemeral -> ephemeral)
        run_rife_interpolation(input_frames_dir, output_frames_dir, ai_model, frames_to_generate)

        # Pad frames (in ephemeral)
        pad_frames(output_frames_dir, frames_to_pad)

        # Determine final output filename
        if not output_filename:
            output_filename = f"output_{target_fps}fps_{job_id}.mp4"
        
        # Build final video in ephemeral storage first
        ephemeral_output_path = job_workspace / output_filename
        build_final_video(output_frames_dir, target_fps, original_input_video, ephemeral_output_path)
        
        # Now, move the final video to the persistent volume
        final_output_path = persistent_output_dir / output_filename
        print(f"[INFO] Moving final video to persistent storage: {final_output_path}")
        shutil.move(ephemeral_output_path, final_output_path)

        # Clean up *only* the ephemeral workspace
        cleanup_job_workspace(job_workspace)

        print("[INFO] Job completed successfully!")
        print(f"[INFO] Final output saved to: {final_output_path}")

        return {
            "status": "success",
            "output_path": str(final_output_path),
            "job_id": job_id,
            "original_fps": original_fps,
            "original_frames": original_frames,
            "target_fps": target_fps,
            "multiplier": multiplier,
            "frames_generated": frames_to_generate,
            "total_frames": total_frames_needed
        }

    except Exception as e:
        print(f"[ERROR] Job failed: {str(e)}")
        traceback.print_exc()

        # Try to clean up ephemeral data on error
        if job_workspace:
            cleanup_job_workspace(job_workspace)

        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }


if __name__ == "__main__":
    print("[INFO] Starting RunPod Serverless Worker for RIFE")
    runpod.serverless.start({"handler": handler})
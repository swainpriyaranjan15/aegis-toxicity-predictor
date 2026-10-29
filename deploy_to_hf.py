"""
Automated 24/7 Cloud Deployment Script for Hugging Face Spaces.
This deploys the AegisAI FastAPI model server to a permanent cloud instance (16GB RAM, Free, 24/7 online).
"""

import os
import sys
from pathlib import Path

def main():
    try:
        from huggingface_hub import HfApi, login, whoami
    except ImportError:
        print("Error: huggingface_hub is not installed. Please run: pip install huggingface_hub")
        sys.exit(1)

    print("=" * 65)
    print("   AEGIS AI - 24/7 PERMANENT CLOUD DEPLOYMENT (HUGGING FACE)")
    print("=" * 65)
    print()
    print("This script will deploy your model to Hugging Face Spaces (Free, 16GB RAM).")
    print("Once deployed, your website will stay online 24/7, even when your PC is OFF.")
    print()

    # Step 1: Check Authentication
    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)

    try:
        user_info = whoami(token=token)
        username = user_info["name"]
        print(f"Logged in as Hugging Face user: {username}")
    except Exception:
        print("You are not currently logged in to Hugging Face.")
        print("1. Go to https://huggingface.co/settings/tokens")
        print("2. Create a new token with 'Write' permissions.")
        print()
        entered_token = input("Enter your Hugging Face Token: ").strip()
        if not entered_token:
            print("Token cannot be empty. Aborted.")
            sys.exit(1)
        login(token=entered_token)
        token = entered_token
        api = HfApi(token=token)
        username = whoami(token=token)["name"]
        print(f"Successfully authenticated as: {username}")

    # Step 2: Determine Space Name
    default_space = "aegis-toxicity-moderator"
    space_input = input(f"\nEnter Space name [default: {default_space}]: ").strip()
    space_name = space_input if space_input else default_space
    repo_id = f"{username}/{space_name}"

    print(f"\nTarget Space: https://huggingface.co/spaces/{repo_id}")

    # Step 3: Create the Space if it doesn't exist
    print("\n[1/3] Ensuring Hugging Face Space exists...")
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="docker",
            private=False,
            exist_ok=True
        )
        print("Space is ready.")
    except Exception as e:
        print(f"Note / Warning during space creation: {e}")

    # Step 4: Upload project files
    print("\n[2/3] Uploading Dockerfile, backend, and model weights...")
    print("Note: This will upload the fine-tuned model (~267MB safetensors).")
    print("Large training checkpoints and raw datasets are automatically excluded.")

    project_root = Path(__file__).resolve().parent

    ignore_patterns = [
        "data/**",
        "data/*",
        "outputs/**",
        "outputs/*",
        "tests/**",
        "tests/*",
        "*.pt",
        "*checkpoint_latest.pt",
        "*.bat",
        "cloudflared.exe",
        ".git/**",
        ".git/*",
        "__pycache__/**",
        "__pycache__/*",
        "*.pyc",
        ".gemini/**",
        ".agents/**",
        "scratch/**",
        "render.yaml",
        "report"
    ]

    try:
        api.upload_folder(
            folder_path=str(project_root),
            repo_id=repo_id,
            repo_type="space",
            commit_message="Deploy AegisAI toxicity predictor with Docker",
            ignore_patterns=ignore_patterns
        )
        print("\n[3/3] Upload complete!")
    except Exception as e:
        print(f"\nUpload error: {e}")
        sys.exit(1)

    # Step 5: Finished
    direct_app_url = f"https://{username}-{space_name.replace('_', '-')}.hf.space"
    space_url = f"https://huggingface.co/spaces/{repo_id}"

    print("\n" + "=" * 65)
    print("   DEPLOYMENT INITIATED SUCCESSFULLY!")
    print("=" * 65)
    print(f"Space Overview & Build Logs:  {space_url}")
    print(f"Direct Permanent Web App URL: {direct_app_url}")
    print()
    print("Hugging Face is now building the Docker container (takes ~2-3 minutes).")
    print("Once built, your site will be ONLINE 24/7 FOREVER, even when your PC is turned off!")
    print("=" * 65)

if __name__ == "__main__":
    main()

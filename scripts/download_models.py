# Copyright (C) 2025 Xiaomi Corporation
# This software may be used and distributed according to the terms of the Xiaomi Miloco License Agreement.

"""
Model Download Script
Support downloading models from ModelScope or HuggingFace
"""

import argparse
import sys
import subprocess
from pathlib import Path

# TODO: update default model list before release
# Default model list (model_id and optional filename)
DEFAULT_MODELS = [
    {"model_id": "williamljz/test2-gguf", "filename": None},
    {"model_id": "Qwen/Qwen3-8B-GGUF", "filename": "Qwen3-8B-Q4_K_M.gguf"}
]

DEFUALT_TARGER_PATH = Path(__file__).parent.parent / "models"
DEFUALT_DOWNLOAD_SOURCE = "modelscope"

class ModelDownloader:
    """Model Downloader"""

    def __init__(self, source: str = DEFUALT_DOWNLOAD_SOURCE, target_dir: str = None):
        """
        Initialize downloader

        Args:
            source: Download source, "modelscope" or "huggingface"
            target_dir: Target directory, defaults to .model in project root
        """
        self.source = source.lower()
        self.target_dir = DEFUALT_TARGER_PATH

        # Set target directory
        if target_dir:
            self.target_dir = Path(target_dir).absolute()

        self.target_dir.mkdir(parents=True, exist_ok=True)

        print(f"📁 Download directory: {self.target_dir}")
        print(f"🌐 Download source: {self.source}")
        print("-" * 60)

    def _install_package(self, package_name: str) -> bool:
        """
        Install Python package using pip

        Args:
            package_name: Package name

        Returns:
            bool: Whether installation succeeded
        """
        print(f"📦 Installing {package_name}...")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", package_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            print(f"✅ {package_name} installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ {package_name} installation failed: {e}")
            return False

    def check_environment(self, auto_install: bool = True) -> bool:
        """
        Check if download environment is ready, auto-install missing dependencies if needed

        Args:
            auto_install: Whether to auto-install missing dependencies

        Returns:
            bool: Whether environment is ready
        """
        print(f"🔍 Checking {self.source} download environment...")

        if self.source == "modelscope":
            try:
                import modelscope
                print(f"✅ ModelScope environment ready (version: {modelscope.__version__})")
                return True
            except ImportError:
                print("⚠️  ModelScope not installed")
                if auto_install:
                    print("🔧 Auto-installing ModelScope...")
                    if self._install_package("modelscope"):
                        try:
                            import modelscope
                            print(
                                f"✅ ModelScope environment ready (version: {modelscope.__version__})")
                            return True
                        except ImportError:
                            print("❌ Unable to import ModelScope after installation")
                            return False
                    else:
                        return False
                else:
                    print("📝 Please run manually: pip install modelscope")
                    return False

        elif self.source == "huggingface":
            try:
                import huggingface_hub
                print(
                    f"✅ HuggingFace environment ready (version: {huggingface_hub.__version__})")
                return True
            except ImportError:
                print("⚠️  HuggingFace Hub not installed")
                if auto_install:
                    print("🔧 Auto-installing HuggingFace Hub...")
                    if self._install_package("huggingface-hub"):
                        try:
                            import huggingface_hub
                            print(
                                f"✅ HuggingFace environment ready (version: {huggingface_hub.__version__})")
                            return True
                        except ImportError:
                            print("❌ Unable to import HuggingFace Hub after installation")
                            return False
                    else:
                        return False
                else:
                    print("📝 Please run manually: pip install huggingface-hub")
                    return False
        else:
            print(f"❌ Unsupported download source: {self.source}")
            return False

    def download_model(self, model_id: str, filename: str = None) -> bool:
        """
        Download a single model

        Args:
            model_id: Model ID (e.g., Qwen/Qwen2.5-7B-Instruct-GGUF)
            filename: Optional filename if you want to download a specific file from the repo

        Returns:
            bool: Whether download succeeded
        """
        try:
            if self.source == "modelscope":
                return self._download_from_modelscope(model_id, filename)
            elif self.source == "huggingface":
                return self._download_from_huggingface(model_id, filename)
            else:
                print(f"❌ Unsupported download source: {self.source}")
                return False
        except Exception as e:
            print(f"❌ Download failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _download_from_modelscope(self, model_id: str, filename: str = None) -> bool:
        """Download model from ModelScope"""
        from modelscope.hub.snapshot_download import snapshot_download

        print(f"\n📦 Downloading from ModelScope: {model_id}")
        if filename:
            print(f"   File: {filename}")

        try:
            local_dir = self.target_dir / model_id

            if filename:
                # Download specific file
                model_dir = snapshot_download(
                    model_id,
                    local_dir=str(local_dir),
                    allow_patterns=filename
                )
            else:
                # Download entire repository
                model_dir = snapshot_download(
                    model_id,
                    local_dir=str(local_dir)
                )

            print(f"✅ Download completed: {model_dir}")
            return True

        except Exception as e:
            print(f"❌ ModelScope download failed: {e}")
            return False

    def _download_from_huggingface(self, model_id: str, filename: str = None) -> bool:
        """Download model from HuggingFace"""
        from huggingface_hub import snapshot_download

        print(f"\n📦 Downloading from HuggingFace: {model_id}")
        if filename:
            print(f"   File: {filename}")
        try:
            local_dir = self.target_dir / model_id

            if filename:
                # Download specific file
                model_dir = snapshot_download(
                    repo_id=model_id,
                    local_dir=str(local_dir),
                    allow_patterns=filename
                )
            else:
                # Download entire repository
                model_dir = snapshot_download(
                    repo_id=model_id,
                    local_dir=str(local_dir)
                )

            print(f"✅ Download completed: {model_dir}")
            return True

        except Exception as e:
            print(f"❌ HuggingFace download failed: {e}")
            return False

    def download_models(self, model_ids: list, filenames: list = None) -> bool:
        """
        Download specified model list

        Args:
            model_ids: Model ID list (e.g., ["Qwen/Qwen2.5-7B-Instruct-GGUF"])
            filenames: Optional filename list, corresponding to each model ID

        Returns:
            bool: Whether all downloads succeeded
        """
        print(f"\n🎯 Preparing to download {len(model_ids)} model(s)")
        print("=" * 60)

        success_count = 0
        for idx, model_id in enumerate(model_ids, 1):
            print(f"\n[{idx}/{len(model_ids)}] Downloading model: {model_id}")

            filename = None
            if filenames and idx <= len(filenames):
                filename = filenames[idx - 1]

            if self.download_model(model_id, filename):
                success_count += 1

        print("\n" + "=" * 60)
        print(f"📊 Download completed: {success_count}/{len(model_ids)} succeeded")

        return success_count == len(model_ids)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Model Download Script - Support downloading models from ModelScope or HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download default models (from ModelScope)
  python download_models.py
  
  # Download default models from HuggingFace
  python download_models.py --source huggingface
  
  # Download specific model (use full model_id)
  python download_models.py --models XiaomiMiMo/MiMo-VL-7B-RL-GGUF
  
  # Download multiple models
  python download_models.py --models XiaomiMiMo/MiMo-VL-7B-RL-GGUF XiaomiMiMo/MiMo-VL-7B-SFT-GGUF
  
  # Download specific file (instead of entire repository)
  python download_models.py --models XiaomiMiMo/MiMo-VL-7B-RL-GGUF --files MiMo-VL-7B-RL_BF16.gguf
  
  # Specify download directory
  python download_models.py --target /path/to/models --models XiaomiMiMo/MiMo-VL-7B-RL-GGUF
  
  # Disable auto-install dependencies
  python download_models.py --auto-install false --models XiaomiMiMo/MiMo-VL-7B-RL-GGUF
        """
    )

    parser.add_argument(
        "--source",
        choices=["modelscope", "huggingface"],
        default="modelscope",
        help="Download source (default: modelscope)"
    )

    default_model_str = ", ".join([m["model_id"] for m in DEFAULT_MODELS])
    parser.add_argument(
        "--models",
        nargs="+",
        help=f"Model ID list, e.g.: Qwen/Qwen2.5-7B-Instruct-GGUF (default: {default_model_str})"
    )

    parser.add_argument(
        "--files",
        nargs="+",
        help="Optional: filename for each model, if you want to download specific files from the repo"
    )

    parser.add_argument(
        "--target",
        type=str,
        help="Download target directory (default: project_root/.model)"
    )

    parser.add_argument(
        "--auto-install",
        type=lambda x: x.lower() in ["true", "1", "yes"],
        default=True,
        help="Auto-install missing dependencies (default: True)"
    )

    args = parser.parse_args()

    # Determine model list to download
    if args.models:
        # User specified models via command line
        models_to_download = args.models
        files_to_download = args.files
    else:
        # Use default models
        models_to_download = [m["model_id"] for m in DEFAULT_MODELS]
        files_to_download = [m["filename"] for m in DEFAULT_MODELS]

    # Create downloader
    downloader = ModelDownloader(source=args.source, target_dir=args.target)

    # Check environment (with auto-install support)
    auto_install = args.auto_install
    if not downloader.check_environment(auto_install=auto_install):
        return 1

    print()

    # Download models
    if downloader.download_models(models_to_download, files_to_download):
        print("\n🎉 All models downloaded successfully!")
        return 0
    else:
        print("\n⚠️  Some models failed to download")
        return 1


if __name__ == "__main__":
    sys.exit(main())

import os
import tempfile
import argparse
import warnings
from PIL import Image
import numpy as np
import torch
import pydicom

# Filter warnings
warnings.filterwarnings("ignore")

# Set Hugging Face cache directory to match the notebook's cache location
os.environ["HF_HOME"] = "T:/huggingface_cache"

# Pre-import verification
try:
    import transformers
    import peft
    
    if transformers.__version__ != "4.40.0":
        print(f"[WARN] Current transformers version is {transformers.__version__}. "
              f"CheXagent works best with version 4.40.0.")
except ImportError as e:
    print(f"[WARN] Missing dependency: {e.name}. Please install dependencies listed in the notebooks.")


# Exact SYSTEM_PROMPT from vgroundingcxr.ipynb
SYSTEM_PROMPT = (
    'Kamu adalah asisten AI radiologi. '
    'Ketika ditanya tentang temuan pada foto rontgen dada, jawab dengan nama temuan berformat <|ref|>nama_temuan<|/ref|>'
    'diikuti koordinat bounding box yang dinormalisasi dengan nilai antara 0 hingga 100 dalam format<|box|>(x1,y1),(x2,y2)<|/box|.'
    'Jika ada lebih dari 1 temuan, berikan setiap temuan sepasang <|ref|> dan <|box|>'
)


def load_image(image_path):
    """
    Loads an image from the given path. Supports DICOM as well as standard formats.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at: {image_path}")

    is_dicom = image_path.lower().endswith(('.dcm', '.dicom')) or pydicom.misc.is_dicom(image_path)
    
    if is_dicom:
        print(f"Loading DICOM image: {image_path}")
        try:
            dcm = pydicom.dcmread(image_path)
            arr = dcm.pixel_array.astype(np.float32)
            
            # Normalize to 0-1 range, then scale to 0-255 uint8
            lo, hi = arr.min(), arr.max()
            if hi > lo:
                arr = (arr - lo) / (hi - lo)
            arr = (arr * 255).astype(np.uint8)
            
            # Convert to RGB PIL Image
            image = Image.fromarray(arr).convert("RGB")
            return image
        except Exception as e:
            raise RuntimeError(f"Failed to read DICOM image: {e}")
    else:
        print(f"Loading standard image: {image_path}")
        try:
            image = Image.open(image_path).convert("RGB")
            return image
        except Exception as e:
            raise RuntimeError(f"Failed to read image: {e}")


def parse_args():
    parser = argparse.ArgumentParser(description="Inference script for CheXagent vgroundingcxr.ipynb.")
    parser.add_argument(
        "--image", 
        type=str, 
        required=True, 
        help="Path to the chest X-ray image (DICOM or standard formats)."
    )
    parser.add_argument(
        "--prompt", 
        type=str, 
        default="Sisi mana yang menunjukkan kelainan?", 
        help="Prompt/question for the model."
    )
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True, 
        help="Path to the local CheXagent model (.pt or .pth file)."
    )
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="StanfordAIMI/CheXagent-2-3b", 
        help="Base model HF name/path (used only to load tokenizer & processor locally)."
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda", 
        help="Device to run the model on ('cuda' or 'cpu')."
    )
    parser.add_argument(
        "--max_new_tokens", 
        type=int, 
        default=100, 
        help="Maximum new tokens to generate."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load Image
    try:
        image = load_image(args.image)
    except Exception as e:
        print(f"[ERROR] {e}")
        return

    # Determine device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available, switching to CPU.")
        device = "cpu"

    print(f"device = {device}")
    print(f"Loading tokenizer & processor locally for: {args.model_name}...")
    from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM

    # Load tokenizer and processor locally from cache
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True, local_files_only=False)
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True, local_files_only=False)

    # Load model directly from the .pt file
    print(f"Loading model from PyTorch checkpoint: {args.model_path}...")
    loaded = torch.load(args.model_path, map_location=device)
    
    if isinstance(loaded, torch.nn.Module):
        print("Loaded full serialized model object.")
        model = loaded
    else:
        print("Loaded state dict. Instantiating base model (from local cache if available) and loading weights...")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            local_files_only=False,
        )
        
        # Unpack state dict if nested
        if isinstance(loaded, dict):
            if "model" in loaded:
                loaded = loaded["model"]
            elif "state_dict" in loaded:
                loaded = loaded["state_dict"]

        base_model.load_state_dict(loaded, strict=False)
        model = base_model
    print("done loading model.")
    model.eval()
    print(f"Model ready on {device}.")

    # Format inputs and generate
    temp_img_path = None
    try:
        # Save PIL image to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            temp_img_path = tmp.name
        image.save(temp_img_path)

        formatted_query = tokenizer.from_list_format([
            {"image": temp_img_path},
            {"text": args.prompt},
        ])

        conversation = [
            {"from": "system", "value": SYSTEM_PROMPT},
            {"from": "human", "value": formatted_query},
        ]

        inputs = tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        )
        
        # Move inputs to device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate response using inference mode
        print("Generating response (inference_mode)...")
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
            )

        # Decode output
        input_length = inputs["input_ids"].shape[-1]
        generated_ids = output_ids[0][input_length:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        print("\n" + "="*40)
        print(f"Prompt  : {args.prompt}")
        print(f"Response: {response}")
        
        # Parse and convert bounding boxes
        import re
        # Supports both <|ref|>finding<|/ref|> <|box|>(x1,y1),(x2,y2)<|/box|> and <ref>finding</ref> <box>(x1,y1),(x2,y2)</box>
        pattern = r"(?:<\|ref\|>|<ref>)\s*([^<]+?)\s*(?:<\|/ref\|>|<\/ref>)\s*(?:<\|box\|>|<box>)\s*\((\d+)\s*,\s*(\d+)\)\s*,\s*\((\d+)\s*,\s*(\d+)\)\s*(?:<\|/box\|>|<\/box>)"
        matches = re.findall(pattern, response)
        
        if matches:
            print("\nParsed Bounding Boxes:")
            for match in matches:
                label = match[0].strip()
                x1, y1, x2, y2 = map(int, match[1:])
                print(f"  - {label}: [{x1}, {y1}, {x2}, {y2}]")
        else:
            print("\nNo bounding boxes parsed from response.")
        print("="*40 + "\n")

    finally:
        # Clean up temporary image
        if temp_img_path and os.path.exists(temp_img_path):
            try:
                os.remove(temp_img_path)
            except Exception:
                pass


if __name__ == "__main__":
    main()

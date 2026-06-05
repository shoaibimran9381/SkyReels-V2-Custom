
import torch
import argparse

import pandas as pd
import numpy as np

from tqdm import tqdm
from transformers import AutoTokenizer, AutoProcessor

from torch.utils.data import DataLoader

# Handle platform-specific imports
import sys
try:
    import decord
    DECORD_AVAILABLE = True
except ImportError:
    import cv2
    DECORD_AVAILABLE = False
    print("⚠️  decord not available, using OpenCV fallback", file=sys.stderr)

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("⚠️  vllm not available, using mock LLM", file=sys.stderr)

SYSTEM_PROMPT = "I need you to generate a structured and detailed caption for the provided video. The structured output and the requirements for each field are as shown in the following JSON content: {\"subjects\": [{\"appearance\": \"Main subject appearance description\", \"action\": \"Main subject action\", \"expression\": \"Main subject expression  (Only for human/animal categories, empty otherwise)\", \"position\": \"Subject position in the video (Can be relative position to other objects or spatial description)\", \"TYPES\": {\"type\": \"Main category (e.g., Human)\", \"sub_type\": \"Sub-category (e.g., Man)\"}, \"is_main_subject\": true}, {\"appearance\": \"Non-main subject appearance description\", \"action\": \"Non-main subject action\", \"expression\": \"Non-main subject expression (Only for human/animal categories, empty otherwise)\", \"position\": \"Position of non-main subject 1\", \"TYPES\": {\"type\": \"Main category (e.g., Vehicles)\", \"sub_type\": \"Sub-category (e.g., Ship)\"}, \"is_main_subject\": false}], \"shot_type\": \"Shot type(Options: long_shot/full_shot/medium_shot/close_up/extreme_close_up/other)\", \"shot_angle\": \"Camera angle(Options: eye_level/high_angle/low_angle/other)\", \"shot_position\": \"Camera position(Options: front_view/back_view/side_view/over_the_shoulder/overhead_view/point_of_view/aerial_view/overlooking_view/other)\", \"camera_motion\": \"Camera movement description\", \"environment\": \"Video background/environment description\", \"lighting\": \"Lighting information in the video\"}"


class VideoTextDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, model_path):
        if isinstance(csv_path, pd.DataFrame):
            self.meta = csv_path
        else:
            self.meta = pd.read_csv(csv_path)
        self._path = 'path'
        if VLLM_AVAILABLE:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.processor = AutoProcessor.from_pretrained(model_path)
  
    def __getitem__(self, index):
        row = self.meta.iloc[index]
        path = row[self._path]
        real_index = self.meta.index[index]
        
        # Use decord if available, otherwise use OpenCV
        if DECORD_AVAILABLE:
            vr = decord.VideoReader(path, ctx=decord.cpu(0), width=360, height=420)
            start = 0
            end = len(vr)
            frame_indices = self.get_index(end-start, 16, st=start)
            frames = vr.get_batch(frame_indices).asnumpy()  # n h w c
        else:
            # OpenCV fallback
            cap = cv2.VideoCapture(path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_indices = self.get_index(total_frames, 16, st=0)
            frames = []
            
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, (360, 420))
                    frames.append(frame)
                else:
                    frames.append(frames[-1] if frames else np.zeros((420, 360, 3), dtype=np.uint8))
            
            cap.release()
            frames = np.stack(frames)  # n h w c
        
        video_inputs = [torch.from_numpy(frames).permute(0, 3, 1, 2)]
        conversation = {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": row['path'],
                            "max_pixels": 360 * 420, # 460800
                            "fps": 2.0,
                        },
                        {   
                            "type": "text", 
                            "text": SYSTEM_PROMPT
                        },
                    ],
                }
        
        if VLLM_AVAILABLE:
            # Generate user_input with processor
            user_input = self.processor.apply_chat_template(
                [conversation],
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Fallback user input
            user_input = f"Video: {row['path']}\n{SYSTEM_PROMPT}"
                
        results = dict()
        inputs = {
            'prompt': user_input,
            'multi_modal_data': {'video': video_inputs}
        }
        results["index"] = real_index
        results['input'] = inputs
        return results

    def __len__(self):
        return len(self.meta)

    def get_index(self, video_size, num_frames, st=0):
        seg_size = max(0., float(video_size - 1) / num_frames)
        max_frame = int(video_size) - 1
        seq = []
        # index from 1, must add 1
        for i in range(num_frames):
            start = int(np.round(seg_size * i))
            # end = int(np.round(seg_size * (i + 1)))
            idx = min(start, max_frame)
            seq.append(idx+st)
        return seq
    
def result_writer(indices_list: list, result_list: list, meta: pd.DataFrame, column):
    flat_indices = []
    for x in zip(indices_list):
        flat_indices.extend(x)
    flat_results = []
    for x in zip(result_list):
        flat_results.extend(x)
    
    flat_indices = np.array(flat_indices)
    flat_results = np.array(flat_results)

    unique_indices, unique_indices_idx = np.unique(flat_indices, return_index=True)
    meta.loc[unique_indices, column[0]] = flat_results[unique_indices_idx]

    meta = meta.loc[unique_indices]
    return meta


def worker_init_fn(worker_id):
    # Set different seed for each worker
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    # Prevent deadlocks by setting timeout
    torch.set_num_threads(1)


def mock_llm_generate(batch_user_inputs, sampling_params=None, use_tqdm=False):
    """Mock LLM for testing when vllm is unavailable (e.g., macOS)"""
    import json
    mock_captions = []
    for _ in batch_user_inputs:
        mock_caption = {
            "subjects": [{"appearance": "Subject visible", "action": "Main action", 
                         "expression": "Neutral", "position": "Center", 
                         "TYPES": {"type": "Human", "sub_type": "Person"}, "is_main_subject": True}],
            "shot_type": "medium_shot", "shot_angle": "eye_level", "shot_position": "front_view",
            "camera_motion": "Static", "environment": "Indoor", "lighting": "Natural"
        }
        mock_captions.append(json.dumps(mock_caption))
    return mock_captions

def main():
    parser = argparse.ArgumentParser(description="SkyCaptioner-V1 vllm batch inference")
    parser.add_argument("--input_csv", default="./examples/test.csv")
    parser.add_argument("--out_csv", default="./examples/test_result.csv")
    parser.add_argument("--bs", type=int, default=4)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--model_path", required=True, type=str, help="skycaptioner-v1 model path")
    args = parser.parse_args()
    
    if not VLLM_AVAILABLE:
        print("\n⚠️  vLLM not available - running in MOCK mode")
        print("   For real inference, deploy on Linux with GPU support\n")
    
    if VLLM_AVAILABLE:
        dataset = VideoTextDataset(csv_path=args.input_csv, model_path=args.model_path)
    else:
        dataset = VideoTextDataset(csv_path=args.input_csv, model_path=None)
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.bs,
        num_workers=0 if not DECORD_AVAILABLE else 4,  # macOS: no multiprocessing
        worker_init_fn=worker_init_fn if DECORD_AVAILABLE else None,
        persistent_workers=DECORD_AVAILABLE,
        timeout=180,
    )

    indices_list = []
    caption_save = []
    
    for video_batch in tqdm(dataloader):
        indices = video_batch["index"]
        inputs = video_batch["input"]
        batch_user_inputs = []
        for prompt, video in zip(inputs['prompt'], inputs['multi_modal_data']['video'][0]):
            usi={'prompt':prompt, 'multi_modal_data':{'video':video}}
            batch_user_inputs.append(usi)
        
        if VLLM_AVAILABLE:
            sampling_params = SamplingParams(temperature=0.05, max_tokens=2048)
            llm = LLM(model=args.model_path,
                gpu_memory_utilization=0.6, 
                max_model_len=31920,
                tensor_parallel_size=args.tp)
            outputs = llm.generate(batch_user_inputs, sampling_params, use_tqdm=False)
            struct_outputs = [output.outputs[0].text for output in outputs]
        else:
            struct_outputs = mock_llm_generate(batch_user_inputs)

        indices_list.extend(indices.tolist())
        caption_save.extend(struct_outputs)
    
    meta_new = result_writer(indices_list, caption_save, dataset.meta, column=["structural_caption"])
    meta_new.to_csv(args.out_csv, index=False)
    print(f'✓ Saved structural_caption to {args.out_csv}')

if __name__ == '__main__':
    main()
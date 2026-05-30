#!/usr/bin/env python3
# Copyright (c) 2025 Alibaba Inc
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Modern WebUI for CosyVoice3
Supports:
- Zero-shot inference without training
- Fine-grained control with tags (laughter, breath, etc.)
- Pretrained speaker voice cloning
- Upload reference audio to create custom voice
- Prompt text for better synthesis quality
"""
import os
import sys
import argparse
import tempfile
import shutil
import gradio as gr
import numpy as np
import torch
import torchaudio
import random
import json
from pathlib import Path

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT_DIR, 'third_party/Matcha-TTS'))

from cosyvoice.cli.cosyvoice import AutoModel, CosyVoice3
from cosyvoice.utils.file_utils import logging
from cosyvoice.utils.common import set_all_random_seed

# Constants
MAX_AUDIO_DURATION = 30  # seconds
DEFAULT_SAMPLE_RATE = 24000
SPKINFO_FILE = 'spk2info_custom.json'


def get_available_models():
    """Get list of available pretrained models"""
    model_base = 'pretrained_models'
    models = []
    
    if os.path.exists(model_base):
        for item in os.listdir(model_base):
            item_path = os.path.join(model_base, item)
            if os.path.isdir(item_path):
                # Check for cosyvoice3.yaml
                if os.path.exists(os.path.join(item_path, 'cosyvoice3.yaml')):
                    models.append((item, item_path))
                elif os.path.exists(os.path.join(item_path, 'cosyvoice2.yaml')):
                    models.append((item, item_path))
                elif os.path.exists(os.path.join(item_path, 'cosyvoice.yaml')):
                    models.append((item, item_path))
    
    if not models:
        # Default fallback
        models = [('Fun-CosyVoice3-0.5B', 'pretrained_models/Fun-CosyVoice3-0.5B')]
    
    return models


def generate_seed():
    """Generate a random seed"""
    seed = random.randint(1, 100000000)
    return {"__type__": "update", "value": seed}


def load_custom_spks(model_dir):
    """Load custom speaker voices from spk2info file"""
    spkinfo_path = os.path.join(model_dir, SPKINFO_FILE)
    if os.path.exists(spkinfo_path):
        try:
            with open(spkinfo_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return list(data.keys())
        except Exception as e:
            logging.warning(f'Failed to load custom speakers: {e}')
    return []


def save_custom_spk(model, prompt_text, prompt_wav, spk_name):
    """Save a custom speaker voice"""
    try:
        if hasattr(model, 'add_zero_shot_spk'):
            model.add_zero_shot_spk(prompt_text, prompt_wav, spk_name)
            model.save_spkinfo()
            return True, f"✓ Speaker '{spk_name}' saved successfully!"
        else:
            return False, "Current model does not support saving custom speakers"
    except Exception as e:
        return False, f"Error saving speaker: {str(e)}"


def process_audio_upload(audio_file):
    """Process uploaded audio file and return info"""
    if audio_file is None:
        return "No audio uploaded", None
    
    try:
        waveform, sample_rate = torchaudio.load(audio_file)
        duration = waveform.shape[1] / sample_rate
        
        if duration > MAX_AUDIO_DURATION:
            return f"⚠️ Audio too long ({duration:.1f}s). Max {MAX_AUDIO_DURATION}s recommended.", audio_file
        
        return f"✓ Audio loaded ({duration:.1f}s, {sample_rate}Hz)", audio_file
    except Exception as e:
        return f"❌ Error loading audio: {str(e)}", None


def validate_inputs(mode, tts_text, prompt_text, prompt_audio, instruct_text, selected_spk):
    """Validate inputs based on mode"""
    warnings = []
    
    # Text validation
    if not tts_text or len(tts_text.strip()) == 0:
        return False, "❌ Please enter synthesis text"
    
    if mode == 'Zero-Shot Voice Cloning':
        if prompt_audio is None:
            return False, "❌ Please upload or record reference audio for zero-shot cloning"
        if not prompt_text or len(prompt_text.strip()) == 0:
            return False, "❌ Please enter prompt text matching the reference audio"
    
    elif mode == 'Cross-Lingual Synthesis':
        if prompt_audio is None:
            return False, "❌ Please upload reference audio for cross-lingual synthesis"
    
    elif mode == 'Instruct Control':
        if not instruct_text or len(instruct_text.strip()) == 0:
            return False, "❌ Please enter instruction text"
    
    elif mode == 'Pretrained Speaker':
        if not selected_spk:
            return False, "❌ Please select a pretrained speaker"
    
    return True, ""


def generate_audio(tts_text, mode, selected_spk, prompt_text, prompt_audio_upload, 
                   prompt_audio_record, instruct_text, custom_spk_name, save_spk_btn,
                   seed, stream, speed, model):
    """Main audio generation function"""
    
    # Handle audio input priority
    prompt_audio = None
    if prompt_audio_upload is not None:
        prompt_audio = prompt_audio_upload
    elif prompt_audio_record is not None:
        prompt_audio = prompt_audio_record
    
    # Validate inputs
    valid, message = validate_inputs(mode, tts_text, prompt_text, prompt_audio, 
                                      instruct_text, selected_spk)
    if not valid:
        gr.Warning(message)
        yield (model.sample_rate, np.zeros(model.sample_rate))
        return
    
    # Set random seed
    set_all_random_seed(seed)
    
    try:
        if mode == 'Pretrained Speaker':
            logging.info('Using pretrained speaker mode')
            if save_spk_btn:
                # Save custom speaker
                if custom_spk_name and selected_spk:
                    success, msg = save_custom_spk(model, "", "", selected_spk)
                    if success:
                        gr.Info(msg)
            
            for output in model.inference_sft(tts_text, selected_spk, stream=stream, speed=speed):
                yield (model.sample_rate, output['tts_speech'].numpy().flatten())
        
        elif mode == 'Zero-Shot Voice Cloning':
            logging.info('Using zero-shot voice cloning mode')
            
            # Save custom speaker if requested
            if save_spk_btn and custom_spk_name:
                success, msg = save_custom_spk(model, prompt_text, prompt_audio, custom_spk_name)
                if success:
                    gr.Info(msg)
            
            for output in model.inference_zero_shot(tts_text, prompt_text, prompt_audio, 
                                                     stream=stream, speed=speed):
                yield (model.sample_rate, output['tts_speech'].numpy().flatten())
        
        elif mode == 'Cross-Lingual Synthesis':
            logging.info('Using cross-lingual synthesis mode')
            gr.Info('Cross-lingual mode: Ensure synthesis text and prompt audio are in different languages')
            
            for output in model.inference_cross_lingual(tts_text, prompt_audio, 
                                                         stream=stream, speed=speed):
                yield (model.sample_rate, output['tts_speech'].numpy().flatten())
        
        elif mode == 'Instruct Control':
            logging.info('Using instruct control mode')
            
            # Try inference_instruct2 first (CosyVoice2/3), fallback to inference_instruct
            if hasattr(model, 'inference_instruct2'):
                for output in model.inference_instruct2(tts_text, instruct_text, prompt_audio,
                                                         stream=stream, speed=speed):
                    yield (model.sample_rate, output['tts_speech'].numpy().flatten())
            else:
                # Fallback to original instruct mode with pretrained speaker
                spk_id = selected_spk if selected_spk else '中文女'
                for output in model.inference_instruct(tts_text, spk_id, instruct_text,
                                                        stream=stream, speed=speed):
                    yield (model.sample_rate, output['tts_speech'].numpy().flatten())
        
        else:
            gr.Warning(f"Unknown mode: {mode}")
            yield (model.sample_rate, np.zeros(model.sample_rate))
    
    except Exception as e:
        logging.error(f'Generation error: {str(e)}')
        gr.Error(f"Generation failed: {str(e)}")
        yield (model.sample_rate, np.zeros(model.sample_rate))


def update_instructions(mode):
    """Update instruction text based on selected mode"""
    instructions = {
        'Pretrained Speaker': '''
### 📋 Usage Guide
1. Select a pretrained speaker voice from the dropdown
2. Enter the text you want to synthesize
3. Click "Generate Audio"

### 💡 Tips
- Pretrained speakers provide consistent, high-quality voices
- Available speakers depend on the loaded model
''',
        'Zero-Shot Voice Cloning': '''
### 📋 Usage Guide
1. Upload or record reference audio (max 30s)
2. Enter the text spoken in the reference audio
3. Enter your target synthesis text
4. Click "Generate Audio"

### 💡 Tips
- Reference audio should be clear with minimal background noise
- For best results, use audio similar in length to your target text
- You can save the cloned voice for future use
''',
        'Cross-Lingual Synthesis': '''
### 📋 Usage Guide
1. Upload reference audio in one language
2. Enter synthesis text in a different language
3. Click "Generate Audio"

### 💡 Tips
- Supports 9+ languages: Chinese, English, Japanese, Korean, German, Spanish, French, Italian, Russian
- The speaker's voice characteristics will be preserved across languages
''',
        'Instruct Control': '''
### 📋 Usage Guide
1. Optionally upload reference audio for voice characteristics
2. Enter instruction text (e.g., "Speak slowly", "Use happy emotion")
3. Enter your synthesis text
4. Click "Generate Audio"

### 💡 Supported Instructions
- Languages: "请用广东话说", "Speak in English"
- Emotions: "happy", "sad", "angry", "excited"
- Speed: "speak faster", "speak slowly"
- Volume: "speak louder", "speak quietly"
- Special effects: [laughter], [breath], [pause]
'''
    }
    return instructions.get(mode, '')


def refresh_spk_list(model):
    """Refresh the speaker list"""
    try:
        spks = model.list_available_spks()
        custom_spks = load_custom_spks(model.model_dir)
        all_spks = spks + custom_spks
        return gr.Dropdown(choices=all_spks, value=all_spks[0] if all_spks else '')
    except Exception as e:
        return gr.Dropdown(choices=[], value='')


def main():
    parser = argparse.ArgumentParser(description='Modern WebUI for CosyVoice3')
    parser.add_argument('--port', type=int, default=7860, help='Server port')
    parser.add_argument('--model_dir', type=str, 
                       default='pretrained_models/Fun-CosyVoice3-0.5B',
                       help='Model directory or ModelScope/HuggingFace repo ID')
    parser.add_argument('--share', action='store_true', help='Create public link')
    parser.add_argument('--server_name', type=str, default='0.0.0.0', help='Server name')
    args = parser.parse_args()
    
    # Initialize model
    print(f"Loading model from {args.model_dir}...")
    try:
        model = AutoModel(model_dir=args.model_dir)
        print(f"✓ Model loaded successfully: {type(model).__name__}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("Please ensure the model is downloaded correctly.")
        sys.exit(1)
    
    # Get initial speaker list
    sft_spks = model.list_available_spks()
    custom_spks = load_custom_spks(args.model_dir)
    all_spks = sft_spks + custom_spks
    
    if not all_spks:
        all_spks = ['']
    
    # Create Gradio interface
    with gr.Blocks(title="CosyVoice3 Studio", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🎙️ CosyVoice3 Studio
        
        **Next-generation Text-to-Speech powered by Large Language Models**
        
        [📚 Documentation](https://github.com/FunAudioLLM/CosyVoice) | 
        [🤗 HuggingFace](https://huggingface.co/FunAudioLLM) | 
        [🔮 ModelScope](https://www.modelscope.cn/models/FunAudioLLM)
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Configuration")
                
                # Mode selection
                mode_dropdown = gr.Radio(
                    choices=['Pretrained Speaker', 'Zero-Shot Voice Cloning', 
                             'Cross-Lingual Synthesis', 'Instruct Control'],
                    value='Zero-Shot Voice Cloning',
                    label='🎯 Inference Mode'
                )
                
                instruction_text = gr.Markdown(update_instructions('Zero-Shot Voice Cloning'))
                
                # Model info
                with gr.Accordion("📦 Model Information", open=False):
                    gr.Markdown(f"**Current Model:** `{args.model_dir}`")
                    gr.Markdown(f"**Sample Rate:** {model.sample_rate} Hz")
                    gr.Markdown(f"**Available Speakers:** {len(all_spks)}")
            
            with gr.Column(scale=2):
                gr.Markdown("### 📝 Input")
                
                # Synthesis text
                tts_text = gr.Textbox(
                    label='📄 Synthesis Text',
                    placeholder='Enter the text you want to synthesize...',
                    lines=3,
                    value='你好，我是通义实验室研发的生成式语音大模型，可以提供舒适自然的语音合成服务。'
                )
                
                # Dynamic controls based on mode
                with gr.Group() as pretrained_group:
                    selected_spk = gr.Dropdown(
                        choices=all_spks,
                        value=all_spks[0] if all_spks else '',
                        label='🎤 Select Pretrained Speaker'
                    )
                    refresh_btn = gr.Button("🔄 Refresh Speaker List", size='sm')
                
                with gr.Group() as zero_shot_group:
                    with gr.Row():
                        prompt_audio_upload = gr.Audio(
                            sources=['upload'],
                            type='filepath',
                            label='📁 Upload Reference Audio (max 30s)',
                            scale=1
                        )
                        prompt_audio_record = gr.Audio(
                            sources=['microphone'],
                            type='filepath',
                            label='🎙️ Record Reference Audio',
                            scale=1
                        )
                    
                    prompt_text = gr.Textbox(
                        label='📝 Prompt Text (text spoken in reference audio)',
                        placeholder='Enter the exact text from the reference audio...',
                        lines=2
                    )
                    
                    with gr.Row():
                        custom_spk_name = gr.Textbox(
                            label='💾 Save as Custom Speaker (optional)',
                            placeholder='Enter name to save this voice...',
                            scale=2
                        )
                        save_spk_btn = gr.Checkbox(label='Save Voice', scale=0, value=False)
                
                with gr.Group() as cross_lingual_group:
                    cl_audio_upload = gr.Audio(
                        sources=['upload'],
                        type='filepath',
                        label='📁 Upload Reference Audio'
                    )
                    cl_audio_record = gr.Audio(
                        sources=['microphone'],
                        type='filepath',
                        label='🎙️ Record Reference Audio'
                    )
                
                with gr.Group() as instruct_group:
                    instruct_audio = gr.Audio(
                        sources=['upload'],
                        type='filepath',
                        label='📁 Optional Reference Audio for Voice Style'
                    )
                    instruct_text = gr.Textbox(
                        label='🎭 Instruction Text',
                        placeholder='E.g., "Speak happily", "用广东话说", "Speak faster"',
                        lines=2
                    )
        
        with gr.Row():
            with gr.Column(scale=3):
                generate_btn = gr.Button("🚀 Generate Audio", variant='primary', size='lg')
            
            with gr.Column(scale=1):
                with gr.Row():
                    seed_btn = gr.Button("🎲", size='sm')
                    seed = gr.Number(label='🌱 Seed', value=0, precision=0)
                
                with gr.Row():
                    stream_mode = gr.Checkbox(label='📡 Stream Mode', value=False)
                    speed_slider = gr.Slider(
                        minimum=0.5, maximum=2.0, value=1.0, step=0.1,
                        label='⚡ Speed'
                    )
        
        # Output
        gr.Markdown("### 🔊 Output")
        audio_output = gr.Audio(
            label='Synthesized Speech',
            autoplay=True,
            streaming=True,
            show_download_button=True
        )
        
        # Examples
        gr.Markdown("### 💡 Examples")
        gr.Examples(
            examples=[
                ['Zero-Shot Voice Cloning', '今天天气真好，适合出去散步。', 
                 '希望你以后能够做的比我还好呦。', './asset/zero_shot_prompt.wav'],
                ['Cross-Lingual Synthesis', '<|en|>And then later on, fully acquiring that company.',
                 '', './asset/cross_lingual_prompt.wav'],
                ['Instruct Control', '在面对挑战时，他展现了非凡的勇气与智慧。',
                 '', 'You are a helpful assistant. 请用四川话说。<|endofprompt|>'],
            ],
            inputs=[mode_dropdown, tts_text, prompt_text, prompt_audio_upload],
            label='Click example to load'
        )
        
        # Event handlers
        def update_mode_visibility(mode):
            """Show/hide controls based on mode"""
            return {
                pretrained_group: gr.update(visible=mode == 'Pretrained Speaker'),
                zero_shot_group: gr.update(visible=mode == 'Zero-Shot Voice Cloning'),
                cross_lingual_group: gr.update(visible=mode == 'Cross-Lingual Synthesis'),
                instruct_group: gr.update(visible=mode == 'Instruct Control'),
            }
        
        mode_dropdown.change(
            fn=update_instructions,
            inputs=[mode_dropdown],
            outputs=[instruction_text]
        ).then(
            fn=update_mode_visibility,
            inputs=[mode_dropdown],
            outputs=[pretrained_group, zero_shot_group, cross_lingual_group, instruct_group]
        )
        
        seed_btn.click(generate_seed, inputs=[], outputs=[seed])
        refresh_btn.click(refresh_spk_list, inputs=[model], outputs=[selected_spk])
        
        # Connect audio inputs between modes
        prompt_audio_upload.change(
            fn=process_audio_upload,
            inputs=[prompt_audio_upload],
            outputs=[gr.Textbox(label='Audio Info')]
        )
        
        # Main generation
        generate_btn.click(
            fn=generate_audio,
            inputs=[tts_text, mode_dropdown, selected_spk, prompt_text, 
                    prompt_audio_upload, prompt_audio_record, instruct_text,
                    custom_spk_name, save_spk_btn, seed, stream_mode, speed_slider,
                    model],
            outputs=[audio_output]
        )
        
        # Initial visibility setup
        demo.load(
            fn=update_mode_visibility,
            inputs=[mode_dropdown],
            outputs=[pretrained_group, zero_shot_group, cross_lingual_group, instruct_group]
        )
    
    # Launch
    print(f"\n🚀 Launching CosyVoice3 Studio on http://{args.server_name}:{args.port}")
    demo.queue(max_size=10, default_concurrency_limit=2)
    demo.launch(
        server_name=args.server_name,
        server_port=args.port,
        share=args.share
    )


if __name__ == '__main__':
    main()

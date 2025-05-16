# Vision Multimodal LLM from Scratch

This project documents the theory and design behind building a vision-language large language model (VLM-LLM) from scratch. It is conceptually inspired by models like PaLI-Gemma, with an emphasis on simplicity, scalability, and unified modeling.

## Objective

To build a transformer-based multimodal model capable of understanding and generating text conditioned on visual inputs. The model should support a range of tasks including image captioning, OCR, VQA, and visual reasoning — all framed as text generation.

## Architecture

- **Vision Encoder**: A ViT-style encoder that converts images into a sequence of patch embeddings.
- **Text Decoder**: An autoregressive transformer that takes both image and text tokens as input.
- **Fusion Strategy**: Late fusion by concatenating image embeddings with text tokens. No cross-attention layers are used.

## Training Strategy

All tasks are unified as text generation problems. Examples include:

- Image → Caption
- Image + Question → Answer
- Image with Text → Transcription

The decoder is trained using standard language modeling objectives over a mixture of such tasks.

## Key Ideas

- Use a single decoder for all tasks, regardless of modality.
- Keep the architecture modular: vision encoder and language decoder are loosely coupled.
- Avoid modality-specific layers; rely on scale and data diversity instead.

## Inspirations

- **PaLI-Gemma**: Late fusion, decoder-only design.
- **Flamingo**: Multimodal task unification.
- **Pix2Struct**: Flattened image representation for structured outputs.

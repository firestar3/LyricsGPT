# LyricsGPT: AI-Powered Lyric Generation

Welcome to **LyricsGPT**, a custom-built autoregressive language model designed from the ground up to generate original song lyrics. Inspired by the GPT family of models, this project demonstrates end-to-end deep learning engineering—from byte-level BPE tokenization to custom transformer blocks and a localized web interface.

## Architecture Overview

At its core, LyricsGPT is a **Transformer-based Decoder-Only** model designed to understand the intricate patterns, rhymes, and structures of song lyrics. 

Key architectural highlights include:
- **Causal Self-Attention**: Ensures the model only attends to past tokens, maintaining strict autoregressive generation.
- **Scaled Dot-Product Attention**: Implements highly optimized attention mechanisms for fast training and inference.
- **Weight Tying**: Shares weights between the token embedding layer and the final output layer, regularizing the model and significantly reducing the parameter count while maintaining performance.
- **GELU Activations**: Utilizes Gaussian Error Linear Units within the Multi-Layer Perceptrons (MLPs) for smoother gradients and better convergence compared to standard ReLUs.
- **Pre-Layer Normalization**: Applies LayerNorm *before* attention and MLP blocks (a crucial modification in modern Transformers like GPT-2) for improved training stability.

## Model Statistics

Despite being designed to run efficiently on local hardware, LyricsGPT packs a punch with a highly optimized parameter distribution:

- **Total Parameters**: **~13.8 Million**
- **Transformer Layers**: 6 Hyper-optimized blocks
- **Attention Heads**: 6 parallel heads per layer
- **Embedding Dimension**: 384
- **Context Window (Block Size)**: 256 tokens 
- **Vocabulary Size**: 8,000 highly curated byte-level BPE tokens
- **Training Techniques**: Cosine learning rate decay, gradient clipping, learning rate warmup, and aggressive weight decay for superior generalization.

This compact yet powerful architecture allows the model to deeply learn the semantic flow of lyrics without suffering from severe overfitting.

## Sample Generation

Here is an unedited example of what LyricsGPT can produce. Notice the emotional resonance and structural repetition characteristic of real songwriting:

> Late at night, I think of you  
> I know I'll never change your ways  
> But you make me feel alive again  
>   
> And I, I’ve got a place to go  
> But I still can't believe it's real  
> Now that I know what I'm feeling about  
>   
> (Just a little bit closer)  
> So I'll always be around the corner  
> And I know how much it hurts  
> When I don't wanna miss you  
> But you make me feel this way again  
> 'Cause I really think of you  
> And I’d love to see you there  
>   
> No matter what  
> No matter how hard it breaks my heart  
> I still choose to believe in you, oh yeah  
> Oh, oh, oh yeah, oh  
> No one ever hurts like you  
> That’s why—  
> You made me feel this way again  
> 'Cause I really think of you  
> And I’d love to see you there  
>   
> You make me feel this spark around  
> Even when you took advantage of me  
> (Just a little bit closer)  
> So I'll always be around the corner  
> And I know how much it hurts  
> When I don't wanna miss you  
> Yet you make my world stay round  
> 'Cause I really think of you  
> And I’d love to see you there, yeah  

## How to Use

A lightweight, zero-dependency (other than PyTorch and Tokenizers) web interface is included to interact with the model seamlessly.

1. Ensure you have the required dependencies installed:
   ```bash
   pip install torch tokenizers
   ```
2. Run the application:
   ```bash
   python app.py
   ```
3. Open your web browser and navigate to:
   ```text
   http://127.0.0.1:8000
   ```
4. Enter a starting prompt and let the model write your next hit song!
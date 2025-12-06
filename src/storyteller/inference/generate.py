"""
Story generation script.

Usage:
    # Interactive mode
    storyteller-generate --checkpoint checkpoints/best_model.pt --interactive

    # Generate from prompt
    storyteller-generate --checkpoint checkpoints/best_model.pt --prompt "Once upon a time"

    # Batch generation
    storyteller-generate --checkpoint checkpoints/best_model.pt --prompts_file prompts.txt --output stories.txt
"""

import argparse
from typing import Optional

import torch
from transformers import PreTrainedTokenizerFast

from storyteller.model import StorytellerModel, ModelConfig
from storyteller.inference.sampling import sample_from_logits


class StoryGenerator:
    """
    Story generator with advanced sampling strategies.
    """

    def __init__(
        self,
        model: StorytellerModel,
        tokenizer: PreTrainedTokenizerFast,
        device: str = "cuda",
    ):
        """
        Initialize generator.

        Args:
            model: Trained StorytellerModel
            tokenizer: Tokenizer
            device: Device to run on
        """
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.9,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.95,
        repetition_penalty: float = 1.1,
        num_return_sequences: int = 1,
    ) -> list[str]:
        """
        Generate story from prompt.

        Args:
            prompt: Text prompt to start generation
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            repetition_penalty: Repetition penalty
            num_return_sequences: Number of sequences to generate

        Returns:
            List of generated stories
        """
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        # Expand for multiple sequences
        if num_return_sequences > 1:
            input_ids = input_ids.repeat(num_return_sequences, 1)

        # Generate
        for _ in range(max_new_tokens):
            # Get logits for next token
            outputs = self.model(input_ids, return_dict=True)
            next_token_logits = outputs["logits"][:, -1, :]

            # Sample next token
            next_token = sample_from_logits(
                next_token_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                generated_tokens=input_ids,
            )

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Check for EOS
            if self.tokenizer.eos_token_id is not None:
                eos_reached = (next_token == self.tokenizer.eos_token_id).all()
                if eos_reached:
                    break

        # Decode
        generated_texts = []
        for seq in input_ids:
            text = self.tokenizer.decode(seq, skip_special_tokens=True)
            generated_texts.append(text)

        return generated_texts

    def interactive_mode(self):
        """
        Interactive story generation mode.
        """
        print("\n" + "=" * 60)
        print("Interactive Story Generation")
        print("=" * 60)
        print("\nCommands:")
        print("  /quit - Exit interactive mode")
        print("  /temp <value> - Set temperature (default: 0.9)")
        print("  /topk <value> - Set top-k (default: 50)")
        print("  /topp <value> - Set top-p (default: 0.95)")
        print("  /length <value> - Set max length (default: 512)")
        print("=" * 60 + "\n")

        # Default parameters
        temperature = 0.9
        top_k = 50
        top_p = 0.95
        max_length = 512

        while True:
            try:
                prompt = input("\nEnter prompt (or command): ").strip()

                if not prompt:
                    continue

                # Handle commands
                if prompt.startswith("/"):
                    if prompt == "/quit":
                        print("Goodbye!")
                        break

                    elif prompt.startswith("/temp"):
                        try:
                            temperature = float(prompt.split()[1])
                            print(f"Temperature set to {temperature}")
                        except (ValueError, IndexError):
                            print("Usage: /temp <value>")

                    elif prompt.startswith("/topk"):
                        try:
                            top_k = int(prompt.split()[1])
                            print(f"Top-k set to {top_k}")
                        except (ValueError, IndexError):
                            print("Usage: /topk <value>")

                    elif prompt.startswith("/topp"):
                        try:
                            top_p = float(prompt.split()[1])
                            print(f"Top-p set to {top_p}")
                        except (ValueError, IndexError):
                            print("Usage: /topp <value>")

                    elif prompt.startswith("/length"):
                        try:
                            max_length = int(prompt.split()[1])
                            print(f"Max length set to {max_length}")
                        except (ValueError, IndexError):
                            print("Usage: /length <value>")

                    else:
                        print("Unknown command")

                    continue

                # Generate story
                print("\nGenerating story...")
                print("-" * 60)

                stories = self.generate(
                    prompt=prompt,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )

                for i, story in enumerate(stories, 1):
                    if len(stories) > 1:
                        print(f"\nStory {i}:")
                    print(story)

                print("-" * 60)

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break

            except Exception as e:
                print(f"\nError: {e}")
                continue


def load_model_from_checkpoint(
    checkpoint_path: str,
    tokenizer_path: str,
    device: str = "cuda",
) -> tuple[StorytellerModel, PreTrainedTokenizerFast]:
    """
    Load model and tokenizer from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint
        tokenizer_path: Path to tokenizer
        device: Device to load on

    Returns:
        Tuple of (model, tokenizer)
    """
    # Load tokenizer
    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)

    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create model from config
    config = ModelConfig(**checkpoint["config"])
    model = StorytellerModel(config)

    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"✓ Model loaded (step {checkpoint.get('global_step', 'unknown')})")

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Generate stories with Storyteller")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="data/tokenizers/storyteller-tokenizer",
        help="Path to tokenizer",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on",
    )

    # Generation modes
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive generation mode",
    )
    mode_group.add_argument(
        "--prompt",
        type=str,
        help="Single prompt to generate from",
    )
    mode_group.add_argument(
        "--prompts_file",
        type=str,
        help="File with prompts (one per line)",
    )

    # Generation parameters
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--num_return_sequences", type=int, default=1)

    # Output
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for generated stories",
    )

    args = parser.parse_args()

    # Load model and tokenizer
    model, tokenizer = load_model_from_checkpoint(
        args.checkpoint,
        args.tokenizer_path,
        args.device,
    )

    # Create generator
    generator = StoryGenerator(model, tokenizer, args.device)

    # Run generation based on mode
    if args.interactive:
        # Interactive mode
        generator.interactive_mode()

    elif args.prompt:
        # Single prompt
        print(f"\nPrompt: {args.prompt}")
        print("-" * 60)

        stories = generator.generate(
            prompt=args.prompt,
            max_new_tokens=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            num_return_sequences=args.num_return_sequences,
        )

        for i, story in enumerate(stories, 1):
            if args.num_return_sequences > 1:
                print(f"\nStory {i}:")
            print(story)
            print("-" * 60)

        # Save to file if specified
        if args.output:
            with open(args.output, "w") as f:
                for story in stories:
                    f.write(story + "\n\n")
            print(f"\n✓ Stories saved to {args.output}")

    elif args.prompts_file:
        # Batch generation
        with open(args.prompts_file, "r") as f:
            prompts = [line.strip() for line in f if line.strip()]

        print(f"\nGenerating from {len(prompts)} prompts...")

        all_stories = []
        for i, prompt in enumerate(prompts, 1):
            print(f"\n[{i}/{len(prompts)}] {prompt[:50]}...")

            stories = generator.generate(
                prompt=prompt,
                max_new_tokens=args.max_length,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                num_return_sequences=args.num_return_sequences,
            )

            all_stories.extend(stories)

        # Save to file
        if args.output:
            with open(args.output, "w") as f:
                for story in all_stories:
                    f.write(story + "\n\n" + "=" * 60 + "\n\n")
            print(f"\n✓ {len(all_stories)} stories saved to {args.output}")

    else:
        print(
            "Please specify a generation mode: --interactive, --prompt, or --prompts_file"
        )
        parser.print_help()


if __name__ == "__main__":
    main()

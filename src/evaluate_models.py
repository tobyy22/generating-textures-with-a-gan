import argparse
import wandb

from src.DCWGAN.train_dcwgan_renderer import DCWGANRendererTrainer
from src.WGANUnet.train import WGANUnetTrainer
from src.PytorchMRIUnet.train import PytorchUnetTrainer
from src.PytorchMRIUnet.train_with_similarity_loss import PytorchUnetSimilarityLossTrainer


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate models.")
    parser.add_argument("--trainer", type=str, required=True, choices=[
        "DCWGANRenderer", "WGANUnet", "PytorchUnet", "PytorchUnetSimilarityLoss"
    ], help="Specify the trainer to use.")
    parser.add_argument("--evaluate_fid_score", action="store_true", help="Evaluate FID score.")
    parser.add_argument("--visualize_results", action="store_true", help="Visualize results.")
    parser.add_argument("--visualized_object_id", type=int, default=1433, help="ID of the object to visualize.")
    parser.add_argument("--dataset_path", type=str, default="./3Dataset", help="Path to the dataset.")

    args = parser.parse_args()

    # Select the trainer based on the command-line argument
    if args.trainer == "DCWGANRenderer":
        agent = DCWGANRendererTrainer(
            name="model",
            models_dir="my_data/WGANRenderer",
            image_size=64,
            texture_size=64,
            wgan=True,
            dataset_path=args.dataset_path,
        )
    elif args.trainer == "WGANUnet":
        agent = WGANUnetTrainer(
            name="model",
            models_dir="my_data/WGANUnet",
            pre_generated_uv_textures_dir="my_data/uv_textures_64",
            image_size=64,
            texture_size=64,
            uv_textures_pregenerated=True,
            dataset_path=args.dataset_path,
        )
    elif args.trainer == "PytorchUnet":
        agent = PytorchUnetTrainer(
            name="model_128",
            models_dir="my_data/PytorchMRIUnet",
            image_size=256,
            texture_size=128,
            pre_generated_uv_textures_dir="my_data/uv_textures_128",
            uv_textures_pregenerated=True,
            dataset_path=args.dataset_path,
        )
    elif args.trainer == "PytorchUnetSimilarityLoss":
        agent = PytorchUnetSimilarityLossTrainer(
            name="model_128_with_similarity_loss",
            models_dir="my_data/PytorchMRIUnet",
            image_size=256,
            texture_size=128,
            pre_generated_uv_textures_dir="my_data/uv_textures_128",
            uv_textures_pregenerated=True,
            dataset_path=args.dataset_path,
        )

    # Load and initialize the dataset
    agent.load()
    agent.init_dataset()

    # Evaluate FID score or visualize results
    if args.visualize_results:
        for _ in range(20):
            data = agent.visualize_data(args.visualized_object_id)
            wandb.log(data)
    elif args.evaluate_fid_score:
        fid_score = agent.compute_fid_score()
        wandb.log(fid_score)
        print(fid_score)


if __name__ == "__main__":
    main()

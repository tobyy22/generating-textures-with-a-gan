from src.texture_optimization.TextureOptimizer import TextureOptimizationGridSearch


if __name__ == "__main__":
    """
    This runs the experiment with optimizing cow. Optimization progress can be viewed in wandb. Process runs indefinitely. 
    """
    texture_optimizer = TextureOptimizationGridSearch(dataset_path='./my_data/texture_prediction/dataset_just_cow', 
                                                      time_to_optimize=float('inf'),
                                                      param_grid = {
                                                            "learning_rate": [1.0],
                                                            "loss_rgb": [15000],
                                                            "loss_penalization": [0.15],
                                                            "momentum": [0.9],
                                                        }
                                                    )
    # We will simply run grid search with just one object and one parameter combination. 
    texture_optimizer.grid_search()